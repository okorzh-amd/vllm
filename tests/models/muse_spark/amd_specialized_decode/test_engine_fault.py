# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-stop end to end: a TP rank that stalls inside a served Muse Spark
specialized decode step fails a request, and the engine stops instead of
hanging: every TP rank must stop at the same step (``guards.FailStop``), or the
ranks that went on would wait in the next collective for ever.

The model is the real checkpoint's config cut to 4 layers with dummy weights:
layer 0 keeps BF16 experts (Quark's ``exclude``), layers 1..3 are kernel
layers. ``--load-format dummy`` never calls ``load_weights``, so this also
covers the residual-constant materialization in
``process_weights_after_loading``.

Two stalls: K2 of the step's first kernel layer (Phase A), and K1 of that layer
with Phase B1's front on (``additional_config`` "mono_front": "k1"; the stock
attention core and K2's Phase A core keep the variant independent of B2 / C):
its waits are rank-local, so the other ranks give up in the K2 that waits on
the stalled rank's all-reduce partials."""

import json
import os
import time

import pytest
import requests
import torch

from tests.utils import RemoteOpenAIServer, multi_gpu_test

MODEL_ID = "amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4"
# the HF id, resolved through HF_HUB_CACHE (offline in CI); a local snapshot
# directory may be given instead
MODEL = os.environ.get("MUSE_SPARK_MODEL", MODEL_ID)
STALLED = 3
FAULT_EPOCH = 256  # a served step well past the self-test, warm-ups and capture
# (the runner's fault attribute, the stalled launch's slot, additional_config):
# the step's first K2 launch (spec.k2_slot(1)) or the step's first K1 launch
# (spec.k1_slot(1)), which runs right after the epoch bump
FAULTS = {
    "k2": ("fault", 128 + 1, {}),
    "k1": ("fault_k1", 1, {"mono_front": "k1"}),
}
STARTUP_TIMEOUT = 1800
FAULT_DEADLINE = 900
EXIT_TIMEOUT = 300


def _on_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


needs_gfx950 = pytest.mark.skipif(not _on_gfx950(), reason="needs MI355X (gfx950)")

# Builds every rank's K2 (or K1) with the fault hook (``K2Build.fault_*``,
# ``K1Build.fault_*``): rank STALLED stalls past every in-kernel wait's budget in
# the launch of its slot (FAULTS) at epoch FAULT_EPOCH. Patched into
# ``MonoRunner.fault`` (``fault_k1``) as the runner module loads, in every vLLM
# process. ``SpecializedDecode.after_capture`` (the end of
# the startup) also appends the rank's device epoch to EPOCHS, so the test can
# tell a fault that fired during the startup from a lost one; ``FailStop``'s
# stop appends "rank epoch" to STOPS before it raises, so the test can tell the
# fail-stop from any other engine death.
_SITECUSTOMIZE = """\
import importlib.abc
import importlib.machinery
import sys

_RUNNER = "vllm.models.muse_spark.amd.specialized_decode.runtime.runner"
_DECODE = "vllm.models.muse_spark.amd.specialized_decode.runtime.decode"
_GUARDS = "vllm.models.muse_spark.amd.specialized_decode.runtime.guards"


def _patch_runner(module):
    def fault(runner):
        return {{
            "fault_rank": {rank},
            "fault_epoch": {epoch},
            "fault_slot": {slot},
            "fault_ticks": {ticks},
        }}

    module.MonoRunner.{attr} = property(fault, lambda runner, value: None)


def _patch_decode(module):
    after_capture = module.SpecializedDecode.after_capture

    def logged(decode):
        after_capture(decode)
        module._MUSE_FAULT_RUNNER = decode.runner  # FailStop's epoch, below
        with open({epochs!r}, "a") as f:
            f.write(f"{{int(decode.runner.epoch.item())}}\\n")

    module.SpecializedDecode.after_capture = logged


def _patch_guards(module):
    after_step = module.FailStop.after_step

    def logged(failstop):
        try:
            after_step(failstop)
        except RuntimeError:
            decode = sys.modules.get(_DECODE)
            runner = getattr(decode, "_MUSE_FAULT_RUNNER", None)
            epoch = -1 if runner is None else int(runner.epoch.item())
            with open({stops!r}, "a") as f:
                f.write(f"{{failstop.rank}} {{epoch}}\\n")
            raise

    module.FailStop.after_step = logged


_PATCHES = {{_RUNNER: _patch_runner, _DECODE: _patch_decode, _GUARDS: _patch_guards}}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _PATCHES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        exec_module = spec.loader.exec_module

        def patched_exec(module):
            exec_module(module)
            _PATCHES[name](module)

        spec.loader.exec_module = patched_exec
        return spec


sys.meta_path.insert(0, _Finder())
"""

HF_OVERRIDES = {
    "text_config": {
        "num_hidden_layers": 4,
        "layer_types": [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
        ],
        "layer_rope_theta": [500000.0, 0, 500000.0, 500000.0],
        "mlp_layer_types": ["sparse", "sparse", "sparse", "sparse"],
    }
}

SERVER_ARGS = [
    "--trust-remote-code",
    "--language-model-only",
    "--load-format",
    "dummy",
    "--hf-overrides",
    json.dumps(HF_OVERRIDES),
    "--tensor-parallel-size",
    "8",
    "--max-model-len",
    "8192",
    "--max-num-seqs",
    "2",
    "--attention-backend",
    "ROCM_AITER_FA",
    "--block-size",
    "16",
    "--compilation-config",
    json.dumps(
        {
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": [1, 2, 4, 8],
        }
    ),
    "--monokernel",
]


@needs_gfx950
@multi_gpu_test(num_gpus=8)
@pytest.mark.parametrize("kernel", list(FAULTS))
def test_a_stalled_rank_stops_the_engine(tmp_path, kernel):
    from vllm.models.muse_spark.amd.specialized_decode.mono.common.sync import (
        WAIT_TICKS,
    )

    attr, fault_slot, extra = FAULTS[kernel]
    site = tmp_path / "mono_fault"
    site.mkdir()
    epochs = tmp_path / "epochs.txt"
    stops = tmp_path / "stops.txt"
    (site / "sitecustomize.py").write_text(
        _SITECUSTOMIZE.format(
            attr=attr,
            rank=STALLED,
            epoch=FAULT_EPOCH,
            slot=fault_slot,
            ticks=WAIT_TICKS * 3 // 2,
            epochs=str(epochs),
            stops=str(stops),
        )
    )
    env = {
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(site), os.getenv("PYTHONPATH")])
        ),
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_ROCM_USE_AITER": "1",
        "VLLM_MOE_SKIP_PADDING": "1",
    }
    body = {
        "model": MODEL,
        "prompt": "hello " * 64,
        "max_tokens": 128,
        "temperature": 0,
        "ignore_eos": True,
    }
    args = SERVER_ARGS
    if extra:
        args = [*args, "--additional-config", json.dumps(extra)]
    with RemoteOpenAIServer(
        MODEL, args, env_dict=env, max_wait_seconds=STARTUP_TIMEOUT
    ) as server:
        # the startup ended (after_capture on every rank) below the fault's step
        ready = [int(e) for e in epochs.read_text().split()] if epochs.exists() else []
        assert ready, "the specialized decode never finished a CUDA-graph capture"
        assert max(ready) < FAULT_EPOCH, (
            f"the startup reached epoch {max(ready)} >= {FAULT_EPOCH}: raise "
            "FAULT_EPOCH"
        )
        url = server.url_root
        served = 0
        deadline = time.monotonic() + FAULT_DEADLINE
        while time.monotonic() < deadline:
            try:
                r = requests.post(f"{url}/v1/completions", json=body, timeout=300)
            except requests.ConnectionError:
                break
            except requests.Timeout:
                pytest.fail("the engine hung after the injected stall")
            if r.status_code != 200:
                break
            served += 1
        else:
            # the stall only exists inside the kernels: also catches a decode
            # that never ran them
            pytest.fail(f"no request failed in {FAULT_DEADLINE}s: the fault is lost")

        # one request is 128 single-row decode steps: the fault fires in the
        # second or third request
        assert served > 0, "the engine failed before the injected step"
        # the failed step stops the engine and the server exits
        assert server.proc.wait(timeout=EXIT_TIMEOUT) is not None
    # ... and it was the fail-stop: every TP rank stopped, at or past the step
    # of the injected stall (not some other engine death)
    lines = stops.read_text().split() if stops.exists() else []
    seen = {int(r): int(e) for r, e in zip(lines[::2], lines[1::2])}
    assert sorted(seen) == list(range(8)), f"FailStop stopped ranks {sorted(seen)}"
    assert all(e >= FAULT_EPOCH for e in seen.values()), seen


@pytest.mark.parametrize("kernel", list(FAULTS))
def test_fault_hook_renders(tmp_path, kernel):
    """CPU: the injected sitecustomize of each variant is valid Python that
    patches the runner attribute its build key reads, with a fault slot the key
    accepts (``K1Build`` / ``K2Build`` validate it)."""
    from vllm.models.muse_spark.amd.specialized_decode.mono import spec

    attr, fault_slot, extra = FAULTS[kernel]
    code = _SITECUSTOMIZE.format(
        attr=attr,
        rank=STALLED,
        epoch=FAULT_EPOCH,
        slot=fault_slot,
        ticks=7,
        epochs=str(tmp_path / "e"),
        stops=str(tmp_path / "s"),
    )
    compile(code, "sitecustomize.py", "exec")
    assert f"module.MonoRunner.{attr} = property(" in code
    fault = dict(fault_rank=STALLED, fault_epoch=FAULT_EPOCH, fault_slot=fault_slot)
    if kernel == "k1":
        spec.K1Build(tokens=1, rope=False, fault_ticks=7, **fault)
        assert fault_slot == spec.k1_slot(1) and extra == {"mono_front": "k1"}
    else:
        spec.K2Build(tokens=1, fault_ticks=7, **fault)
        assert fault_slot == spec.k2_slot(1) and extra == {}
