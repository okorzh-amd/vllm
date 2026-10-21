# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The stock gfx950 code Phase B reproduces bit for bit, pinned on CPU.

K1's epilogue (DESIGN_BC 1.3, 5.6) copies the rounding sequence of vLLM's fused
QK-norm / RoPE / KV insert as the compiled gfx950 code object runs it, and the
gated core load (1.4, 6.2) copies the gfx950 reduction tree Triton builds for
FLA's ``layer_norm_gated_fwd_kernel``. Both were read from the ISA; a compiler
or source bump that changes either would silently turn the GPU probes'
``torch.equal`` into an ulp mismatch. These tests fail first, without a GPU:

- the insert: the gfx950 code object is read out of vLLM's own
  ``_C_stable_libtorch.abi3.so`` (its ``.hip_fatbin``: uncompressed clang
  offload bundles), and every bf16 / HD 64 / ``kAuto`` instance that inserts
  (HEADS_PER_GROUP 1, 2, 4 x RoPE / NoPE; any cache and RoPE-table type) keeps
  the per-head op signature: 1 ``v_sub_f32`` and 7 ``v_pk_fma_f32`` (RoPE
  only: the uneven pair table), 3 ``ds_bpermute`` (the 8-lane group reduce),
  1 ``v_fmamk_f32 ... 0x3c800000`` (``fma(sumsq, 2^-6, eps)``) and 1
  ``v_rsq_f32``; and the two decode instances (HEADS_PER_GROUP 1) are pinned
  instruction for instruction, operands included, to the code 1.3 was read
  from;
- the gate: the kernel compiled offline for gfx950 with the served launch's
  specialization (16-byte aligned pointers: two columns a lane) has the leaf
  ``v_pk_mul_f32`` + ``v_add_f32``, then the butterfly ``v_permlane16_swap``,
  row_ror 8, row_half_mirror / quad_perm 3210, quad_perm 2301, 1032, a raw
  ``v_sqrt_f32`` and the IEEE divide.
"""

import ast
import hashlib
import importlib.util
import os
import struct
import subprocess
from pathlib import Path

import pytest
import regex as re

BUNDLE = b"__CLANG_OFFLOAD_BUNDLE__"
GFX950 = "hipv4-amdgcn-amd-amdhsa--gfx950"
KERNEL = b"fusedMuseSparkQKNormRopeKVInsertKernel"
# <c10::BFloat16, cache_t, kAuto, rope_t, HD 64, HEADS_PER_GROUP, rope, insert>
INSTANCE = re.compile(
    r"_ZN4vllm20muse_spark_fused_ops38fusedMuseSparkQKNormRopeKVInsertKernel"
    r"IN3c108BFloat16E(?P<cache>\w+?)LNS_18Fp8KVCacheDataTypeE0E"
    r"(?P<table>f|S3_)Li64ELi(?P<hpg>[124])ELb(?P<rope>[01])ELb1E\w*"
)
# the instances decode and prefill run with bf16 KV (cache_t = __hip_bfloat16)
# and the fp32 RoPE table (DESIGN_BC 1.3)
SERVED_CACHE, SERVED_TABLE = "14__hip_bfloat16", "f"
# The two instances decode runs (HEADS_PER_GROUP 1), RoPE -> the sha256 of
# their disassembly (``_normalized``): DESIGN_BC 1.3's table was read operand
# by operand from exactly this code (scratch_bcdesign/muse_ins.dis:252304 and
# :252968), and the op counts alone do not see which product a fma fuses.
DECODE_DIGESTS = {
    True: "03484cdef14a0e0bdc9e09dac688833d9f452f54a266475b7137873a01ca5645",
    False: "737a8f7048634a5f82f4f632439a8a10883486d68055582aadc599ce08ac46ac",
}


def _llvm_objdump() -> str:
    """ROCm's ``llvm-objdump``, or "" (none found)."""
    dirs = [
        Path(os.environ[e]) / "llvm/bin"
        for e in ("ROCM_PATH", "ROCM_HOME")
        if os.environ.get(e)
    ]
    dirs.append(Path("/opt/rocm/llvm/bin"))
    sdk = importlib.util.find_spec("_rocm_sdk_core")
    if sdk is not None and sdk.submodule_search_locations:
        dirs += [Path(p) / "lib/llvm/bin" for p in sdk.submodule_search_locations]
    found = [d / "llvm-objdump" for d in dirs if (d / "llvm-objdump").exists()]
    return str(found[0]) if found else ""


def _elf_sections(read) -> dict[str, tuple[int, int, int]]:
    """An ELF64 image's sections, ``read(offset, n)`` its bytes: name ->
    (offset, size, entry size)."""
    head = read(0, 64)
    if head[:4] != b"\x7fELF" or head[4] != 2:
        raise ValueError("not an ELF64 image")
    (shoff,) = struct.unpack_from("<Q", head, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", head, 0x3A)
    raw = read(shoff, shentsize * shnum)
    headers = []
    for i in range(shnum):
        name, _, _, _, off, size, _, _, _, entsize = struct.unpack_from(
            "<IIQQQQIIQQ", raw, i * shentsize
        )
        headers.append((name, off, size, entsize))
    _, stroff, strsize, _ = headers[shstrndx]
    names = read(stroff, strsize)
    return {
        names[n : names.index(b"\0", n)].decode(): (off, size, entsize)
        for n, off, size, entsize in headers
    }


def _fatbin(so: Path) -> bytes:
    """The ``.hip_fatbin`` section of ``so`` (reading only what it needs)."""
    with open(so, "rb") as f:

        def read(off, n):
            f.seek(off)
            return f.read(n)

        sections = _elf_sections(read)
        if ".hip_fatbin" not in sections:
            return b""
        off, size, _ = sections[".hip_fatbin"]
        return read(off, size)


def _code_objects(fatbin: bytes, triple: str):
    """Every code object for ``triple`` in the fatbin's uncompressed bundles
    (clang's OffloadBundler layout: magic, u64 count, then per entry u64
    offset, u64 size, u64 triple length, the triple)."""
    for m in re.finditer(re.escape(BUNDLE), fatbin):
        base = m.start()
        (n,) = struct.unpack_from("<Q", fatbin, base + len(BUNDLE))
        pos = base + len(BUNDLE) + 8
        for _ in range(n):
            off, size, tlen = struct.unpack_from("<QQQ", fatbin, pos)
            name = fatbin[pos + 24 : pos + 24 + tlen].decode()
            pos += 24 + tlen
            if name == triple and size:
                yield fatbin[base + off : base + off + size]


def _symbols(co: bytes) -> list[str]:
    """The names in a code object's ``.symtab``."""
    sec = _elf_sections(lambda off, n: co[off : off + n])
    off, size, entsize = sec[".symtab"]
    stroff = sec[".strtab"][0]
    names = []
    for i in range(size // entsize):
        (name,) = struct.unpack_from("<I", co, off + i * entsize)
        if name:
            end = co.index(b"\0", stroff + name)
            names.append(co[stroff + name : end].decode())
    return names


@pytest.fixture(scope="module")
def insert_instances(tmp_path_factory) -> dict[tuple, str]:
    """(cache, table, hpg, rope) -> the disassembly of that instance."""
    import vllm

    so = Path(vllm.__file__).parent / "_C_stable_libtorch.abi3.so"
    if not so.exists():
        pytest.skip(f"no {so.name} (not a ROCm build of vLLM)")
    objdump = _llvm_objdump()
    if not objdump:
        pytest.skip("no ROCm llvm-objdump")
    fatbin = _fatbin(so)
    cos = [co for co in _code_objects(fatbin, GFX950) if KERNEL in co]
    if not cos:
        if b"CCOB" in fatbin:
            pytest.skip("compressed offload bundles: not read by this pin")
        pytest.skip(f"{so.name} carries no gfx950 code object with the insert")
    assert len(cos) == 1, "the insert in two gfx950 code objects"
    syms = sorted({s for s in _symbols(cos[0]) if INSTANCE.fullmatch(s)})
    path = tmp_path_factory.mktemp("insert") / "gfx950.co"
    path.write_bytes(cos[0])
    dis = subprocess.run(
        [
            objdump,
            "-d",
            "--disassemble-symbols=" + ",".join(syms),
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    out = {}
    for body in re.split(r"\n(?=[0-9a-f]+ <)", dis):
        m = INSTANCE.search(body.split("\n", 1)[0])
        if m:
            key = (m["cache"], m["table"], int(m["hpg"]), m["rope"] == "1")
            out[key] = body
    assert len(out) == len(syms)
    return out


def test_the_served_insert_instances_exist(insert_instances):
    """bf16 KV and the fp32 RoPE table: HEADS_PER_GROUP 1 (decode, S <= 8;
    DESIGN_BC 1.3), 2 and 4 (prefill chunks), RoPE and NoPE."""
    served = {
        (hpg, rope)
        for cache, table, hpg, rope in insert_instances
        if (cache, table) == (SERVED_CACHE, SERVED_TABLE)
    }
    assert served == {(h, r) for h in (1, 2, 4) for r in (False, True)}


def _normalized(body: str) -> str:
    """An instance's disassembly without its symbol line, addresses,
    encodings and branch-target symbols: the instructions and operands."""
    out = []
    for line in body.split("\n")[1:]:
        line = re.sub(r"\s*//.*$", "", line)  # address: encoding
        line = re.sub(r"\s*<[^>]*>$", "", line).strip()  # branch target
        if line and line != "...":
            out.append(re.sub(r"\s+", " ", line))
    return "\n".join(out)


def test_decode_instances_are_the_read_code(insert_instances):
    """The decode instances are, instruction for instruction, the code 1.3's
    rounding table was read from. On a mismatch, re-read 1.3 step 6 (which
    product each ``v_pk_fma_f32`` fuses, ``v_sub_f32``'s operands) from the
    new disassembly before updating the digest: K1's epilogue copies it."""
    for rope, want in DECODE_DIGESTS.items():
        body = insert_instances[(SERVED_CACHE, SERVED_TABLE, 1, rope)]
        got = hashlib.sha256(_normalized(body).encode()).hexdigest()
        assert got == want, (f"rope={rope}", got)


def test_insert_op_signature_per_head(insert_instances):
    """The per-head signature of every bf16 HD 64 kAuto inserting instance,
    times HEADS_PER_GROUP."""

    def count(body, pat):
        return len(re.findall(pat, body))

    for (cache, table, hpg, rope), body in insert_instances.items():
        what = (cache, table, hpg, rope)
        # 1.3 step 6: dim 0's v_sub of two rounded products, the 7 others fma
        assert count(body, r"\bv_sub_f32") == hpg * rope, what
        assert count(body, r"\bv_pk_fma_f32") == 7 * hpg * rope, what
        # 1.3 step 3: groupReduceSum<8>, xor 4, 2, 1
        assert count(body, r"\bds_bpermute_b32") == 3 * hpg, what
        # 1.3 step 4: rsq(fma(sumsq, 2^-6, eps))
        assert count(body, r"\bv_fmamk_f32\b[^\n]*0x3c800000") == hpg, what
        assert count(body, r"\bv_rsq_f32") == hpg, what
        # 1.3 step 7: one bf16 rounding (no fp32 -> bf16 by other means)
        assert count(body, r"\bv_cvt_pk_bf16_f32") > 0, what


def _fla_kernel(name: str, tmp_path: Path):
    """FLA kernel ``name`` (decorators included) from vLLM's ``kda.py`` (the
    ``rms_norm_gated`` the model imports), imported against the real Triton:
    vLLM's own import hands out placeholders when no GPU driver is loaded."""
    import vllm

    src = Path(vllm.__file__).parent / "third_party/flash_linear_attention/ops/kda.py"
    text = src.read_text()
    lines = text.splitlines()
    [node] = [
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    first = min(d.lineno for d in node.decorator_list)
    body = "\n".join(lines[first - 1 : node.end_lineno])
    path = tmp_path / "fla_gate.py"
    path.write_text(f"import triton\nimport triton.language as tl\n\n{body}\n")
    spec = importlib.util.spec_from_file_location("fla_gate", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


# The specialization the served ``rms_norm_gated`` launch compiles (Triton's
# cache key, read on the GPU: scratch_bcdbg/out/gate_rt_1.key): x, g, y, rstd
# 16-byte aligned and in buffer range ("DS") and T = 16 S divisible by 16 ("D");
# kda.py's kernel takes g_stride_n (2304: q | k | v | o_gate per token) as a
# constexpr. The pointers' divisibility picks the layout and so the
# reduction tree: unspecialized, the kernel has another tree (a column a lane,
# row_shr / row_bcast / readlane), which DESIGN_BC 1.4 first analysed and which
# differs from the served one on ~2e-4 of the rows.
SERVED_GATE_SPECIALIZATION = {
    "x": "DS",
    "g": "DS",
    "y": "DS",
    "rstd": "DS",
    "T": "D",
}


def test_gate_reduction_tree(tmp_path):
    """FLA's ``layer_norm_gated_fwd_kernel`` as ``rms_norm_gated`` launches it
    for Muse (BT 16, BD 64, H 16, 8 warps, RMS, sigmoid, no weight), compiled
    offline for gfx950 with the served launch's specialization (the offline
    compile is then the runtime-compiled kernel instruction for instruction,
    checked on the GPU 2026-10-21): the reduction tree and divides of
    DESIGN_BC 1.4, in order. A row over 32 lanes, two adjacent columns a lane:
    the leaf RN(x_2c^2) + RN(x_2c+1^2) (``v_pk_mul_f32``, ``v_add_f32``, no
    fma), then lanes xor 16 (``v_permlane16_swap``), 8 (row_ror 8), 4
    (row_half_mirror + quad_perm 3210), 2, 1 (quad_perm); fma(tot, 2^-6, eps),
    a raw ``v_sqrt_f32`` and the IEEE divide. test_probe_gate and
    test_probe_harness compare with the runtime-compiled kernel."""
    triton = pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.runtime.jit import JITFunction

    try:
        from triton.backends.amd.compiler import HIPBackend
    except Exception as e:  # no HIP backend in this build of Triton
        pytest.skip(f"no Triton HIP backend here: {e}")

    fn = _fla_kernel("layer_norm_gated_fwd_kernel", tmp_path)
    while not isinstance(fn, JITFunction):
        fn = fn.fn
    const = dict(
        w=None,
        b=None,
        residual=None,
        residual_out=None,
        mean=None,
        H=16,
        D=64,
        BT=16,
        BD=64,
        ACTIVATION="sigmoid",
        IS_RMS_NORM=True,
        STORE_RESIDUAL_OUT=False,
        HAS_RESIDUAL=False,
        HAS_WEIGHT=False,
        HAS_BIAS=False,
        g_stride_n=2304,
    )
    sig = dict(
        x="*bf16",
        g="*bf16",
        y="*bf16",
        rstd="*fp32",
        eps="fp32",
        T="i32",
    )
    sig.update((n, "constexpr") for n in const)
    assert set(sig) == set(fn.arg_names), fn.arg_names
    attrs = {
        (fn.arg_names.index(n),): HIPBackend.parse_attr(d)
        for n, d in SERVED_GATE_SPECIALIZATION.items()
    }
    try:
        compiled = triton.compile(
            ASTSource(fn, signature=sig, constexprs=const, attrs=attrs),
            target=GPUTarget("hip", "gfx950", 64),
            options={"num_warps": 8},
        )
    except Exception as e:  # no HIP backend in this build of Triton
        pytest.skip(f"Triton cannot compile for gfx950 here: {e}")
    asm = compiled.asm["amdgcn"]
    # one row a lane half (BT 16 over 8 waves, 32 lanes a row): every step once
    steps = [
        r"v_pk_mul_f32 ",  # RN(x_2c^2), RN(x_2c+1^2)
        r"v_add_f32_e32 ",  # the leaf s_c
        r"v_permlane16_swap_b32",  # xor 16
        r"v_add_f32_e32 ",
        r"v_add_f32_dpp v\d+, v\d+, v\d+ row_ror:8 ",  # xor 8
        r"v_mov_b32_dpp v\d+, v\d+ row_half_mirror ",  # xor 4 ...
        r"v_add_f32_dpp v\d+, v\d+, v\d+ quad_perm:\[3,2,1,0\] ",  # ... with this
        r"v_add_f32_dpp v\d+, v\d+, v\d+ quad_perm:\[2,3,0,1\] ",  # xor 2
        r"v_add_f32_dpp v\d+, v\d+, v\d+ quad_perm:\[1,0,3,2\] ",  # xor 1
        r"v_fmamk_f32 v\d+, v\d+, 0x3c800000, ",  # fma(tot, 2^-6, eps)
        r"v_sqrt_f32",  # raw: no denormal scaling around it
        r"v_div_fixup_f32",  # rstd = 1 / sqrt: the IEEE divide
        r"v_exp_f32",  # the sigmoid's exp2
        r"v_div_fixup_f32",  # ... and its IEEE divide
    ]
    pos = 0
    for step in steps:
        m = re.compile(step).search(asm, pos)
        assert m, f"{step!r} missing after offset {pos}"
        pos = m.end()
    for op, n in (
        ("v_permlane16_swap_b32", 1),
        ("row_ror:8", 1),
        ("row_half_mirror", 1),
        ("v_sqrt_f32", 1),
        ("v_div_fixup_f32", 3),
        ("row_shr", 0),  # the unspecialized instance's tree
        ("row_bcast", 0),
        ("v_readlane_b32", 0),
    ):
        assert asm.count(op) == n, (op, asm.count(op))
    # the leaf is a separate multiply and add: no fma before fma(tot, 2^-6, eps)
    tree = asm[: asm.index("v_fmamk_f32")]
    assert not re.search(r"\bv_(?:pk_)?fma|\bv_fmac|\bv_mad|\bv_mac", tree)
    # rstd: no fast reciprocal square root in place of sqrt + divide
    assert "v_rsq_f32" not in asm
    # ... and the sqrt is raw: no denormal range scaling (2^32 / 2^-96 around
    # it), as LLVM's correctly rounded f32 sqrt would emit
    assert "0x4f800000" not in asm and "0xf800000" not in asm
