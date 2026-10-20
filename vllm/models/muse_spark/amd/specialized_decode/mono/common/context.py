# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The stage context every K2 stage takes (``c``), built once at the top of a
kernel by ``make_context``, and the probe harness that builds, compiles and
runs single-stage kernels with the same context on one GPU.

The context (built in the kernel's trace; every value below is traced unless
noted):

    S                     launch rows (a Python int, a build constant)
    tid, bid, lane, wave  thread, CTA, tid % 64, tid // 64
    rank, slot, e, par    ABI scalars; the step epoch (22 bits) and its parity
    put, put_bf,          the launch's ``Mailbox`` methods (bound: the AST
    put_words, poll       rewriter must see no method call on a local)
    red                   LDS f32[WAVES 64 4]: MFMA accumulators (``row_sum``)
    rt                    ``RouteLds`` field -> LDS pointer
    lds                   stage -> field -> LDS pointer (the stage union)
    args                  ABI name -> kernel argument (Int64 address / Int32)
    scr                   scratch mailbox region -> ``Addr``; ``xrow``, ``xqd``:
                          plain Int64 bases
    peer                  peer region -> [an ``Addr`` a rank], and
                          region + "_own" -> this rank's, parity-shifted

LDS: the fixed ``red`` and ``RouteLds``, then one union of the stages' structs
(``lds_structs``: stage -> ``fx.struct``, or stage -> {member: ``fx.struct``}
for a sequential composite whose members are all live at once, e.g. K2's
``xq_ug``). A composite's fields are reachable both under its own name and
under each member's. Union members share bytes: a stage that touches its
struct or ``red`` ends with ``gpu.barrier()`` (the hand-over rule).

A probe (``probe_kernel``) is a kernel whose body is one caller-given function
of ``c``: the fixed probe ABI (``PROBE_ARGS``), any K2 argument the stage needs
loaded from an int64 table (``argv``), one CTA or the full grid. With
``peers_alias`` every rank's peer address is this rank's own buffer, so a
probe that pushes to "every rank" and polls its own region runs on one GPU;
``ProbeBuffers`` holds its buffers and writes the pairs other ranks would have
pushed.
"""

import enum
import hashlib
import inspect
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import regex as re
from flydsl.compiler.protocol import dsl_align_of, dsl_size_of
from flydsl.expr.struct import is_struct_type
from flydsl.expr.typing import Int32, Int64

from vllm.models.muse_spark.amd.specialized_decode.mono import spec
from vllm.models.muse_spark.amd.specialized_decode.mono.abi import (
    FAILED,
    HEALTH_WORDS,
    PEER_HEAD,
    WAIT_BUDGET,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.debug import (
    given_up_waits,
    read_pairs,
    region_ids,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import load_ptr64
from vllm.models.muse_spark.amd.specialized_decode.mono.common.plan import (
    LDS_BYTES,
    THREADS,
    WAVES,
    KernelAbi,
    source_digest,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ranks import peer_bases
from vllm.models.muse_spark.amd.specialized_decode.mono.common.route_lds import (
    RouteLds,
    route_ptrs,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.sync import (
    Mailbox,
    load_budget,
    load_epoch,
    preg,
    shift,
    sreg,
    stall,
    step_tag,
)

RED_WORDS = WAVES * 64 * 4  # a wave's 16x16 accumulator, 4 f32 a lane


@fx.struct
class FixedLds:
    red: fx.Array[fx.Float32, RED_WORDS, 16]


# The arguments make_context reads; a kernel's ``args`` carries them and any
# other K2 argument its stages use.
CONTEXT_ARGS = ("scratch", "peers", "rank", "epoch", "slot", "health", "alarm")


# ---------------------------------------------------------------- LDS


def _member(stage: str, st):
    """A union member: a stage's struct, or its members laid out in order."""
    if isinstance(st, Mapping):
        if not st:
            raise ValueError(f"LDS composite {stage!r} has no member")
        for name, t in st.items():
            if not is_struct_type(t):
                raise TypeError(f"LDS {stage}.{name} is not an fx.struct: {t!r}")
        return fx.struct[tuple(slice(name, t) for name, t in st.items())]
    if not is_struct_type(st):
        raise TypeError(f"LDS {stage!r} is not an fx.struct: {st!r}")
    return st


def stage_union(lds_structs: Mapping | None):
    """The union of the stages' LDS structs (None when there is none)."""
    if not lds_structs:
        return None
    return fx.union[
        tuple(slice(stage, _member(stage, st)) for stage, st in lds_structs.items())
    ]


def _align_up(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def lds_total_bytes(lds_structs: Mapping | None = None) -> int:
    """The LDS a context with ``lds_structs`` allocates: ``red``, ``RouteLds``,
    then the stage union, each at its alignment (``SharedAllocator``'s bump)."""
    off = 0
    parts = [FixedLds, RouteLds]
    union = stage_union(lds_structs)
    if union is not None:
        parts.append(union)
    for t in parts:
        off = _align_up(off, dsl_align_of(t)) + dsl_size_of(t)
    return off


def stage_bytes(lds_structs: Mapping) -> dict[str, int]:
    """Stage -> its union member's bytes (a composite's members in order)."""
    return {s: dsl_size_of(_member(s, st)) for s, st in lds_structs.items()}


def _field_ptrs(inst, t, out: dict, where: str) -> dict:
    """Field -> LDS pointer of a peeked struct ``inst`` of type ``t``, nested
    structs flattened (field names must be unique)."""
    for f in t.__dsl_field_defs__:
        v = getattr(inst, f.name)
        if is_struct_type(f.type_spec):
            _field_ptrs(v, f.type_spec, out, f"{where}.{f.name}")
            continue
        if f.name in out:
            raise ValueError(f"LDS field {f.name!r} appears twice in {where}")
        out[f.name] = v.ptr
    return out


def _stage_ptrs(alloc, lds_structs: Mapping | None) -> dict:
    """Allocate the stage union; stage (and composite member) -> field ->
    pointer."""
    lds: dict = {}
    union = stage_union(lds_structs)
    if union is None or lds_structs is None:
        return lds
    st = alloc.allocate(union)
    for stage, member in lds_structs.items():
        inst = getattr(st, stage).peek()
        if isinstance(member, Mapping):
            flat: dict = {}
            for name, t in member.items():
                if name in lds or name in lds_structs:
                    raise ValueError(f"LDS name {name!r} names two stages")
                lds[name] = _field_ptrs(getattr(inst, name), t, {}, f"{stage}.{name}")
                for k, p in lds[name].items():
                    if k in flat:
                        raise ValueError(f"LDS field {k!r} appears twice in {stage}")
                    flat[k] = p
            lds[stage] = flat
        else:
            if stage in lds:
                raise ValueError(f"LDS name {stage!r} names two stages")
            lds[stage] = _field_ptrs(inst, member, {}, stage)
    return lds


# ---------------------------------------------------------------- the context


def make_context(
    S: int,
    args: Mapping,
    lds_structs: Mapping | None = None,
    *,
    peers_alias: bool = False,
    key: spec.K2Build | None = None,
) -> dict:
    """The stage context ``c`` (module docstring), traced in a kernel body.

    ``args``: ABI name -> kernel argument, at least ``CONTEXT_ARGS``.
    ``lds_structs``: stage -> struct (or {member: struct}), the stage union.
    ``peers_alias``: every rank's peer base is ``peers[0]`` (one-GPU probes).
    ``key``: a fault build compiles ``sync.stall`` in first (its prologue).
    """
    if not 1 <= S <= spec.MAX_ROWS:
        raise ValueError(f"launch rows {S} (1 .. {spec.MAX_ROWS})")
    missing = [n for n in CONTEXT_ARGS if n not in args]
    if missing:
        raise KeyError(f"make_context needs the arguments {missing}")
    if lds_total_bytes(lds_structs) > LDS_BYTES:
        raise ValueError(
            f"{lds_total_bytes(lds_structs)} B of LDS > {LDS_BYTES} (a CU's)"
        )
    scratch, peers, rank = args["scratch"], args["peers"], args["rank"]
    epoch, slot, health, alarm = (
        args["epoch"],
        args["slot"],
        args["health"],
        args["alarm"],
    )
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    alloc = fx.SharedAllocator()
    fixed = alloc.allocate(FixedLds).peek()
    rl = alloc.allocate(RouteLds).peek()
    lds = _stage_ptrs(alloc, lds_structs)
    if key is not None and key.faulty:
        stall(
            key.fault_rank,
            key.fault_epoch,
            key.fault_slot,
            key.fault_ticks,
            rank,
            epoch,
            slot,
        )
    e = load_epoch(epoch)
    mb = Mailbox(
        step_tag(e, slot) - 1,
        health,
        alarm,
        region_ids(spec.REGIONS),
        load_budget(health),
    )
    par = e & 1
    if peers_alias:
        bases = [load_ptr64(peers, 0)] * spec.TP
        own = bases[0]
    else:
        bases = peer_bases(peers, spec.TP)
        own = load_ptr64(peers, rank)
    delta = fx.Int64(par) * spec.peer_half_bytes()
    peer: dict = {}
    for region, (off, _) in spec.peer_layout().items():
        peer[region] = [shift(preg(b, off, region), delta) for b in bases]
        peer[region + "_own"] = shift(preg(own, off, region), delta)
    lay = spec.scratch_layout()
    scr: dict = {r: sreg(scratch, lay[r][0], r) for r in spec.SCRATCH_REGIONS}
    scr[spec.XROW] = scratch + fx.Int64(lay[spec.XROW][0])
    scr[spec.XQD] = scratch + fx.Int64(lay[spec.XQD][0])
    return {
        "S": S,
        "tid": tid,
        "bid": bid,
        "lane": tid % 64,
        "wave": tid // 64,
        "rank": rank,
        "slot": slot,
        "e": e,
        "par": par,
        "put": mb.put,
        "put_bf": mb.put_bf,
        "put_words": mb.put_words,
        "poll": mb.poll,
        "red": fixed.red.ptr,
        "rt": route_ptrs(rl),
        "lds": lds,
        "args": dict(args),
        "scr": scr,
        "peer": peer,
        # the build key (None in probes) and its debug flag: pool.load_xq's
        # default (design 5.2 calls it with no flag)
        "key": key,
        "debug_dump": bool(key is not None and key.debug_dump),
    }


# ---------------------------------------------------------------- probes

PROBE_ARGS = ("scratch", "peers", "rank", "epoch", "slot", "health", "alarm", "argv")
PROBE_ABI = KernelAbi(PROBE_ARGS)
SOURCES = source_digest(".")  # the mono package's kernel sources

_STREAM = fx.Stream(None)
_PROBES: dict = {}


_KEY_SCALARS = (int, float, bool, str, bytes, type(None), enum.Enum)


# FlyDSL's AST rewriter names the functions it generates from a process-wide
# counter (``__then_27``): not part of a body's identity
_GENERATED = re.compile(r"(__[A-Za-z]+_)\d+")


def _code_ident(code) -> str:
    """A code object's bytecode, constants (nested code objects recursively)
    and global names: the same in every process (no addresses, no line
    numbers, no names the rewriter numbered; locals are indexes in the
    bytecode)."""
    consts = []
    for k in code.co_consts:
        if inspect.iscode(k):
            consts.append(_code_ident(k))
        elif isinstance(k, frozenset):
            consts.append(repr(sorted(map(repr, k))))
        else:
            consts.append(repr(k))
    text = f"{code.co_code.hex()}|{consts}|{code.co_names}"
    return _GENERATED.sub(r"\1N", text)


def _global_names(code) -> set:
    out = set(code.co_names)
    for k in code.co_consts:
        if inspect.iscode(k):
            out |= _global_names(k)
    return out


def _value_ident(v, seen: set) -> str:
    """A value a probe body captures, as key text: scalars and containers by
    value, functions by ``_fn_ident``, LDS structs by layout; any other object
    by identity (never shared with another object's build)."""
    if isinstance(v, _KEY_SCALARS):
        return repr(v)
    if isinstance(v, (tuple, list)):
        return f"{type(v).__name__}({','.join(_value_ident(x, seen) for x in v)})"
    if isinstance(v, Mapping):
        items = (
            f"{_value_ident(k, seen)}:{_value_ident(x, seen)}" for k, x in v.items()
        )
        return "{" + ",".join(items) + "}"
    if is_struct_type(v):
        names: list = []
        _field_names(v, names)
        return f"struct {v.__qualname__} {dsl_size_of(v)} {','.join(names)}"
    if inspect.isfunction(v):
        return _fn_ident(v, seen)
    if inspect.ismodule(v):
        return f"module {v.__name__}"
    if isinstance(v, type):
        return f"type {v.__module__}.{v.__qualname__}"
    return f"object {type(v).__qualname__}@{id(v):x}"


def _fn_ident(fn, seen: set) -> str:
    """A body function as key text: its code, the values its closure cells and
    defaults hold, and the module-level scalars, LDS structs and same-module
    functions it reads. A factory's bodies that differ only in a captured value
    are different builds (FlyDSL keys no object in a closure, and walks no
    function it reaches only through one)."""
    name = f"fn {fn.__module__}.{fn.__qualname__}"
    if fn in seen:
        return name
    seen.add(fn)
    parts = [name, _code_ident(fn.__code__)]
    for var, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        try:
            v = cell.cell_contents
        except ValueError:  # a cell not yet bound
            parts.append(f"{var}=<unbound>")
            continue
        parts.append(f"{var}={_value_ident(v, seen)}")
    parts += [_value_ident(d, seen) for d in fn.__defaults__ or ()]
    parts += [
        f"{k}={_value_ident(d, seen)}" for k, d in (fn.__kwdefaults__ or {}).items()
    ]
    g = fn.__globals__
    for var in sorted(_global_names(fn.__code__)):
        if var not in g:
            continue
        v = g[var]
        if (
            isinstance(v, (*_KEY_SCALARS, tuple))
            or is_struct_type(v)
            or (inspect.isfunction(v) and v.__module__ == fn.__module__)
        ):
            parts.append(f"global {var}={_value_ident(v, seen)}")
    return "\n".join(parts)


def _body_digest(body: Callable) -> str:
    """The probe body's key: ``_fn_ident`` (or the identity of a callable
    object), hashed."""
    ident = _value_ident(body, set())
    return hashlib.sha256(ident.encode()).hexdigest()[:16]


def _lds_signature(lds_structs: Mapping | None) -> tuple:
    """Scalars naming the stage structs (names, bytes, fields): a key part."""
    out = []
    for stage, st in (lds_structs or {}).items():
        t = _member(stage, st)
        names: list[str] = []
        _field_names(t, names)
        out.append((stage, dsl_size_of(t), ",".join(names)))
    return tuple(out)


def _field_names(t, out: list) -> None:
    for f in t.__dsl_field_defs__:
        if is_struct_type(f.type_spec):
            _field_names(f.type_spec, out)
        else:
            out.append(f"{f.name}:{dsl_size_of(f.type_spec)}")


def _load_argv(argv, names: Sequence[str]) -> dict:
    """The probe's extra arguments, entry i of the int64 table ``argv``."""
    return {n: load_ptr64(argv, i) for i, n in enumerate(names)}


@dataclass(frozen=True)
class Probe:
    """A built probe: ``launch`` is its ``@flyc.jit`` launcher, taking
    ``PROBE_ARGS`` (then ``stream``)."""

    name: str
    tokens: int
    blocks: int
    extra_args: tuple[str, ...]
    launch: Callable[..., None]

    def compile(self) -> None:
        """Compile without running (FlyDSL's compile-only mode; no GPU needed:
        the load then fails harmlessly with hipErrorNoDevice)."""
        old = os.environ.get("COMPILE_ONLY")
        os.environ["COMPILE_ONLY"] = "1"
        try:
            self.launch(*PROBE_ABI.zeros())
        finally:
            if old is None:
                del os.environ["COMPILE_ONLY"]
            else:
                os.environ["COMPILE_ONLY"] = old

    def __call__(self, bufs: "ProbeBuffers", *, hip_stream=None, **tensors) -> None:
        """Launch on ``bufs`` with the extra arguments ``tensors`` (by name: a
        tensor's address, or an int). ``hip_stream`` is the launch stream (the
        current one if None); it is not named ``stream``, which is a K2 tensor
        argument (the residual stream)."""
        import torch

        extra = set(tensors) ^ set(self.extra_args)
        if extra:
            raise TypeError(f"probe {self.name}: arguments {sorted(extra)} mismatch")
        argv = bufs.argv(self.extra_args, tensors)
        cur = torch.cuda.current_stream(bufs.device)
        stream = hip_stream
        if isinstance(stream, torch.cuda.Stream) and stream != cur:
            # the table was copied on the current stream: order the launch
            # after it, and keep it alive until the launch's stream is done
            stream.wait_stream(cur)
            argv.record_stream(stream)
        self.launch(
            *PROBE_ABI.pack(
                {
                    "scratch": bufs.scratch.data_ptr(),
                    "peers": bufs.peers.data_ptr(),
                    "rank": bufs.rank,
                    "epoch": bufs.epoch.data_ptr(),
                    "slot": bufs.slot,
                    "health": bufs.health.data_ptr(),
                    "alarm": bufs.alarm,
                    "argv": argv.data_ptr(),
                }
            ),
            stream=stream if stream is not None else cur,
        )


def probe_kernel(
    name: str,
    S: int,
    body: Callable[[dict], None],
    lds_structs: Mapping | None = None,
    extra_args: Sequence[str] = (),
    *,
    blocks: int = 1,
    peers_alias: bool = True,
    key: tuple = (),
) -> Probe:
    """A single-stage kernel: ``make_context(S, args, lds_structs)`` and then
    ``body(c)``, on ``blocks`` CTAs (1, or the full grid) of THREADS threads.

    ``body`` runs in the kernel's trace: dynamic control flow in it needs
    ``ops.traced``, calls must be CTA-uniform where they barrier. ``extra_args``
    names the K2 arguments the stage reads (``c["args"][name]``, an Int64 from
    the ``argv`` table). ``key``: more (name, scalar) pairs for the JIT key
    (name, S, blocks, arguments, aliasing, the body -- its code, the values
    its closure and defaults hold, the module-level scalars and same-module
    functions it reads -- the LDS structs and the mono sources are keyed
    already)."""
    extra = tuple(extra_args)
    clash = sorted(set(extra) & set(PROBE_ARGS))
    if clash:
        raise ValueError(f"probe arguments {clash} are fixed probe arguments")
    if len(set(extra)) != len(extra):
        raise ValueError(f"probe arguments {extra} repeat a name")
    if not 1 <= blocks <= spec.BLOCKS:
        raise ValueError(f"probe grid {blocks} (1 .. {spec.BLOCKS})")
    if not 1 <= S <= spec.MAX_ROWS:
        raise ValueError(f"probe rows {S} (1 .. {spec.MAX_ROWS})")
    if lds_total_bytes(lds_structs) > LDS_BYTES:
        raise ValueError(
            f"probe {name}: {lds_total_bytes(lds_structs)} B of LDS > {LDS_BYTES}"
        )
    keyed = (
        ("probe", name),
        ("tokens", S),
        ("blocks", blocks),
        ("extra_args", extra),
        ("peers_alias", peers_alias),
        ("body", _body_digest(body)),
        ("lds", _lds_signature(lds_structs)),
        ("sources", SOURCES),
        *key,
    )
    if keyed in _PROBES:
        return _PROBES[keyed]
    tokens = S
    alias = peers_alias

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def probe(
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        slot: Int32,
        health: Int64,
        alarm: Int64,
        argv: Int64,
    ):
        _ = keyed
        args = {
            "scratch": scratch,
            "peers": peers,
            "rank": rank,
            "epoch": epoch,
            "slot": slot,
            "health": health,
            "alarm": alarm,
            **_load_argv(argv, extra),
        }
        body(make_context(tokens, args, lds_structs, peers_alias=alias))

    @flyc.jit
    def launch(
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        slot: Int32,
        health: Int64,
        alarm: Int64,
        argv: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        probe(scratch, peers, rank, epoch, slot, health, alarm, argv).launch(
            grid=(blocks,), block=(THREADS,), stream=stream
        )

    PROBE_ABI.check(probe, launch)
    built = Probe(name, S, blocks, extra, launch)
    _PROBES[keyed] = built
    return built


class ProbeBuffers:
    """One GPU's buffers for probe launches, as ``MonoRunner`` holds a rank's:
    the scratch, one peer buffer every rank slot of ``peers`` points into (past
    ``PEER_HEAD``), the epoch word, the health words (with a wait budget) and
    the alarm word (this rank's word of the peer buffer's head)."""

    def __init__(
        self,
        device="cuda",
        *,
        epoch: int = 1,
        budget: int = 2,
        rank: int = 0,
        slot: int = spec.K2_SLOT0,
    ):
        import torch

        self.device = torch.device(device)
        self.scratch = torch.zeros(
            spec.scratch_bytes(), dtype=torch.uint8, device=device
        )
        self.peer = torch.zeros(
            PEER_HEAD + spec.peer_bytes(), dtype=torch.uint8, device=device
        )
        base = self.peer.data_ptr()
        self.peers = torch.full(
            (spec.TP,), base + PEER_HEAD, dtype=torch.int64, device=device
        )
        self.epoch = torch.tensor([epoch], dtype=torch.int32, device=device)
        self.health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device=device)
        self.health[WAIT_BUDGET] = budget
        self.alarm = base + 4 * rank
        self.rank = rank
        self.slot = slot
        self._argv = None  # the last launch's table: alive until the next

    @property
    def e(self) -> int:
        """The host's copy of the epoch word (a device read)."""
        return int(self.epoch.item())

    def tag(self) -> int:
        """The tag a launch now writes and waits for (``spec.step_tag``)."""
        return spec.step_tag(self.e, self.slot)

    def flags(self):
        """The peer buffer's head words (``alarm``: a word a rank)."""
        import torch

        return self.peer[:PEER_HEAD].view(torch.int32)

    def failed(self) -> bool:
        return int(self.health[FAILED].item()) != 0

    def given_up_waits(self) -> list[str]:
        return given_up_waits(self.health.cpu(), spec.REGIONS)

    def _region(self, region: str):
        """(buffer, byte offset, bytes) of ``region``."""
        lay = spec.scratch_layout()
        if region in lay:
            return self.scratch, lay[region][0], lay[region][1]
        off, nbytes = spec.peer_layout()[region]
        par = self.e & spec.EPOCH_MASK & 1
        return self.peer, PEER_HEAD + par * spec.peer_half_bytes() + off, nbytes

    def where(self, region: str):
        """(buffer, byte offset) of ``region``: a scratch region, or a peer
        region in the current epoch's parity half (past ``PEER_HEAD``)."""
        buf, off, _ = self._region(region)
        return buf, off

    def _pairs_at(self, region: str, index: int, n: int):
        """(buffer, byte offset) of pairs ``index`` .. ``index + n`` of
        ``region``, refused past the region's end (a wrong index would
        otherwise read or fill the next region's pairs silently)."""
        buf, off, nbytes = self._region(region)
        if index < 0 or n < 0 or 8 * (index + n) > nbytes:
            raise IndexError(
                f"pairs [{index}, {index + n}) of {region} ({nbytes // 8} pairs)"
            )
        return buf, off + 8 * index

    def read_pairs(self, region: str, index: int, n: int):
        """(values, tags) of pairs ``index`` .. ``index + n`` of ``region``."""
        buf, off = self._pairs_at(region, index, n)
        return read_pairs(buf, off, n)

    def write_pairs(self, region: str, index: int, values, tag: int | None = None):
        """Pairs ``index`` .. of ``region`` := (``values``, ``tag``; the current
        tag by default): what another rank or CTA would have put. ``values``:
        int32 words, fp32, or bf16 (two a word)."""
        import torch

        words = values.reshape(-1)
        if words.element_size() != 4:
            words = words.contiguous().view(torch.int32)
        words = words.view(torch.int32)
        n = words.numel()
        buf, off = self._pairs_at(region, index, n)
        pairs = buf[off : off + 8 * n].view(torch.int32)
        pairs = pairs.view(n, 2)
        pairs[:, 0] = words.to(self.device)
        pairs[:, 1] = self.tag() if tag is None else tag

    def argv(self, names: Sequence[str], tensors: Mapping):
        """The int64 table of a probe's extra arguments (kept alive)."""
        import torch

        vals = []
        for n in names:
            v = tensors[n]
            vals.append(v if isinstance(v, int) else v.data_ptr())
        self._argv = torch.tensor(vals or [0], dtype=torch.int64, device=self.device)
        return self._argv
