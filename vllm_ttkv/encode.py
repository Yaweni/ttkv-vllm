"""The WRITE half: this step's RAW K and V turned into packed slots in the paged cache.

WHY THIS IS NOT A METHOD ON `rope_store`
    `patch.py` captures raw K by intercepting the model's `rotary_emb(positions, q, k)`, which has no
    slot mapping -- vLLM's slot mapping reaches the attention layer and nothing earlier. So the write
    has to happen in `impl`, which holds the raw K AND the slot mapping AND the per-request split.
    `rope_store` keeps the angle table, the layouts and the registries; the slot arithmetic is here.

THE TWO PHASES, AND WHY THE SPLIT IS THE WHOLE DESIGN
    PREFILL (a request's first chunk) -- a real TT-SVD fit, exactly as `build_k_layer` does it,
    tokens first: `[n, d, H]` so the first unfolding is `n x (d*H)` and its rank counts key
    archetypes shared across heads. Passing `[H, n, d]` would give an `H x (n*d)` unfolding with
    `r1 <= H`, which buys nothing. The fit produces the decoder, and the decoder is what the request
    keeps.

    EVERY LATER STEP -- NO refit. Project onto that frozen decoder with one least-squares solve,
    quantise the coefficients, patch the residual lanes. This is a STREAMING property, not an
    optimisation: refitting would invalidate every coefficient already written, so the store could
    never be read while being appended to. Measured cost of the frozen encode: **0.009 ms/token**
    (0.25 ms over 28 layers, against a 22.9 ms decode), which is why there is no dense tail buffering
    rows up for a cheaper bulk encode -- see `impl._tail_scores`.

WHY THE RANK IS A BUDGET AND NOT A MEASUREMENT
    vLLM allocates the pages before any sequence exists, so a layer's slot width is fixed by the
    calibration table. A request's TT-SVD lands at its own natural rank (measured 6..166 across
    layers), so the fit is made to fit the budget: capped by `max_rank`, then ZERO-PADDED up to it.
    The padding is exact rather than approximate -- a zero coefficient times a zero decoder row is
    zero, and `pinv` of a decoder with zero rows has zero columns there -- so the reconstruction, and
    therefore every residual computed against it, is unchanged. `tests/test_encode.py` asserts that
    bit-for-bit.

    A budget that BINDS is a real accuracy cost and is the allocation decision the calibration table
    makes: measured, a cap of 64 under-ranks `tight` by 1.75x. But the alternative is not a bigger
    rank, it is a second cache region whose width is still fixed -- so this is the honest form of the
    constraint rather than a workaround for it.
"""
from __future__ import annotations

import os

import torch

from vllm_ttkv import prof
from vllm_ttkv.layout import TTKLayout, pack_k_slots, pack_v_slots
from vllm_ttkv.rope_store import CONFIG_ENV

__all__ = [
    "codec", "active_config", "check_cfg_matches_layout", "as_kv_heads",
    "KDecoder", "FittedK", "fit_into_budget", "encode_frozen", "encode_v", "write_slots",
]

_DEFAULT_CONFIG = "safe"
_CODEC = None


def codec():
    """The TT-KV codec, imported LAZILY, with an error that names the actual problem.

    The WRITE is the only half of this integration that needs it. Decoding reads slots, and the slot
    byte format lives in `layout.py`, which has no codec dependency -- so keeping the import inside
    means the read path, and its tests, can be imported and exercised where the codec is not
    packaged. A real run always encodes a prefill, so it gets this message rather than an
    ImportError from three frames down.
    """
    global _CODEC
    if _CODEC is None:
        try:
            import ttkv_packed_cache as pc
        except ImportError as exc:                                        # pragma: no cover
            raise ImportError(
                "vllm_ttkv.encode needs the TT-KV codec (`ttkv_packed_cache`). It is the single "
                "definition of the bit packing, the residual scheme and the TT-SVD fit, so this "
                "module reuses it rather than reimplementing the format -- put the integration "
                "directory on sys.path, or vendor the codec into this package. (Decoding needs no "
                "codec: the slot format lives in vllm_ttkv.layout.)") from exc
        _CODEC = pc
    return _CODEC


def active_config() -> dict:
    """The allocation in force: `ttkv_codec.CONFIGS[$VLLM_TTKV_CONFIG]`, default `safe`.

    Read from the SAME env var the backend uses to size the cache (`rope_store.CONFIG_ENV`), and
    that is load-bearing rather than tidy. The backend derives the slot width from a calibration
    table built for one allocation; the encoder fits the store from the config it reads here. If the
    two disagreed, the fit would be written at one rank and read at another -- which does not raise,
    it misreads every field after the coefficient block.
    """
    from ttkv_codec import CONFIGS
    name = os.environ.get(CONFIG_ENV, _DEFAULT_CONFIG)
    try:
        return CONFIGS[name]
    except KeyError:
        try:
            known = sorted(CONFIGS)
        except Exception:                                                 # pragma: no cover
            known = ["safe", "tight", "campaign4x"]
        raise RuntimeError(
            f"{CONFIG_ENV}={name!r} is not a shipped allocation. Try one of {known}.") from None


def check_cfg_matches_layout(lay: TTKLayout, cfg: dict) -> None:
    """The layout must have been built from THIS config, or the store will not fit its own cache.

    `layout_for_config` derives every field below from the config, so they agree by construction --
    until someone swaps the env var, or passes a layout from a different geometry. The failure that
    prevents is silent: a `res_k` mismatch shifts the residual field inside each head row, and a
    `qbits` mismatch shifts the whole coefficient block, in both cases reading a neighbouring field's
    bytes as codes.
    """
    want = {
        "qbits": cfg["qbits"],
        "res_k": max(int(lay.d * cfg["res_frac"]), 1),
        "res_bits": cfg["res_bits"],
        "v_lanes": cfg["k_lanes"],
        "v_bits": cfg["vbits"],
    }
    got = {
        "qbits": lay.qbits, "res_k": lay.res_k, "res_bits": lay.res_bits,
        "v_lanes": lay.v_lanes, "v_bits": lay.v_bits,
    }
    diff = {k: {"layout": got[k], "config": want[k]} for k in want if got[k] != want[k]}
    if diff:
        raise RuntimeError(
            f"the layout does not match the active allocation: {diff}. The layout sizes the cache "
            f"slots, so a store encoded under one of these and read under the other has every field "
            f"after the coefficients at the wrong offset. Rebuild the calibration table for "
            f"{CONFIG_ENV} or unset it.")


def as_kv_heads(x: torch.Tensor, H: int, d: int, what: str = "tensor") -> torch.Tensor:
    """`[T, H*d]` or `[T, H, d]` -> `[T, H, d]`.

    vLLM's models hand attention a 3D key/value, but some pass a flat `[T, H*d]`, and `Attention`
    itself reshapes with `.view(-1, num_kv_heads, head_size)` -- implying both reach it. A wrong
    reshape here would mis-assign heads, which shows up as attention over permuted heads and nothing
    else.
    """
    if x.dim() == 2:
        if x.shape[-1] != H * d:
            raise ValueError(
                f"{what} is [T, {x.shape[-1]}] but {H} kv heads of width {d} need {H * d}")
        return x.reshape(x.shape[0], H, d)
    if x.dim() == 3:
        if tuple(x.shape[1:]) != (H, d):
            raise ValueError(f"{what} is [T, {x.shape[1]}, {x.shape[2]}] but expected [T, {H}, {d}]")
        return x
    raise ValueError(f"{what} has {x.dim()} dims; expected 2 or 3")


def as_fit_dtype(x: torch.Tensor, what: str) -> torch.Tensor:
    """Cast an incoming activation to the dtype the FIT runs in.

    vLLM hands over bf16 -- `--dtype bfloat16` is the serving default -- and `tt_svd` calls
    `torch.linalg.svd`, which has no bf16 CUDA kernel. Measured on the first real run:

        NotImplementedError: "svd_cuda_gesvdj" not implemented for 'BFloat16'

    Casting is not merely a workaround for that. The rank decision is "dropped energy <= tol^2",
    i.e. a numerical judgement on a singular spectrum, and every number in the calibration table
    (and every ratio quoted from it) was measured in fp32 with the offline tooling. Fitting in bf16
    would decide ranks from a spectrum carrying ~3 fewer decimal digits than the table was built
    on, so the shipped `r1` and the runtime rank would be answering slightly different questions --
    and the result would be a plausible rank rather than an error.

    `encode_frozen` needs no cast: it already computes in fp32 (`.float()` on `M` and `R`). This is
    the fit and V-encode path, where the codec is handed the tensor directly.
    """
    if x.dtype == torch.float32:
        return x
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            f"{what} arrived as {x.dtype}, which the fit does not accept. Activations should be a "
            f"floating dtype; an integer tensor here means the wrong thing was passed.")
    return x.float()


class KDecoder:
    """A request's frozen decoder, plus the pseudo-inverse the frozen encode needs.

    `pinv` is cached, and that is not premature: it is a `pinv` of an `[r1, H*d]` matrix, and
    recomputing it per token would swamp the encode it exists to serve. It is constant for the
    request's whole life, because "frozen" is the whole point of it.
    """

    __slots__ = ("decoder_f", "_pinv")

    def __init__(self, decoder_f: torch.Tensor):
        if decoder_f.dim() != 2:
            raise ValueError(
                f"a decoder is [r1, H*d]; got {tuple(decoder_f.shape)}. `krt` reads it as "
                f"`[r1, H*d]` with the head axis folded in, so a 3D tensor here would be flattened "
                f"into the wrong row order.")
        self.decoder_f = decoder_f
        self._pinv: torch.Tensor | None = None

    @property
    def pinv(self) -> torch.Tensor:
        """`[H*d, r1]`, the least-squares projector onto this basis.

        NOTE the shape asymmetry, which is easy to get backwards: `pinv` is built from `decoder_f`
        (`[r1, H*d]`), so it is `[H*d, r1]` and the encode is `M @ pinv` with `M` being
        `[n, H*d]`. Writing `c = pinv @ M` instead gives an `[r1, H*d]`-shaped object that cannot be
        broadcast into a slot at all -- one of the few mistakes on this path that fails loudly.
        """
        if self._pinv is None:
            self._pinv = torch.linalg.pinv(self.decoder_f)
        return self._pinv

    def __repr__(self) -> str:
        return f"KDecoder(r1={self.decoder_f.shape[0]}, H*d={self.decoder_f.shape[1]})"


class FittedK:
    """Everything `layout.pack_k_slots` needs, plus the decoder the request keeps.

    THE RESIDUAL IS BUILT ON FIRST USE, and that is a performance decision rather than tidiness.
    `PackedRows.build` is about twenty launches (`abs`, `topk`, `scatter_`, `amax`, the bitmap fold,
    the bit-plane fold) and it is the largest single item of the write -- measured `proj.resid`
    0.728 ms a layer, against 0.192 ms for the projection it is a residual OF. The fused Triton path
    (`select_triton`) consumes the residual VALUES and packs them itself, so on that path the build
    is dead work done once per layer per step.

    Keeping it reachable through the property is the point rather than a nicety: the torch path,
    `pack_k_slots`, the reference decoder and the accuracy tests all still ask for `fk.resid` and get
    exactly the object they got before, just later. That is also what makes the fused path
    testable -- `tests/test_encode_triton.py` takes the bytes this builds and requires the kernel to
    reproduce them.
    """

    __slots__ = ("coeff_q", "coeff_scale", "_resid", "_build_resid", "resid_src", "decoder",
                 "fitted_rank", "budget")

    def __init__(self, coeff_q, coeff_scale, resid, decoder, fitted_rank, budget,
                 resid_src=None, build_resid=None):
        self.coeff_q = coeff_q            # [n, budget] int8
        self.coeff_scale = coeff_scale    # [n] fp16
        self._resid = resid               # PackedRows over H*n rows, or None until asked for
        self._build_resid = build_resid   # how to build it from `resid_src`, if anything asks
        self.resid_src = resid_src        # [H*n, d] fp32, the residual the fused path packs
        self.decoder = decoder            # KDecoder
        self.fitted_rank = fitted_rank    # the natural rank before padding; == budget if it bound
        self.budget = budget

    @property
    def resid(self):
        """The packed residual rows, built on first use. `None` on a store that has no residual."""
        if self._resid is None and self._build_resid is not None:
            self._resid = self._build_resid()
        return self._resid

    def __repr__(self) -> str:
        return (f"FittedK(rank {self.fitted_rank}/{self.budget}, "
                f"{self.coeff_q.shape[0]} tokens)")


def fit_into_budget(K: torch.Tensor, lay: TTKLayout, cfg: dict) -> FittedK:
    """A request's first chunk: TT-SVD, capped at the layer's rank budget, then zero-padded to it.

    `K` is `[H, n, d]`, RAW (position-free) -- this is the only place the fit runs for a request.
    Any floating dtype is accepted and the fit itself runs in fp32 (`as_fit_dtype`).
    """
    pc = codec()
    check_cfg_matches_layout(lay, cfg)
    K = as_fit_dtype(K, "the raw key")
    H, n, d = K.shape
    if (H, d) != (lay.H, lay.d):
        raise ValueError(
            f"K is [H={H}, n={n}, d={d}] but the layout is for H={lay.H}, d={lay.d}. `H` here is "
            f"the KV HEAD COUNT, which has to equal the layout's -- the slot holds one residual row "
            f"per head, so a mismatch writes rows into the wrong heads.")

    kl = pc.build_k_layer(K.contiguous(), cfg["tols"], cfg["qbits"], cfg["res_frac"],
                          cfg["res_bits"], max_rank=lay.r1, d=lay.d)
    if kl.r1 > lay.r1:
        raise RuntimeError(
            f"`build_k_layer` returned r1={kl.r1} against max_rank={lay.r1}; the cap is supposed "
            f"to be hard, and a rank above the budget cannot be written into the slot at all.")
    if kl.resid is not None and kl.resid.k != lay.res_k:
        raise RuntimeError(
            f"the fit kept {kl.resid.k} residual lanes per row but the layout reserves "
            f"{lay.res_k}. `res_k` is derived from `res_frac` as `int(d * res_frac)`, so this means "
            f"the layout and the config disagree about the lane budget -- and the residual field "
            f"inside every head row would be read at the wrong offset.")

    dec_f = kl.decoder_f()
    if kl.r1 == lay.r1:
        coeff_q, padded = kl.coeff_q, dec_f
    else:
        # Zero-padding: a zero coefficient times a zero decoder row is zero, and `pinv` of a
        # decoder with zero rows has zero COLUMNS where those rows were -- so the frozen encode
        # solves into the shared columns and leaves the padded ones at exactly zero.
        coeff_q = torch.zeros(n, lay.r1, dtype=torch.int8, device=K.device)
        coeff_q[:, :kl.r1] = kl.coeff_q
        padded = torch.zeros(lay.r1, H * d, device=K.device, dtype=torch.float32)
        padded[:kl.r1] = dec_f
    return FittedK(coeff_q=coeff_q, coeff_scale=kl.coeff_scale, resid=kl.resid,
                   decoder=KDecoder(padded), fitted_rank=kl.r1, budget=lay.r1)


def encode_frozen(K: torch.Tensor, kdec: KDecoder, lay: TTKLayout) -> FittedK:
    """Every step after a request's first chunk: project onto its frozen decoder. No refit.

    The body mirrors `build_k_layer`'s tail deliberately, INCLUDING the row conventions:
    `build_k_layer` forms `kflat = c @ dec` with column index `m = h*d + j`, compares it against
    `K.reshape(H, n, d)`, and keeps the residual in that (head, token) order. Re-deriving the layout
    here rather than copying it is how a port ends up reporting a misalignment as an accuracy loss.

    `tools/measure_fold_gap.py` measured the fold penalty with this same arithmetic, and
    `tests/test_encode.py` pins the two together -- otherwise the measured numbers would stop
    describing the shipped path the moment either drifted.
    """
    pc = codec()
    H, n, d = K.shape
    if (H, d) != (lay.H, lay.d):
        raise ValueError(f"K is [H={H}, n={n}, d={d}] but the layout is for H={lay.H}, d={lay.d}")
    if kdec.decoder_f.shape[0] != lay.r1:
        raise ValueError(
            f"the frozen decoder has r1={kdec.decoder_f.shape[0]} but the layout budgets "
            f"{lay.r1}; a coefficient row written at one rank cannot be read at another")

    M = K.permute(1, 0, 2).reshape(n, H * d).float()          # token-major, m = h*d + j
    with prof.phase("proj.matmul"):
        c = M @ kdec.pinv                                     # [n, r1] least squares

    # The ruler formula is the codec's, and is written the same way here because it IS the format:
    # one ruler per row, from the row's own max. A second spelling would be a second format.
    with prof.phase("proj.quant"):
        cscale = c.abs().amax(1, keepdim=True) / (2 ** (lay.qbits - 1) - 1)
        cscale = torch.where(cscale == 0, torch.ones_like(cscale), cscale)
        cq = torch.clamp(torch.round(c / cscale),
                         -(2 ** (lay.qbits - 1)), 2 ** (lay.qbits - 1) - 1)

    with prof.phase("proj.recon"):
        kd = ((cq * cscale) @ kdec.decoder_f).reshape(n, H, d).permute(1, 0, 2)     # [H, n, d]
        R = (K.reshape(H, n, d).float() - kd).reshape(H * n, d)

    def build_resid():
        """The torch residual rows, over this same `R`. Only the fallback path asks for it.

        `check=False`: the mask has exactly `res_k` True per row by construction here, so the
        raggedness assertion in `PackedRows.build` would be a device sync guarding nothing.
        """
        with prof.phase("proj.resid"):
            mask = torch.zeros_like(R, dtype=torch.bool)
            mask.scatter_(1, torch.topk(R.abs(), lay.res_k, dim=1).indices, True)
            return pc.PackedRows.build(R, mask, lay.res_bits, lay.d, k=lay.res_k, check=False)

    return FittedK(coeff_q=cq.to(torch.int8), coeff_scale=cscale.half().reshape(n), resid=None,
                   decoder=kdec, fitted_rank=lay.r1, budget=lay.r1, resid_src=R,
                   build_resid=build_resid)


def encode_frozen_batch(K: torch.Tensor, kdecs, lay: TTKLayout) -> list:
    """`encode_frozen` for a whole group of requests at once: `G` requests, ONE set of launches.

    WHY THIS EXISTS, and the number is the reason. `encode_frozen` is about twenty torch ops, and on
    a DECODE step every one of them is a launch on a ONE-TOKEN tensor. At batch 8 that is 28 layers
    x 8 requests x ~20 = ~4,480 launches a step, and the host cannot issue them fast enough -- the
    profile reads `write.project` at 1.14 ms/step at batch 1 and **102.5 ms/step at batch 8**, a
    12.6x growth for 8x the work, because the OP COUNT rises with the batch while the size of each op
    does not. Batching turns `G` sequences of ~20 tiny ops into one sequence of ~20 ops on `G`x the
    data, which is the same "fewer, larger rounds" fix the write's pack and the plan path both got.

    THE GROUP MUST SHARE A TOKEN COUNT, so the caller groups by `end - start` rather than handing
    over the step. A decode step has `n == 1` for every request, so it is ONE group; a prefill step
    has requests of different lengths and they stay per-request (or group among themselves). That is
    the right trade rather than a limitation: padding a 1-token request up to a 5,753-token one to
    share a kernel would cost far more than the launches it saves.

    `r1` IS THE SAME FOR EVERY MEMBER, which is what makes the batch dimension free. The layout is
    per LAYER, so every request in this layer has the same rank budget; only the VALUES of the
    per-request decoder differ, and those become the batch dimension of the two `bmm`s.

    THE DECODER STAYS PER-REQUEST. A shared basis across a batch measured 5.8 points wrong, so this
    batches the OPERATIONS and never the fit. `K` is `[G, H, n, d]`, stacked from the same
    `[H, n, d]` slices `encode_frozen` takes one at a time, and the arithmetic mirrors that function
    line for line -- including the `m = h*d + j` row convention, which a re-derivation here would
    turn into a reported accuracy loss.
    """
    pc = codec()
    G, H, n, d = K.shape
    if (H, d) != (lay.H, lay.d):
        raise ValueError(
            f"K is [G={G}, H={H}, n={n}, d={d}] but the layout is for H={lay.H}, d={lay.d}")
    if len(kdecs) != G:
        raise ValueError(f"K holds {G} requests but {len(kdecs)} decoders were given")
    for i, kd in enumerate(kdecs):
        if kd.decoder_f.shape[0] != lay.r1:
            raise ValueError(
                f"decoder {i} has r1={kd.decoder_f.shape[0]} but the layout budgets {lay.r1}; a "
                f"coefficient row written at one rank cannot be read at another")

    M = K.permute(0, 2, 1, 3).reshape(G, n, H * d).float()          # [G, n, H*d], m = h*d + j
    with prof.phase("proj.matmul"):
        c = torch.bmm(M, torch.stack([kd.pinv for kd in kdecs]))    # [G, n, r1] least squares

    with prof.phase("proj.quant"):
        cscale = c.abs().amax(2, keepdim=True) / (2 ** (lay.qbits - 1) - 1)
        cscale = torch.where(cscale == 0, torch.ones_like(cscale), cscale)
        cq = torch.clamp(torch.round(c / cscale),
                         -(2 ** (lay.qbits - 1)), 2 ** (lay.qbits - 1) - 1)

    # THE RECONSTRUCTION IS PER REQUEST, and it is the one deliberate exception to the batching
    # above. `bmm` and `@` select different dense kernels, so a batched reconstruction differs from
    # the per-request one in the last bits: 1.5e-08 over 795 of 8,192 residual elements, measured.
    # The COEFFICIENTS are unaffected -- they are quantized and the test requires them to be
    # bit-identical -- but the residual goes to a MAGNITUDE SELECTION, where a last-bit difference
    # is a tie-break that can move a bit in the bitmap and change stored bytes. It did not move one
    # on any of the three shipped allocations, which is precisely why this is worth removing rather
    # than measuring: the cost is one `@` a request against the ~20 launches a request this function
    # exists to save, and it buys a `torch.equal` the whole way through instead of a spot check.
    out = []
    for g, kdec in enumerate(kdecs):
        with prof.phase("proj.recon"):
            kd = ((cq[g] * cscale[g]) @ kdec.decoder_f).reshape(n, H, d).permute(1, 0, 2)
            Rg = (K[g].reshape(H, n, d).float() - kd).reshape(H * n, d)   # [H*n, d]

        def build_resid(Rg=Rg):
            with prof.phase("proj.resid"):
                mask = torch.zeros_like(Rg, dtype=torch.bool)
                mask.scatter_(1, torch.topk(Rg.abs(), lay.res_k, dim=1).indices, True)
                return pc.PackedRows.build(Rg, mask, lay.res_bits, lay.d, k=lay.res_k, check=False)

        out.append(FittedK(coeff_q=cq[g].to(torch.int8), coeff_scale=cscale[g].reshape(n).half(),
                           resid=None, decoder=kdec, fitted_rank=lay.r1, budget=lay.r1,
                           resid_src=Rg, build_resid=build_resid))
    return out


def encode_v(V: torch.Tensor, lay: TTKLayout):
    """V is `[H, n, d]`, RAW and unrotated, and has no basis: a V slot is self-contained.

    `k_lanes` in the codec is an ABSOLUTE lane count, not a fraction of `d` -- so it is taken from
    the layout (which is what sized the slot) rather than from the config, and
    `check_cfg_matches_layout` is what proves those two agree.
    """
    pc = codec()
    V = as_fit_dtype(V, "value")
    H, n, d = V.shape
    if (H, d) != (lay.H, lay.d):
        raise ValueError(f"V is [H={H}, n={n}, d={d}] but the layout is for H={lay.H}, d={lay.d}")
    return pc.build_v_layer(V.contiguous(), lay.v_lanes, lay.v_bits, d=lay.d).rows


def check_slot_ids(slot_ids: torch.Tensor, n_slots: int) -> None:
    """The write's two refusals, in one place: `PAD_SLOT_ID`, and past the end of the cache.

    Both are readbacks (`min`/`max` on a device tensor), which is why `impl._Step` validates a whole
    step's mapping ONCE on the host and its callers pass `check=False` -- per layer, two drains is
    what the decode profile was mostly made of.

    A missing key is not a shape error. Decode reads exactly `seq_len` rows, so a token whose slot was
    never written silently drops its key from the softmax denominator instead of shortening the read,
    and a slot id past the end addresses another request's row. Neither raises on its own.
    """
    if slot_ids.numel() == 0:
        return
    lo, hi = int(slot_ids.min()), int(slot_ids.max())
    if lo < 0:
        raise RuntimeError(
            f"slot_mapping contains {lo}, i.e. vLLM's PAD_SLOT_ID. A padded token has no cache "
            f"slot, so its key cannot be stored -- and since decode reads exactly `seq_len` rows, a "
            f"missing one would silently drop a key from the softmax denominator rather than "
            f"shorten the read. Refusing instead.")
    if hi >= n_slots:
        raise IndexError(
            f"slot {hi} is outside the cache, which holds {n_slots} slots")


def write_slots(cache: torch.Tensor, slot_mapping: torch.Tensor, k_slots: torch.Tensor,
                v_slots: torch.Tensor, lay: TTKLayout, check: bool = True) -> None:
    """Scatter `n` tokens' blobs into the paged cache at vLLM's own slot ids.

    A slot id is `block * block_size + offset`, which is the convention `slot_mapping` carries and
    `decode.block_slots` reads -- so the write and the read agree by construction rather than by two
    matching assumptions.

    `check=False` skips the two device bounds checks (PAD_SLOT_ID and past-the-end), for a caller
    that has already validated the same `slot_mapping` on the host -- `impl._Step` reads it back once
    per step, because each of those two checks is a device readback and this function used to run
    once per layer.

    `view` rather than `reshape`, deliberately. `reshape` will silent-copy a non-contiguous tensor,
    and the write would then land in the copy and be discarded -- so this asks for a view and lets
    it raise. The BYTE view of the page comes from `decode.byte_rows`, for the reason recorded
    there: vLLM's page last dimension counts MODEL-DTYPE ELEMENTS, not bytes (measured: an 816-byte
    blob arrived as a 408-wide bf16 page), so a payload written without translating would land in
    the wrong half of every row.
    """
    from vllm_ttkv.decode import byte_rows

    C = lay.compressed_bytes_per_token()
    flat = byte_rows(cache, C)
    slots = slot_mapping.to(torch.long).reshape(-1)
    if slots.numel() != k_slots.shape[0]:
        raise ValueError(
            f"{k_slots.shape[0]} tokens to write but {slots.numel()} slot ids; these are two views "
            f"of the same token range, so a mismatch means tokens would go unwritten or overwritten.")
    if check:
        check_slot_ids(slots, flat.shape[0])
    payload = torch.cat([k_slots.contiguous(), v_slots.contiguous()], dim=-1)
    if flat.shape[-1] == C:
        flat[slots] = payload
    else:
        # The page was rounded up to whole elements, so only the blob's own bytes are ours and the
        # tail of each row belongs to nobody. Writing the full row would overwrite it.
        flat[slots, :C] = payload


def _fused_write(cache, slot_mapping, fk: FittedK, V: torch.Tensor, lay: TTKLayout,
                 check: bool) -> bool:
    """Pack this store straight into the cache's own bytes. `False` when the torch path has to.

    WHAT IT SAVES, MEASURED: the engine's own profile put the torch path at ~1.99 ms a layer between
    `proj.resid`, `pack.k` and `pack.v` (see `select_triton`'s header for why that is launch count
    rather than arithmetic). Against the other arm with the queue held deep the same store is **1.794
    -> 0.307 ms a layer-call, 5.84x** (`tests/bench_write_path.py`), which is what the decode step
    actually got. Nothing about the FORMAT changes: the same fields at the same offsets, and
    `tests/test_encode_triton.py` requires the bytes to be identical.

    THREE THINGS FALL BACK rather than fail, and each one has a real caller:

      * `resid_src is None` -- a store from the FIT path (a request's first chunk). The codec's
        `build_k_layer` derives its residual from the UNPADDED product internally, so recomputing it
        here to feed the kernel would be a second spelling of the residual. V could still be fused
        on its own (it never depends on the fit), and that is deliberately not done: the decision is
        per STORE so that a store has one writer, and the fit runs once per request on a path whose
        cost is the SVD. Measured, the fused V would save 26 ms of a 3.8 s prefill.
      * a geometry `select_triton.supported` refuses (a head dim that is not a power of two, a bit
        width outside 1..8).
      * a non-contiguous or non-fp V, which is only ever a caller error that the torch path reports
        with its own message.

    What does NOT fall back is validation: the slot ids get the same two refusals (`check_slot_ids`)
    before any byte is written, so `check=False` means "this step already validated them", exactly as
    it does on the torch path.
    """
    if fk.resid_src is None:
        return False
    # V is `[H, n, d]` here, the order `encode_v` takes (a chunk permuted to head-major). The kernel
    # wants the rows, and `reshape` is a view because the caller passes it contiguous.
    if V.dim() != 3 or V.shape[0] != lay.H or V.shape[2] != lay.d or not V.is_contiguous():
        return False
    n_tok = int(slot_mapping.numel())
    if fk.resid_src.shape[0] != n_tok * lay.H or V.shape[1] != n_tok:
        return False
    M = V.reshape(n_tok * lay.H, lay.d)

    from vllm_ttkv import select_triton
    from vllm_ttkv.decode import byte_rows
    # `pack.fused.prep`: this function's own preamble -- the shape checks, `supported()` on both
    # stores, and the byte view of the cache. It is host work with no launch in it, so if the write
    # is expensive this is where caching per (layer, device) actually helps.
    with prof.phase("pack.fused.prep"):
        if not select_triton.supported(fk.resid_src, lay.d, lay.res_k, lay.res_bits):
            return False
        if not select_triton.supported(M, lay.d, lay.v_lanes, lay.v_bits):
            return False
        coeff = fk.coeff_q.contiguous().view(torch.uint8)
        if coeff.shape != (n_tok, lay.coeff_bytes) or fk.coeff_scale.numel() != n_tok:
            return False
        flat = byte_rows(cache, lay.compressed_bytes_per_token())
    slots = slot_mapping.reshape(-1)
    if check:
        check_slot_ids(slots, flat.shape[0])

    # The coefficient block and its one-per-token ruler are contiguous at the head of every slot, so
    # they are one advanced-index assignment each rather than a kernel: the rows are the token's own.
    with prof.phase("pack.coeff"):
        flat[slots, :lay.coeff_bytes] = coeff
        flat[slots, lay.coeff_bytes:lay.coeff_bytes + 2] = \
            fk.coeff_scale.contiguous().view(torch.uint8).reshape(n_tok, 2)

    with prof.phase("pack.k"):
        select_triton.write_k_residual(flat, slots, fk.resid_src, lay)
    with prof.phase("pack.v"):
        select_triton.write_v_rows(flat, slots, M, lay)
    return True


def pack_and_write(cache: torch.Tensor, slot_mapping: torch.Tensor, fk: FittedK,
                   V: torch.Tensor, lay: TTKLayout, check: bool = True) -> None:
    """`FittedK` + V -> the cache. The one call the impl makes per request per step.

    TWO WRITERS, ONE FORMAT. The fused path (`_fused_write`) is what the decode step runs; the torch
    path below is what it has to reproduce, and it stays live as the fallback, as the definition of
    the format, and as the arm `VLLM_TTKV_TRITON_PACK=0` selects.
    """
    # `pack.fused.total` wraps the WHOLE fused write, so the difference between it and the phases
    # inside it is host work that no phase names. That distinction decides the fix: if the total is
    # mostly `pack.coeff`/`pack.k`/`pack.v` the answer is fewer kernels, and if it is mostly NOT,
    # the answer is somewhere in this function's own preamble and worth naming.
    with prof.phase("pack.fused.total"):
        if _fused_write(cache, slot_mapping, fk, V, lay, check):
            return
    with prof.phase("pack.k"):
        k_slots = pack_k_slots(lay, fk.coeff_q, fk.coeff_scale, fk.resid)
    with prof.phase("pack.v"):
        v_rows = encode_v(V, lay)
        v_slots = pack_v_slots(lay, v_rows)
    with prof.phase("pack.scatter"):
        write_slots(cache, slot_mapping, k_slots, v_slots, lay, check=check)
