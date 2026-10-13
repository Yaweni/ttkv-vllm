"""TTKV codec, ported out of ttkv_roof_global.ipynb / ttkv_babilong_reasoning.ipynb.

Self-contained and free of hooks, so it can be applied to a transformers Cache after a prefill
instead of intercepting k_proj/v_proj. The maths is copied exactly from the notebooks; only the
calling convention is new.

    K block [H, n, d]:  TT-SVD on the [n, d, H] rearrangement (TOKENS FIRST, load-bearing) with
                        truncation by dropped energy <= tol**2, 8-bit cores, then a per-row
                        magnitude residual K - K_TT in a d-bit bitmap container.
    V block [H, n, d]:  per-row top-k of |V|, `bits`-bit values, one fp16 scale per row, d-bit
                        bitmap. No shared basis -- V is high-rank per token by measurement, so a
                        shared low-rank basis destroys it (+75 to +700 PPL).

Byte ledger, per row, per side (d = 128):
    K = cores_elems*qbits/8 / rows  +  CORE RULERS  +  16 B bitmap  +  2 B scale
        +  kept_res*res_bits/8
    V =                                                16 B bitmap  +  2 B scale  +  k_lanes*bits/8
    joint ratio = 2*d*2 / (K_row + V_row)

The CORE RULERS are `2 * sum(rows per core)` bytes per layer, which at n=3064 is ~0.25 B/row. They
used to be omitted from the hook ledger entirely while `PackedKLayer.nbytes()` counted them, so the
hook and container ratios were measured on different ledgers. Billing them costs the joint ratio
~0.2-0.4% (tight 7.092 -> 7.067, safe 5.12 -> 5.11, campaign4x 4.252 -> 4.243).

THREE CONFIGS, TWO OF THEM SHIPPABLE. Renamed 2026-09-30; the old names are deprecated aliases.

    safe         5.12x   RES(0.633,3) = 81 lanes @3b, V 81 @3b    THE DEFAULT
    tight        7.09x   RES(0.25,3)  = 32 lanes @3b, V 45 @3b    opt-in, costs 8-12 points
    campaign4x   4.25x   RES(0.50,4)  = 64 lanes @4b, V 64 @4b    ARCHIVED, reproduces published rows

THE TWO HALVES OF THE PROJECT MEASURE DIFFERENT THINGS, AND THIS IS THE RULE:

    the CONTAINER  measures SPEED and holds the kernels. Its job is ms/token, peak VRAM and capacity.
    the HOOKS      measure ACCURACY. `safe` and `tight` are hook-family allocations, and every
                   accuracy number attached to them comes from ttkv.ipynb's hook ladder.

So `campaign` is the HOOK-FAMILY LEDGER ratio -- the same kind of number as the campaign's 4.252 and
7.092. `ratios_from_bytes()` over `nbytes()` is a measurement; this field is a label. Do not expect
the container to reproduce it, and do not run the container as an accuracy arm to check.

Why they differ, precisely: a hook arm reads `k_proj`/`v_proj`, which is PRE-RoPE. The container
compresses the POST-RoPE cache. Same lanes, different tensor, and post-RoPE needs ~3.6x more rank for
the same error budget (ROPE_DISPERSION.md), so by the ledger at n=16384 on prose the container's K_row
is ~94 B against ~61 B. Storing pre-RoPE is what closes the gap -- that is the port's entire argument.

The container DOES appear in `ttkv.ipynb`'s accuracy ladder as `syspre_*`/`syspos_*`. That is a
CROSS-IMPLEMENTATION CHECK and nothing more: two implementations of the same format landing on the
same score is the validation. It is never a headline, and a container row must never be compared
against a hook row as if the difference were a result.

BOTH CONFIGS RUN A UNIFORM TOLERANCE OF 0.20. An earlier version of this docstring claimed
`loose_late` carried a per-layer schedule (0.05 on layers 0-15, 0.20 on 16-31). That schedule is the
WITHDRAWN `loose-late16` arm, not this format, and the arithmetic rejects it: scheduled r1 runs ~340
mean = 53 B/row of cores = a 3.34x joint ratio, against the published 4.252x for ttkv_ll -- a 21%
gap that is not item variance. Uniform 0.20 reproduces 4.34x, within 2%. The depth non-uniformity is
real and measured (on real K at n=4096, r1 runs 400-620 on the early layers against 93-154 on the
late ones), but it is a DESCRIPTIVE property of the key block under uniform tol -- not a schedule
this config runs, and not a lever that beat uniform when it was tried (TTKV_PRIMER ch. 9).

RANKS ARE NOT CAPPED. The press's `trunc_rank` is bounded only by `s.numel()`, which for the first
unfolding is min(n, d*H) = 1024. Measured: loose_late r1 in 93..616 (median 279), core8 r1 in
81..154 (median 112). A max_rank of 64 under-ranks core8 by 1.75x and loose_late by ~4.4x, which
changes the bytes AND the accuracy -- so it silently runs a third format.

WARNING about self_test(): it runs on WHITE NOISE, which has no low-rank structure, so the TT ranks
come out huge and the measured ratios will look far worse than the campaign numbers. It is a smoke
test for correctness, not a ratio check. Real ratios need real K/V, and are measured in the
notebook against a live cache.
"""
import math

import torch

DEFAULT_D = 128


# --------------------------------------------------------------------------- TT primitives
def tt_svd(X, tol=0.20, max_rank=None):
    """X: [d0, d1, d2] -> (cores, ranks). Truncation is by DROPPED ENERGY, not fixed rank: keep the
    smallest rank whose dropped energy fraction is <= tol**2, so tol 0.20 keeps >= 96% of energy.

    max_rank is None by default because the canonical implementation has no cap -- `trunc_rank` is
    bounded only by `s.numel()`. Passing 64 here is not a small change: measured on real K it
    under-ranks core8 by 1.75x and loose_late by ~4.4x, changing both the bytes and the accuracy.

    Mode order is the caller's choice and it is load-bearing. The K block is passed [n, d, H] --
    TOKENS FIRST -- so the first unfolding is n x (d*H) and its rank counts independent key
    archetypes. Passing [H, n, d] gives an H x (n*d) unfolding with r1 <= H = 8, which buys
    nothing at all.
    """
    dims = list(X.shape)
    rest = X.reshape(dims[0], -1).contiguous()
    cores, ranks, r_prev = [], [], 1
    for k in range(len(dims) - 1):
        M = rest.reshape(r_prev * dims[k], -1).contiguous()
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        e = S.pow(2)
        tail = torch.flip(torch.cumsum(torch.flip(e, [0]), 0), [0]) / (e.sum() + 1e-30)
        hit = torch.nonzero(tail <= tol ** 2, as_tuple=False)   # tail[r-1] = dropped fraction
        r = int(hit[0].item()) + 1 if hit.numel() else int(S.numel())
        r = max(1, min(r, int(S.numel()))) if max_rank is None \
            else max(1, min(r, max_rank, int(S.numel())))
        ranks.append(r)
        cores.append(U[:, :r].reshape(r_prev, dims[k], r).contiguous())
        rest = (S[:r].unsqueeze(1) * Vh[:r, :]).reshape(r, *dims[k + 1:]).contiguous()
        r_prev = r
    cores.append(rest.reshape(r_prev, dims[-1], 1).contiguous())
    ranks.append(1)
    return cores, ranks


def tt_reconstruct(cores):
    """Contract the chain back to the original [d0, d1, d2] shape."""
    dims = [int(c.shape[1]) for c in cores]
    X = cores[0]
    for c in cores[1:]:
        X = torch.tensordot(X, c, dims=([-1], [0]))
    return X.reshape(*dims)


def _core_rowcount(c):
    """How many rulers a core gets. The token-coefficient core is `[1, n, r1]` -- ~75% of the core
    bytes at every shipped config -- and reshaping THAT by mode 0 gives one row, so a "per-row"
    ruler silently became one global scalar for the whole n x r1 table. Its real rows are mode 1:
    one ruler per token, which is the container's own convention (`PackedKLayer.coeff_scale` is
    `[n]`). Anything else is already mode-0-major and is left alone.
    """
    if PER_TOKEN_CORE_RULER and int(c.shape[0]) == 1:
        return int(c.shape[1])
    return int(c.shape[0])


def core_ruler_bytes(cores, dtype_bytes=2):
    """Bytes the rulers cost. `elements * qbits / 8` does not count them, and they are real: the
    container bills exactly these (`PackedKLayer.nbytes` counts `coeff_scale` and `dec_scale`). The
    hook ledger did not, so the two implementations were reporting ratios on different ledgers.
    At n=3064 the missing term is `2n + 2*r1 + 2*r2` bytes per layer, about 0.25 B/row.
    """
    return float(sum(_core_rowcount(c) for c in cores) * dtype_bytes)


def quant_cores(cores, qbits):
    """Per-ROW symmetric quantization, matching the press's `_qb`.

    One scale per row of each core, not one scale per core. The distinction matters because a
    per-row ruler is what keeps a small row from being rounded to zero by a large one.

    WHICH ROWS, THOUGH. That rule as first written reshaped every core by its mode 0, and the core
    it most needed to protect -- the token-coefficient core `[1, n, r1]` -- has mode 0 == 1. So the
    biggest core got exactly the single global scale the docstring warned against, and the warning
    was inert for 75% of the bytes. `_core_rowcount` gives that core n rulers instead, one per
    token. Set `PER_TOKEN_CORE_RULER = False` to get the old behaviour back.

    The scales are real bytes; `core_ruler_bytes` prices them and `compress_k` now bills them.
    """
    q = 2 ** (qbits - 1) - 1
    out = []
    for c in cores:
        shape = tuple(c.shape)
        M = c.detach().reshape(_core_rowcount(c), -1)
        s = M.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / q
        out.append((torch.clamp(torch.round(M / s), -q, q) * s).reshape(shape).contiguous())
    return out


# --------------------------------------------------------------------------- K and V blocks
# Per-TOKEN ruler on the token-coefficient core, and the rulers billed. Both are the container's
# existing convention (`build_k_layer` reshapes G1 to `[n, r1]` and stores `coeff_scale` as `[n]`,
# and `nbytes()` counts it), so this makes the hook family match the systems half instead of
# diverging from it. False reproduces the pre-2026-10-01 behaviour; the flag exists only so an A/B
# can run through the SHIPPED maths rather than through a second copy of it.
PER_TOKEN_CORE_RULER = True


def compress_k(X, tol=0.20, qbits=8, res_frac=0.25, res_bits=3, max_rank=None, d=None):
    """X: [H, n, d] float32 -> (Yk [H, n, d] float32, bytes, rows, ranks).

    max_rank defaults to None (uncapped), matching the canonical implementation. It used to default
    to 64 here AND in tt_svd AND in build_k_layer, which is why `ranks=[64, 8, 1]` appeared in every
    run of every config -- three separate caps, each of which alone was enough to replace the format.
    """
    H, n, dd = X.shape
    d = d or dd
    Xt = X.permute(1, 2, 0).contiguous()                       # [n, d, H] -- tokens first
    cores, ranks = tt_svd(Xt, tol, max_rank)
    cores = quant_cores(cores, qbits)
    n_el = sum(int(c.numel()) for c in cores)
    Kr = tt_reconstruct(cores).permute(2, 0, 1).contiguous()   # [H, n, d]

    rows = H * n
    M = X.reshape(rows, d)
    Yk = Kr.reshape(rows, d).clone()
    # The rulers are billed, because they are stored. Without this term the hook ledger omits what
    # the container's `nbytes()` includes, and the two implementations' ratios are not comparable.
    nbytes = n_el * qbits / 8.0 + core_ruler_bytes(cores)

    if res_frac > 0:
        R = M - Yk
        kr = max(int(d * res_frac), 1)
        mask = torch.zeros_like(R, dtype=torch.bool)
        mask.scatter_(1, torch.topk(R.abs(), kr, dim=1).indices, True)
        rmax = torch.where(mask, R.abs(), torch.zeros_like(R)).amax(1, keepdim=True)
        scl = torch.where(rmax == 0, torch.ones_like(rmax), rmax) / (2 ** (res_bits - 1) - 1)
        q = torch.clamp(torch.round(R / scl), -(2 ** (res_bits - 1)), 2 ** (res_bits - 1) - 1)
        Yk = Yk + (q.float() * scl) * mask
        nbytes += rows * (d // 8) + int(mask.sum().item()) * res_bits / 8.0 + rows * 2.0
    return Yk.reshape(H, n, d), nbytes, rows, ranks


def compress_v(X, k_lanes=45, bits=3, d=None):
    """X: [H, n, d] float32 -> (Yv, bytes, rows). Fixed per-row k lanes of largest |value|."""
    H, n, dd = X.shape
    d = d or dd
    rows = H * n
    M = X.reshape(rows, d)
    mask = torch.zeros_like(M, dtype=torch.bool)
    mask.scatter_(1, torch.topk(M.abs(), k_lanes, dim=1).indices, True)
    rmax = torch.where(mask, M.abs(), torch.zeros_like(M)).amax(1, keepdim=True)
    scl = torch.where(rmax == 0, torch.ones_like(rmax), rmax) / (2 ** (bits - 1) - 1)
    q = torch.clamp(torch.round(M / scl), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    Yv = (q.float() * scl) * mask
    nbytes = rows * (d // 8) + rows * k_lanes * bits / 8.0 + rows * 2.0
    return Yv.reshape(H, n, d), nbytes, rows


# --------------------------------------------------------------------------- the three configs
def _tols(spec, NL, strict=True):
    """Scalar -> repeated once per layer. A list must match NL exactly unless strict=False, where
    it is sliced -- a convenience for the self-test, never for a real model, because a mismatched
    list means layers get the wrong tolerance and nothing downstream would notice."""
    if isinstance(spec, (list, tuple)):
        if len(spec) != NL:
            assert not strict, f"tol list has {len(spec)} entries, model has {NL} layers"
            spec = list(spec)[:NL]
        return list(spec)
    return [float(spec)] * NL


class _Configs(dict):
    """The config table, with DEPRECATED-NAME resolution.

    Renamed 2026-09-30. The old names were fossils: `loose_late` described a per-layer tolerance
    SCHEDULE ("loose on the late layers") that was withdrawn before this codec existed and that
    this config never ran -- it runs uniform 0.20 -- and `core8`/`balanced` named neither a
    setting nor a role. The new names say what the config IS or what it is FOR.

    `__missing__` is reached by `__getitem__` and by NOTHING ELSE: `in`, `.keys()`, `.items()` and
    `.get()` all ignore it. That is exactly the behaviour needed here, because `self_test()` and
    `compress_cache` ITERATE this table -- an alias stored as a real key would silently
    double-report every config in every ratio table.
    """
    _ALIAS = {
        "loose_late": "campaign4x",   # same allocation, archived name
        "core8":      "tight",        # same allocation, so every timed script is unchanged
        "balanced":   "safe",
        # core8_t35 / core8_t40 are NOT aliased. They were diagnostic arms for the closed question
        # "which tolerance reproduced the published 7.092x", and the answer is in the record. A
        # KeyError is the correct response to asking for them.
    }

    def __missing__(self, key):
        new = self._ALIAS.get(key)
        if new is None:
            raise KeyError(f"{key!r} is not a config and is not a deprecated alias")
        return self[new]


# EVERY CONFIG RUNS A UNIFORM TOLERANCE OF 0.20. The withdrawn per-layer schedule (0.05 on layers
# 0-15, 0.20 on 16-31) is rejected by arithmetic, not by taste: on real K it needs r1 ~ 340 mean =
# 53 B/row of cores = a 3.34x joint ratio, against the published 4.252x. A 21% gap is not item
# variance. Uniform 0.20 gives 4.34x, within 2%. `tols` is the press's own dataclass default.
CONFIGS = _Configs({
    # ---------------------------------------------------------------- the release pair
    "safe": dict(        # 5.12x. THE DEFAULT. No measurable loss at either length.
        # 81 lanes/side @3b, SOLVED from the byte budget rather than tuned -- the `ours_m3`
        # allocation, ttkv.ipynb 2.3. Measured at 16k: 100% against dense's 96%. At 3k: 98% against
        # dense's 100%, i.e. one item at n=50, which is not a measurable difference.
        # Every other rung pays for its bytes: `tight` is 84% at 16k, twelve points down, and the
        # 2-bit allocations collapse outright (ours_m2 16% at 16k, kivi2 12%).
        # Quote as "no measurable loss at ~5x". This is the point a release defaults to.
        tols=0.20, qbits=8, res_frac=0.633, res_bits=3, k_lanes=81, vbits=3, campaign=5.12,
    ),
    "tight": dict(       # 7.09x campaign / 7.45x measured. THE OPT-IN, and it carries a warning.
        # K residual 32 lanes @3b, V 45 lanes @3b. This is the format the campaign's RULER 94.97 row
        # ran under, and the format every recorded serving number uses.
        # IT COSTS ACCURACY AND THE COST IS LARGE: 92% at 3k and 84% at 16k, against dense's 100%
        # and 96%. Quote it as "7x, and you pay 8-12 points for it" -- never as a free option.
        tols=0.20, qbits=8, res_frac=0.25, res_bits=3, k_lanes=45, vbits=3, campaign=7.092,
    ),
    # ---------------------------------------------------------------- archived, do not ship
    "campaign4x": dict(  # 4.252x. KEPT ONLY TO REPRODUCE PUBLISHED NUMBERS. Not a release point.
        # 64 lanes @4b on both sides. This is the campaign's `ttkv_ll` row: RULER 95.35 at 4.252x.
        # It is ARCHIVED because the release pair DOMINATES it -- `safe` holds 5.08x at 98.0% where
        # this holds 4.25x at 100%, so it buys nothing a user would want. It stays in the table
        # because a published ratio has to remain reproducible, and that is its only job.
        tols=0.20, qbits=8, res_frac=0.50, res_bits=4, k_lanes=64, vbits=4, campaign=4.252,
    ),
})


def ratios_from_bytes(k_bytes, v_bytes, k_rows, v_rows, d=DEFAULT_D):
    """Never derive a ratio from configuration intent -- only from measured bytes."""
    kx = (d * 2) / (k_bytes / k_rows) if k_rows else 1.0
    vx = (d * 2) / (v_bytes / v_rows) if v_rows else 1.0
    return kx, vx, 2.0 / (1.0 / kx + 1.0 / vx)


def compress_cache(cache, cfg, NL, d=DEFAULT_D, verbose=True):
    """Rewrite every layer of a transformers Cache with its TTKV approximation, IN PLACE, keeping
    the dense shapes. Returns the measured ratios and the per-layer byte totals.

    Dense shapes are deliberate: the running model still sees fp16 for its full context, so this
    measures the APPROXIMATION, not the bytes. The capacity claim needs the packed container to be
    held and expanded per layer, which is a separate build.
    """
    tol_list = _tols(cfg["tols"], NL)
    k_bytes = v_bytes = 0.0
    k_rows = v_rows = 0
    kk = []
    for li in range(NL):
        L = cache.layers[li]
        k = L.keys[0].float()                                   # [H, n, d]
        v = L.values[0].float()
        Yk, kb, kr_, _ = compress_k(k, tol=tol_list[li], qbits=cfg["qbits"],
                                    res_frac=cfg["res_frac"], res_bits=cfg["res_bits"], d=d)
        Yv, vb, vr_ = compress_v(v, k_lanes=cfg["k_lanes"], bits=cfg["vbits"], d=d)
        L.keys = Yk.to(L.keys.dtype).unsqueeze(0)
        L.values = Yv.to(L.values.dtype).unsqueeze(0)
        k_bytes += kb; v_bytes += vb; k_rows += kr_; v_rows += vr_
        kk.append((Yk - k).abs().mean().item())
        if verbose and (li % 8 == 7 or li == 0):
            kx, vx, jx = ratios_from_bytes(k_bytes, v_bytes, k_rows, v_rows, d)
            print(f"    layer {li:2d}  running K={kx:5.2f}x V={vx:4.2f}x joint={jx:5.2f}x", flush=True)
    kx, vx, jx = ratios_from_bytes(k_bytes, v_bytes, k_rows, v_rows, d)
    return dict(kx=kx, vx=vx, joint=jx, k_bytes=k_bytes, v_bytes=v_bytes,
                k_rows=k_rows, v_rows=v_rows, mean_abs_k_err=sum(kk) / len(kk))


# --------------------------------------------------------------------------- smoke test
def self_test(NL=4, n=512, H=8, d=128, seed=0):
    torch.manual_seed(seed)
    print(f"self test on WHITE NOISE: NL={NL} n={n} H={H} d={d}")
    print("  (white noise has no low-rank structure, so ranks -- and therefore the ratio -- will")
    print("   be far worse than the campaign numbers. This checks correctness, not ratio.)")
    total = {k: [0.0, 0.0, 0, 0] for k in CONFIGS}
    for li in range(NL):
        K = torch.randn(H, n, d) * 0.2
        V = torch.randn(H, n, d) * 0.2
        for nm, cfg in CONFIGS.items():
            tol = _tols(cfg["tols"], NL, strict=False)[li]
            Yk, kb, kr_, ranks = compress_k(K, tol=tol, qbits=cfg["qbits"],
                                            res_frac=cfg["res_frac"], res_bits=cfg["res_bits"], d=d)
            Yv, vb, vr_ = compress_v(V, k_lanes=cfg["k_lanes"], bits=cfg["vbits"], d=d)
            t = total[nm]
            t[0] += kb; t[1] += vb; t[2] += kr_; t[3] += vr_
            if li == 0:
                print(f"  {nm:11s} ranks={ranks} K err={float((Yk-K).abs().mean()):.4f} "
                      f"V err={float((Yv-V).abs().mean()):.4f}")
    print()
    print(f"  {'config':11s} {'K':>7s} {'V':>7s} {'joint':>8s} {'campaign':>9s} {'rows':>8s}")
    for nm in CONFIGS:
        kb, vb, kr_, vr_ = total[nm]
        kx, vx, jx = ratios_from_bytes(kb, vb, kr_, vr_, d)
        print(f"  {nm:11s} {kx:6.2f}x {vx:6.2f}x {jx:7.2f}x {CONFIGS[nm]['campaign']:8.2f}x {kr_:8d}")
    print("\n  a ratio near 1.0x here is EXPECTED (noise is not compressible) and is not a failure.")


if __name__ == "__main__":
    self_test()
