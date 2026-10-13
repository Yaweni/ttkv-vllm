"""Python side of the fused container kernels: compile once per head config, launch, cache.

The PTX is built with NVRTC and specialised on (HQ, HD) because the inner loop over query heads has
to unroll to constants to be worth anything. That is a deliberate coupling: change the model's head
count and the module is recompiled, not silently mis-launched. `module()` caches per configuration.

Everything is launched on torch's current stream, so ordering against the torch ops around it is
automatic and nothing here needs to synchronise.
"""
import os

import torch

from cuda_driver import Module, compile_ptx, nvrtc_available

__all__ = ["kernels_available", "module", "qt", "resid_add", "v_out", "krt", "prime"]

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ttkv_kernels.cu")
_MODS = {}
_THREADS = 256

# Tokens per KRT tile. Mirrors TTKV_KT in the .cu (passed as a define) and sets two things at once:
# the grid, and the shared key tile. 64 keeps Ks at 64*129*4 = 33 KB, which still admits two blocks
# per SM on sm_86's 100 KB, while leaving the rotation a whole (j, j+TD/2) pair per thread.
KRT_TILE = 64

# Thread tiling for the reconstruction: KRT_MT tokens x KRT_MD lanes per output tile. Mirrors
# TTKV_KM / TTKV_KD. The product must fill the block exactly -- (64/4)*(128/8) = 256 -- because the
# kernel's barriers are outside its loops and a short block would reach them partially, which is
# undefined and in practice hangs. Asserted below rather than trusted.
KRT_MT = 4
KRT_MD = 8


def kernels_available():
    return nvrtc_available() and os.path.exists(_SRC)


def module(HQ, HD, G, arch=None):
    key = (HQ, HD, G)
    if key in _MODS:
        return _MODS[key]
    with open(_SRC) as fh:
        src = fh.read()
    if arch is None:
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}"
    ptx = compile_ptx(src, arch=arch, name=f"ttkv_kernels_HQ{HQ}_HD{HD}_G{G}.cu",
                      options=["--std=c++17", f"-DTTKV_HQ={HQ}", f"-DTTKV_HD={HD}",
                               f"-DTTKV_G={G}", f"-DTTKV_KT={KRT_TILE}",
                               f"-DTTKV_KM={KRT_MT}", f"-DTTKV_KD={KRT_MD}"])
    _MODS[key] = Module(ptx=ptx, name=f"ttkv_kernels HQ={HQ} HD={HD} G={G}")
    return _MODS[key]


def prime(HQ, HD, G):
    """Compile eagerly, so the first decode step does not pay for it."""
    return module(HQ, HD, G)


def _in_place_ok(s):
    assert s.dtype == torch.float32 and s.is_cuda, "s must be fp32 CUDA"
    assert s.is_contiguous(), "s must be contiguous: the kernel indexes it as p*n + i"


def _b3(t):
    """Accept a 2-D tensor as a batch of one, and remember to squeeze on the way out.

    The model's existing single-sequence path passes [HQ, hd] everywhere; the batched path passes
    [B, HQ, hd]. Supporting both in the launchers keeps the B=1 path byte-for-byte what it was, so a
    batching change can never silently alter the single-sequence result.
    """
    return (t.unsqueeze(0), True) if t.dim() == 2 else (t, False)


def qt(q, dec_q, dec_scale, r1, HQ, G, HD):
    """Q-absorption. q [B, HQ, HD] (or [HQ, HD]) fp32 -> QT [B, HQ, r1], no [r1, HQ, HD] copy.

    Replaces `einsum("apj,pj->pa", P_rep, q)` where P_rep was a per-layer, per-token gather of the
    decoder to [r1, HQ, HD]. The head offset is folded into the decoder row address instead.

    `dec_q` is [B, r1, H*HD] -- ONE DECODER PER SEQUENCE. A single [r1, H*HD] decoder is broadcast
    across the batch, which is what the current single-sequence model path passes. Sharing one
    fitted basis across independent sequences is NOT equivalent: pooled fitting measured a 5.8-point
    absolute score disagreement against a sequence's own B=1 run, because the basis has to span B
    sequences' key diversity at the same rank. Batching must not change the answer.
    """
    assert q.dtype == torch.float32
    q3, sq = _b3(q)
    B = q3.shape[0]
    H = HQ // G
    dq = dec_q if dec_q.dim() == 3 else dec_q.unsqueeze(0)
    ds = dec_scale if dec_scale.dim() == 2 else dec_scale.reshape(1, -1)
    if dq.shape[0] == 1 and B > 1:
        dq = dq.expand(B, -1, -1).contiguous()
        ds = ds.expand(B, -1).contiguous()
    QT = torch.empty(B, HQ, r1, device=q3.device, dtype=torch.float32)
    m = module(HQ, HD, G)
    m.launch("ttkv_qt", (HQ, (r1 + _THREADS - 1) // _THREADS, B), _THREADS,
             [q3.contiguous(), dq.contiguous(), ds.float().contiguous(), QT, r1, G, H])
    return QT[0] if sq else QT


def resid_add(s, q, rows, n, HQ, G, HD):
    """s [B, HQ, n] fp32, updated in place with the K residual. No [H*B*n, HD] buffer is ever built.

    `n` is the PER-SEQUENCE token count and the container is POOLED: row = h*(B*n) + b*n + i. That
    layout is what keeps this a grid-axis change rather than a rewrite -- a block owns one batch
    element, so the reduction over cached tokens never crosses a sequence boundary, and the only new
    arithmetic is the row offset.

    Equal-length sequences are required. Variable lengths would need a per-sequence row cursor rather
    than a uniform n, because a tile would then straddle two sequences' storage.
    """
    _in_place_ok(s)
    assert q.dtype == torch.float32
    q3, sq = _b3(q)
    if sq:
        s = s.unsqueeze(0)
    B = q3.shape[0]
    assert s.shape == (B, HQ, n), f"s {tuple(s.shape)} != ({B}, {HQ}, {n})"
    H = HQ // G
    assert rows.packed.is_cuda and rows.bitmap.is_cuda
    packed = rows.packed.contiguous()
    PB = packed.shape[1]
    assert int(rows.packed.shape[0]) == H * B * n, (
        f"container has {rows.packed.shape[0]} rows, expected H*B*n = {H * B * n}")
    m = module(HQ, HD, G)
    grid = ((n + _THREADS - 1) // _THREADS, H, B)
    diag = os.environ.get("TTKV_DIAG", "").strip() == "1"
    before = s.clone() if diag else None
    m.launch("ttkv_resid_add", grid, _THREADS,
             [q3.contiguous(), packed, rows.bitmap.contiguous(),
              rows.scale.float().contiguous(), s, n, PB, int(rows.bits)],
             shared=HD * G * 4)
    if diag:
        _diag_resid(before, s, q3, rows, n, HQ, G, HD, PB, B)
    return s[0] if sq else s


TV_TILE = 64


def v_out(p, rows, n, HQ, G, HD):
    """The V half of the output: p [B, HQ, n] fp32 -> o [B, HQ, HD] fp32.

    `p` is expected to be the container slice `p_attn[:, :, :n]`, so its row stride is n + Tt and is
    passed explicitly -- indexing it as though it were contiguous would read the tail's weights for
    every row past the first. Only the last dim is required to be stride 1.

    `o` is zeroed here because the kernel accumulates across token tiles with atomicAdd.
    """
    assert p.dtype == torch.float32
    assert p.stride(-1) == 1, "p must be contiguous along the token axis"
    p3, sp = _b3(p)
    B = p3.shape[0]
    H = HQ // G
    P4 = rows.packed.contiguous()
    o = torch.zeros(B, HQ, HD, device=p3.device, dtype=torch.float32)
    m = module(HQ, HD, G)
    grid = ((n + TV_TILE - 1) // TV_TILE, H, B)
    m.launch("ttkv_v_out", grid, 256,
             [p3, P4, rows.bitmap.contiguous(), rows.scale.float().contiguous(),
              o, n, p3.stride(1), P4.shape[1], int(rows.bits)],
             shared=(TV_TILE * HD + G * TV_TILE) * 4)
    return o[0] if sp else o


def krt(coeff_q, coeff_scale, dec_q, dec_scale, resid, cos, sin, q, n, HQ, G, HD, mode=7):
    """s [HQ, n] fp32 -- post-RoPE scores straight from a PRE-ROPE store. The pre-RoPE twin of
    `qt` + `resid_add`, fused.

    `mode` exists to DECOMPOSE the cost, not to be used in production: bit 1 the reconstruction,
    bit 2 the residual, bit 4 the rotation and contraction. Timing the pieces separately is the
    only way to know which one to attack -- this project has had six isolated benchmarks that
    failed to predict the loop, all of them because the cost was assumed rather than isolated.
    A mode that drops bit 1 leaves `Ks` uninitialised shared (fine to read, wrong to trust), so
    the reduced modes are for timing only and must never be asserted against.

    What it keeps from that pair is the reason they were worth writing: no dense [H*n, HD] block, no
    masked_scatter_, and the residual's bitmap scatter amortised across the TG query heads sharing a
    kv head. What it cannot keep is the query absorption -- every stored row carries its own
    position, so the rotation has to land between the reconstruction and the contraction, per tile,
    in shared memory. That is the entire cost of pre-RoPE storage, and it is why `attend_container`
    used to refuse this case outright.

    THE STORE MUST BE PRE-ROPE, and the caller must know that it is. Handed a post-RoPE store this
    does not fail loudly -- it rotates already-rotated keys, which is a well-defined thing to do to
    well-formed vectors, and returns scores of the right magnitude and the wrong value. The check
    belongs where the storage mode is recorded, not here.

    `q` is the model's own post-RoPE query [HQ, HD]. `cos`/`sin` are [n, HD] fp32 in the frame
    `rope_apply` uses, indexed by ABSOLUTE position -- that indexing is what makes the tile loop's
    `i0 + t` correct rather than merely plausible.

    BATCH IS NOW A GRID AXIS, matching `qt`/`resid_add`/`v_out` -- launch count per token is unchanged
    at 96 regardless of B, so the ~8.7 us/launch toll does not multiply. Two row conventions coexist
    inside a pooled store and getting either wrong is SILENT:
        coeff_q / coeff_scale   (batch, token)        = bz*n + i
        residual bitmap/packed  (batch, head, token)  = bz*(H*n) + hv*n + i
    `cos`/`sin` stay [n, HD] at every B: every sequence starts at position 0, so the ramp is SHARED
    and batching costs ZERO extra angle memory. At bz == 0 every kernel offset reduces to the
    single-sequence expression, so B=1 is unchanged to the byte.

    `shared=` is deliberately left at 0: the key tile is STATIC (`__shared__ float Ks[KT][TD+1]`,
    ~33 KB). Asking for dynamic bytes the kernel never declares would reserve them anyway and cut
    resident blocks per SM for nothing.

    THE DECODER IS CONVERTED HERE, not in the kernel. `dec_f = dec_q * dec_scale` is exactly what
    `PackedKLayer.decoder_f()` returns and exactly what torch's `materialise` multiplies against,
    so handing the kernel the fp32 product removes a convert AND a multiply from every MAC while
    making the kernel's expression match the reference term for term. It is a compute-time
    transient: the ledger still bills the decoder at `qbits`.
    """
    assert q.dim() in (2, 3), f"krt takes [HQ, HD] or [B, HQ, HD], got {tuple(q.shape)}"
    two_d = q.dim() == 2
    q3 = q.unsqueeze(0) if two_d else q
    assert q3.dim() == 3 and q3.shape[1] == HQ, (
        f"krt takes [B, HQ, HD] or [HQ, HD], got {tuple(q.shape)}")
    B = int(q3.shape[0])
    assert coeff_q.dtype == torch.int8 and dec_q.dtype == torch.int8, "coeff_q/dec_q are int8"
    assert int(coeff_q.shape[0]) == B * n, (
        f"coeff_q has {int(coeff_q.shape[0])} rows, expected B*n = {B * n} (row = bz*n + i)")
    H = HQ // G
    n_rank = int(coeff_q.shape[1])
    assert (KRT_TILE % KRT_MT) == 0 and (HD % KRT_MD) == 0, "thread tiling must divide the tile"
    assert (KRT_TILE // KRT_MT) * (HD // KRT_MD) == _THREADS, (
        f"({KRT_TILE}/{KRT_MT})*({HD}/{KRT_MD}) must equal {_THREADS}; the kernel's barriers are "
        f"outside its loops, so a short block would deadlock rather than mis-compute")
    if resid is None:
        # has_res=0 means these are never dereferenced; they exist only because the launcher cannot
        # marshal a null pointer.
        z = torch.zeros(1, device=q.device, dtype=torch.uint8)
        bm, pk, rs, PB, bits, has = z, z, z.float(), 1, 1, 0
    else:
        bm, pk, rs = resid.bitmap.contiguous(), resid.packed.contiguous(), resid.scale.float()
        PB, bits, has = int(resid.packed.shape[1]), int(resid.bits), 1
        assert int(resid.packed.shape[0]) == B * H * n, (
            f"residual has {int(resid.packed.shape[0])} rows, expected B*H*n = {B * H * n} "
            f"(slab order = bz*(H*n) + hv*n + i, NOT coeff_q's bz*n + i)")
    s = torch.empty(B, HQ, n, device=q.device, dtype=torch.float32)
    if dec_q.dim() == 2:
        dec_f = (dec_q.to(torch.float32) * dec_scale.to(torch.float32).unsqueeze(1)).contiguous()
    else:
        # PooledKStore.decoder_f() -> [B, n_rank, H*HD]: the scale broadcasts over the LAST dim.
        assert int(dec_q.shape[0]) == B, f"decoder has {int(dec_q.shape[0])} sequences, expected {B}"
        dec_f = (dec_q.to(torch.float32) * dec_scale.to(torch.float32).unsqueeze(-1)).contiguous()
    m = module(HQ, HD, G)
    grid = ((n + KRT_TILE - 1) // KRT_TILE, H, B)
    m.launch("ttkv_krt", grid, _THREADS,
             [coeff_q.contiguous(), coeff_scale.float().contiguous(), dec_f,
              bm, pk, rs, cos.contiguous(), sin.contiguous(), q3.contiguous(), s,
              n, n_rank, G, H, PB, bits, has, mode])
    return s[0] if two_d else s


# ------------------------------------------------------------------ diagnostic only# TTKV_DIAG=1 compares the kernel's delta against the torch formulation it replaced, once per
# distinct container shape, and dumps the shape facts the kernel ASSUMES. The kernel is verified
# exact on a container built by `build_k_layer`; a container built by the DECODE LOOP has been
# appended to and folded, so every one of those assumptions is worth printing rather than trusting.
_DIAG_SEEN = set()


def _diag_resid(before, after, q, rows, n, HQ, G, HD, PB, B):
    import ttkv_packed_cache as pcm
    key = (int(rows.rows), int(rows.k), int(rows.bits), int(PB), int(n), B)
    if key in _DIAG_SEEN:
        return
    _DIAG_SEEN.add(key)
    H = HQ // G
    bits = int(rows.bits)
    r_code = pcm.unpack_bits_rows(rows.packed, rows.k, bits).float() - (2 ** (bits - 1))
    r_mask = pcm.unpack_bitmap_rows(rows.bitmap, HD)
    r_scale = rows.scale.float().reshape(int(rows.rows), 1)
    Dd = torch.zeros(int(rows.rows), HD, device=q.device, dtype=torch.float32)
    Dd.masked_scatter_(r_mask, r_code * r_scale)
    # each sequence owns a contiguous slab, so unpooling is reshape(B, H, n, HD)
    D4 = Dd.reshape(B, H, n, HD)
    ref = torch.stack([torch.bmm(q[b].reshape(H, G, HD),
                                 D4[b].transpose(1, 2)).reshape(HQ, n)
                       for b in range(B)])
    delta = after - before
    cos = float(torch.nn.functional.cosine_similarity(delta.flatten(), ref.flatten(), dim=0))
    print(f"[diag resid] rows={int(rows.rows)} H*B*n={H * B * n} n={n} B={B} k={rows.k} "
          f"bits={bits} PB={PB} bm_cols={int(rows.bitmap.shape[1])}")
    print(f"             max|kernel-torch|={float((delta - ref).abs().max()):.3e} "
          f"max|ref|={float(ref.abs().max()):.3e} cosine={cos:.6f}")
