"""TTKV packed cache: the codec stops returning dense fp16 and becomes the cache.

The container
-------------
A side of a layer's cache is stored as, per (head, token) row:
    bitmap   d bits          16 B   which lanes are stored
    values   k * bits / 8    12 B   at 32 lanes @3b, 32 B at 64 lanes @4b
    scale    1 fp16           2 B   one ruler per row
plus, for K only, a per-token coefficient row and one shared decoder per layer.

K is low-rank ACROSS tokens, so it does not need a per-row dictionary at all: the token axis is
compressed to r1 coefficients (one row of G1 per token, SHARED across the 8 heads), and the full
[H, n, d] block is rebuilt by one matmul against a frozen [r1, H*d] decoder. That is why K's
dense 67 MB at n=32768 becomes ~750 KB of coefficients, and it is also why decode can append a
token with one r1-sized least-squares instead of a refit.

V is high-rank per token (measured), so it keeps the per-row container.

Materialisation
---------------
`materialise()` returns the dense [H, n, d] fp16 block for ONE layer. That is deliberate: the
compressed store is what persists, and exactly one layer is ever dense, so peak VRAM is
    weights + compressed store + one layer's dense K/V
instead of weights + the whole dense cache. At n=96k that is the difference between fitting and
not fitting on a 16 GB card.

Cache integration
-----------------
transformers 5.14 made `DynamicCache` a container of per-layer objects:
    keys, values = self.layers[layer_idx].update(key_states, value_states, *args, **kwargs)
so the extension point is a per-layer class, not the cache. `TTKVLayer` subclasses `DynamicLayer`
and overrides `update` to append the new rows compressed and return the materialised full K/V.

Packing convention: MSB-first within each byte, matching numpy's packbits/unpackbits, with signed
values stored offset by 2**(bits-1).
"""
import math

import numpy as np
import torch

D = 128

# The three codec configs live with the codec. Re-exported so a caller needs one import rather than
# two, and so there is exactly one definition of each config.
from ttkv_codec import CONFIGS  # noqa: E402,F401


# ------------------------------------------------------------------ bit packing
def pack_bits_rows(vals_int, bits):
    """vals_int: [rows, k] non-negative ints < 2**bits -> uint8 [rows, ceil(k*bits/8)].

    MSB-first, and EACH ROW IS PADDED TO A WHOLE NUMBER OF BYTES. That padding is what makes the
    rows byte-aligned, which in turn makes an append a plain `torch.cat` along dim 0 instead of
    rebuilding the container. At 3 bits and 45 lanes it costs 1 byte per row.

    Device-only: numpy's packbits would force a D2H copy per layer per step.
    """
    rows, k = vals_int.shape
    shifts = torch.arange(bits - 1, -1, -1, device=vals_int.device)
    b = ((vals_int.to(torch.int64).unsqueeze(-1) >> shifts) & 1).to(torch.uint8).reshape(rows, k * bits)
    pad = (-(k * bits)) % 8
    if pad:
        b = torch.cat([b, torch.zeros(rows, pad, dtype=torch.uint8, device=b.device)], dim=1)
    w = (1 << torch.arange(7, -1, -1, device=b.device)).to(torch.uint8)
    return (b.reshape(rows, -1, 8) * w).sum(-1).to(torch.uint8)


def unpack_bits_rows(packed_u8, k, bits):
    """Inverse of pack_bits_rows, device-only: uint8 [rows, B] -> int32 [rows, k].

    The DTYPE OF THE INTERMEDIATE IS THE COST, which is why this has two paths. Measured at
    n=4096 (one layer, rows = H*n = 32768): the old single-path version cost 1.122 ms and was the
    single most expensive stage of a packed layer -- 22.9% of the whole K+V decode, against 2.0%
    for the matmul everyone was busy optimising.

      - `bits` dividing 8 (1/2/4/8): every byte holds 8//bits whole VALUES, so they fall out of one
        shift and one mask with no [rows, k*bits] intermediate at all. This is exact rather than
        lucky: pack_bits_rows pads each ROW to whole bytes and the pad sits at the end of the row,
        so if k*bits is a multiple of 8 then B*per == k exactly and slicing to k drops only padding.
        The shifts DESCEND (per-1)*bits .. 0 because pack_bits_rows writes the bits MSB-first.
      - other widths (3): the bit planes are separated one at a time and combined with a Horner
        step, so no [rows, k*bits] tensor is built at all. The planes are uint8 and each is
        [rows, k]; the accumulation is int32. The old version built [rows, k*bits] in int64,
        8 bytes per bit -- a [32768, 256] int64 tensor is 67 MB built and thrown away per
        container per layer. That was 1.122 ms, the single most expensive stage of the layer.
    """
    rows, B = packed_u8.shape
    dev = packed_u8.device
    if 8 % bits == 0:
        per = 8 // bits
        sh = (torch.arange(per - 1, -1, -1, device=dev) * bits).to(packed_u8.dtype)
        v = (packed_u8.reshape(rows, B, 1) >> sh) & ((1 << bits) - 1)
        return v.reshape(rows, B * per)[:, :k].to(torch.int32)
    shifts = torch.arange(7, -1, -1, device=dev).to(torch.uint8)
    planes = ((packed_u8.unsqueeze(-1) >> shifts) & 1).reshape(rows, B * 8)[:, :k * bits]
    out = torch.zeros(rows, k, dtype=torch.int32, device=dev)
    for i in range(bits):            # plane 0 is the MSB plane: Horner from the top
        out = out * 2 + planes[:, i::bits].to(torch.int32)
    return out



# A byte's eight bits are a pure function of that byte. Two ways to get them without a pass per
# operation; both are measured in lut_bitmap_check.py and the cheaper one is used.
_BIT_LUT = {}
_BIT_W = {}


def _bit_w(device):
    """[8] uint8 bit weights, MSB-first: 128, 64, ... 1 -- the mask each output bit is tested
    against. Cached: rebuilding it per call would be two extra launches x 64 calls per token."""
    t = _BIT_W.get(device)
    if t is None:
        t = (1 << torch.arange(7, -1, -1, device=device)).to(torch.uint8)
        _BIT_W[device] = t
    return t


def _bit_lut(device):
    """[256, 8] bool, row b = the bits of b, MSB-first.

    This is the LOSING variant, kept because it is the comparison arm in lut_bitmap_check.py. It
    needs the bitmap widened to int32 first -- `__getitem__` reads a uint8 index tensor as a MASK,
    not as indices -- and that widening costs as much as the mask it enables: 0.0838 ms against
    0.0542 for the bit-weight form, at [32768, 128].
    """
    t = _BIT_LUT.get(device)
    if t is not None:
        return t
    idx = torch.arange(256, dtype=torch.int32, device=device).unsqueeze(1)   # [256, 1]
    sh = torch.arange(7, -1, -1, dtype=torch.int32, device=device)           # [8]
    t = ((idx >> sh) & 1).to(torch.bool)                                     # [256, 8]
    _BIT_LUT[device] = t
    return t


def unpack_bitmap_rows(bitmap_u8, d):
    """uint8 [rows, d//8] -> bool [rows, d], device-only.

    It was unsqueeze + `>>` + `& 1` + `.to(torch.bool)`: four ops, three of which materialise a
    [rows, d] intermediate -- three write passes over 4.19 MB, per container, per layer, per token,
    to produce an output that is a byte-per-bit copy of its input.

    It is two ops now, and nothing [rows, d]-shaped is built before the output itself: mask each
    byte with the bit weights and compare. `!= 0` *is* a bool producer, so the cast that used to be
    a whole pass disappears, and testing `& w` against zero does the job of the old `>> w` + `& 1`
    in one op rather than two.

    Measured at [32768, 128] (one residual or V mask, one layer), 200 iters, CUDA events:
        old   0.3691 ms   unsqueeze >> shifts & 1 -> to(bool)
        lut   0.0838 ms   int32 index into a [256, 8] bool table
        ne    0.0542 ms   (x & bit_weights) != 0                    <- this one, 6.81x
    Two masks x 32 layers is 64 calls per token, so the old form's share was ~23.6 ms/token. The
    isolated bench is a direction check only -- isolated benches have repeatedly failed to predict
    this loop -- so the loop number comes from `bench.py measure_container_loop`.

    numpy's unpackbits is not an option: it would round-trip a 4 MB bitmap through host memory on
    every layer of every decode step.
    """
    rows, B = bitmap_u8.shape
    if bitmap_u8.dtype == torch.uint8:
        bits = bitmap_u8.unsqueeze(-1) & _bit_w(bitmap_u8.device)   # [rows, B, 8] uint8
        return (bits.reshape(rows, B * 8)[:, :d]) != 0
    shifts = torch.arange(7, -1, -1, device=bitmap_u8.device)
    b = ((bitmap_u8.unsqueeze(-1) >> shifts) & 1).to(torch.bool).reshape(rows, B * 8)
    return b[:, :d]


def _quant_rows(M, qbits):
    """Per-row symmetric quantisation -> (int8 [rows, d], fp16 scale [rows]). One ruler per row:
    a single global scale rounds small rows to zero (measured: +225 to +271 PPL)."""
    rmax = M.abs().amax(1, keepdim=True)
    sc = torch.where(rmax == 0, torch.ones_like(rmax), rmax) / (2 ** (qbits - 1) - 1)
    q = torch.clamp(torch.round(M / sc), -(2 ** (qbits - 1)), 2 ** (qbits - 1) - 1).to(torch.int8)
    return q, sc.to(torch.float16).reshape(-1)


def _topk_mask(M, k):
    """Exactly k True per row, by magnitude. The mask every container is built from."""
    m = torch.zeros_like(M, dtype=torch.bool)
    m.scatter_(1, torch.topk(M.abs(), k, dim=1).indices, True)
    return m


# ------------------------------------------------------------------ the container
# When container attention is installed it reads the store directly, so `update`'s dense rebuild is
# dead work. Set SKIP_MATERIALISE["on"] = True and `update` returns a 1-row placeholder instead of
# materialising K and V. Nothing else may be relied on while this is on.
SKIP_MATERIALISE = {"on": False}


class PackedRows:
    """A [rows, d] block as bitmap + k packed values + one fp16 scale per row."""

    def __init__(self, rows, d=D, k=32, bits=3):
        self.rows, self.d, self.k, self.bits = rows, d, k, bits
        self.bitmap = None      # uint8 [rows, d//8]
        self.packed = None      # uint8 [ceil(k*bits/8)]
        self.scale = None       # fp16  [rows]

    @staticmethod
    def build(X, mask, bits, d=D, k=None, check=True):
        """X: [rows, d] float. mask: [rows, d] bool with the SAME number of True per row.

        `k` MUST be supplied on the hot path. deriving it reads the tensor with `.item()`, i.e. a
        device SYNC, and this runs twice per layer on every decode step. A sync does not cost the
        microseconds it looks like: it drains the whole queued pipeline, so 64 of them per token
        is hundreds of milliseconds, not microseconds. `check` is the raggedness assertion, which
        also needs a sync and is therefore off on the hot path too.
        """
        rows = X.shape[0]
        if k is None:
            kept = mask.sum(1)
            k = int(kept.max().item())
            assert int(kept.min().item()) == k and k > 0, \
                f"mask is ragged: {int(kept.min().item())}..{k} lanes per row"
        elif check:
            assert int(mask.sum(1)[0].item()) == k, "supplied k disagrees with the mask"
        self = PackedRows(rows, d, k, bits)
        vals = torch.where(mask, X, torch.zeros_like(X))
        rmax = vals.abs().amax(1, keepdim=True)
        sc = torch.where(rmax == 0, torch.ones_like(rmax), rmax) / (2 ** (bits - 1) - 1)
        q = torch.clamp(torch.round(vals / sc), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        q = torch.where(mask, q, torch.zeros_like(q)) + (2 ** (bits - 1))     # offset, unsigned
        # row-major order of the True entries matches row-major order of nonzero(mask)
        q_rows = q[mask].reshape(rows, k).to(torch.int64)
        self.bitmap = PackedRows._pack_mask(mask)
        self.packed = pack_bits_rows(q_rows, bits)
        self.scale = sc.to(torch.float16).reshape(rows)
        return self

    @staticmethod
    def _pack_mask(mask):
        """bool [rows, d] -> uint8 [rows, d//8], device-only. The bitmap is a PER-ROW object, so
        packing it per row makes it byte-aligned by construction -- which is what lets an append be
        a `cat`. Doing this with numpy cost a D2H/H2D round trip per layer on every decode step."""
        rows, d = mask.shape
        b = mask.to(torch.uint8).reshape(rows, d // 8, 8)
        w = (1 << torch.arange(7, -1, -1, device=mask.device)).to(torch.uint8)
        return (b * w).sum(-1).to(torch.uint8)

    def nbytes(self):
        return int(self.bitmap.numel()) + int(self.packed.numel()) + \
               int(self.scale.numel()) * self.scale.element_size()

    def materialise(self, dtype=torch.float16, out=None):
        """Dense [rows, d] in `dtype` -- the tensor a fused kernel would never build.

        Entirely device-side. Every op here used to be a host round trip (`np.unpackbits` on the
        bitmap and again on the values, then `torch.nonzero`), which cost 1.6 s/token at n=4096
        against dense's 81 ms. There is no `nonzero` now either -- boolean indexing computes the set
        positions itself, so neither a search nor an index tensor is materialised.

        THE CACHED ARANGE IS GONE, AND IT WAS WORTH 8.00 GiB AT n=16384. This used to keep
        `self._ari = arange(rows*d)` on the store and assign `o[self._ari[mask]] = vals`. But
        `arange` indexed by a mask IS that mask's nonzero, so the tensor was pure overhead -- and
        because `materialise` runs once per layer, every store retained it for the cache's lifetime:
        8 heads x 16384 tokens x 128 dim x 8 B = 128 MiB per store, two stores per layer (K residual
        and V rows), 32 layers = **8.00 GiB**. It showed up as 8.011 GiB of resident memory that
        belonged to no container tensor, i.e. 4.2x the ENTIRE dense KV cache it was meant to beat.
        Booleans cost nothing here, so the fix is a deletion.
        """
        dev = self.bitmap.device
        mask = unpack_bitmap_rows(self.bitmap, self.d)                      # [rows, d] bool
        vals = unpack_bits_rows(self.packed, self.k, self.bits)             # [rows, k] int64
        vals = (vals - (2 ** (self.bits - 1))).to(torch.float32)
        vals = vals * self.scale.to(torch.float32).unsqueeze(1)
        o = torch.zeros(self.rows * self.d, dtype=torch.float32, device=dev)
        o[mask.reshape(-1)] = vals.reshape(-1)
        return o.reshape(self.rows, self.d).to(dtype)


# ------------------------------------------------------------------ K and V layers
class PackedKLayer:
    """Token coefficients + one shared decoder + a per-row residual container.

    Both the coefficients and the decoder are stored at `qbits`, because the campaign LEDGER bills
    them that way. Storing the decoder at fp16 instead silently over-billed every ratio: at r1=64
    a fp16 [r1, H*d] decoder is 32 B/row against the ledger's 24 B/row for the ENTIRETY of the
    core set, so the measured ratio would have read low and the codec would have taken the blame.
    """

    def __init__(self, coeff_q, coeff_scale, dec_q, dec_scale, resid, r1):
        self.coeff_q, self.coeff_scale = coeff_q, coeff_scale   # [n, r1] int8, [n] fp16
        self.dec_q, self.dec_scale = dec_q, dec_scale           # [r1, H*d] int8, [r1] fp16
        self.resid = resid                                      # PackedRows or None
        self.r1 = r1

    def decoder_f(self):
        """The decoder as fp32. Used by materialise AND by the append solve, so the store and the
        projection can never drift apart."""
        return self.dec_q.to(torch.float32) * self.dec_scale.to(torch.float32).unsqueeze(1)

    def nbytes(self):
        n = int(self.coeff_q.numel()) + int(self.coeff_scale.numel()) * 2 \
            + int(self.dec_q.numel()) + int(self.dec_scale.numel()) * 2
        return n + (self.resid.nbytes() if self.resid is not None else 0)

    def materialise(self, H, dtype=torch.float16):
        """-> [H, n, d]. The whole point: n rows rebuilt from n*r1 coefficients, one matmul."""
        n = self.coeff_q.shape[0]
        g = self.coeff_q.to(torch.float32) * self.coeff_scale.to(torch.float32).unsqueeze(1)
        kflat = g @ self.decoder_f()                            # [n, H*d], m = h*D + j
        # `d` COMES FROM THE DATA, NOT FROM THE MODULE CONSTANT. `K = Wk x` carries the model's own
        # head dim, and hardcoding 128 here reshaped a d=64 store to [H, n, 128]. That surfaced as a
        # shape error only because a reshape was involved; the same assumption anywhere without one
        # is a silent mis-reading instead of a crash.
        d = kflat.shape[1] // H
        if self.resid is not None:
            R = self.resid.materialise(torch.float32)           # [H*n, d]
            kflat = kflat + R.reshape(H, n, d).permute(1, 0, 2).reshape(n, H * d)
        return kflat.reshape(n, H, d).permute(1, 0, 2).contiguous().to(dtype)


class PackedVLayer:
    def __init__(self, rows):
        self.rows = rows

    def nbytes(self):
        return self.rows.nbytes()

    def materialise(self, H, dtype=torch.float16):
        """-> [H, n, d].

        The store's row order is (head, token) -- `V.reshape(H*n, d)` flattens as h*n + i. Reading
        it back needs `reshape(H, n, d)` and NO permute. Reshaping to (n, H, d) instead interprets
        the same buffer as i*H + h, which silently hands every head a different token's value: the
        logits then come back anti-correlated with dense (measured cosine -0.79) rather than merely
        noisy, because the error is a structured permutation.
        """
        n = self.rows.rows // H
        return self.rows.materialise(dtype).reshape(H, n, self.rows.d).contiguous()


# ------------------------------------------------------------------ building from dense K/V
def build_k_layer(K, tol, qbits=8, res_frac=0.25, res_bits=3, max_rank=None, d=D):
    """K: [H, n, d] float -> PackedKLayer. Coefficients come from TT-SVD with TOKENS FIRST, so the
    first unfolding is n x (d*H) and its rank counts key archetypes shared across heads.

    max_rank defaults to None (uncapped) to match the canonical implementation. Measured on real K
    at n=4096: canonical r1 is 93..616 (median 279) for `campaign4x` and 81..154 (median 112) for
    `tight`. A cap of 64 is binding on every layer of both configs, so it does not trim a tail -- it
    replaces the format."""
    from ttkv_codec import tt_svd, quant_cores
    H, n, dd = K.shape
    Xt = K.permute(1, 2, 0).contiguous()                        # [n, d, H]
    cores, ranks = tt_svd(Xt, tol, max_rank)
    n_el = sum(int(c.numel()) for c in cores)
    G1, G2, G3 = cores                                          # [1,n,r1] [r1,d,r2] [r2,H,1]
    r1 = int(G1.shape[2])
    # per-row quantisation of the token coefficients (the only large K object)
    c = G1.reshape(n, r1)
    cscale = c.abs().amax(1, keepdim=True) / (2 ** (qbits - 1) - 1)
    cscale = torch.where(cscale == 0, torch.ones_like(cscale), cscale)
    cq = torch.clamp(torch.round(c / cscale), -(2 ** (qbits - 1)), 2 ** (qbits - 1) - 1).to(torch.int8)
    # quantise the small cores and freeze the decoder in fp16 (the ledger bills it at qbits; at
    # r1*H*d = 23*1024 elements it is ~47 KB, i.e. 1.4e-3 B/row, so the honest thing is to say so
    # rather than pretend it is free)
    qc = quant_cores(cores[1:], qbits)
    dec = torch.einsum("ajb,bh->ahj", qc[0], qc[1][:, :, 0]).reshape(r1, H * dd).contiguous()
    dec_q, dec_scale = _quant_rows(dec, qbits)                          # stored at qbits, not fp16
    dec_f = dec_q.to(torch.float32) * dec_scale.to(torch.float32).unsqueeze(1)
    kflat = (cq.to(torch.float32) * cscale.float()) @ dec_f             # [n, H*d], m = h*dd + j
    resid = None
    if res_frac > 0:
        # The reference must be in the SAME layout the decoder produces, or the "residual" would be
        # measuring a misalignment between two orderings rather than the TT error.
        Kd = kflat.reshape(n, H, dd).permute(1, 0, 2).contiguous()      # [H, n, dd]
        R = (K.reshape(H, n, dd).float() - Kd).reshape(H * n, dd)
        kr = max(int(dd * res_frac), 1)
        mask = torch.zeros_like(R, dtype=torch.bool)
        mask.scatter_(1, torch.topk(R.abs(), kr, dim=1).indices, True)
        resid = PackedRows.build(R, mask, res_bits, dd, k=kr, check=False)
    return PackedKLayer(cq, cscale.to(torch.float16).reshape(n), dec_q, dec_scale, resid, r1)


def build_v_layer(V, k_lanes=45, bits=3, d=D):
    """V: [H, n, d] float -> PackedVLayer. Per-token magnitude pruning: no shared basis, because V
    is high-rank per token and a shared basis destroys exactly the per-token distinctness."""
    H, n, dd = V.shape
    M = V.reshape(H * n, dd).float()
    mask = torch.zeros_like(M, dtype=torch.bool)
    mask.scatter_(1, torch.topk(M.abs(), k_lanes, dim=1).indices, True)
    return PackedVLayer(PackedRows.build(M, mask, bits, dd, k=k_lanes, check=False))


# ------------------------------------------------------------------ batched pools
# B per-sequence stores in one object, pooled into CONTIGUOUS SLABS so that
#     row = bz*(H*n) + h*n + i
# Pooling is then a plain cat along dim 0 -- no row permutation, no gather -- and a kernel block can
# only ever read one sequence's rows.
#
# THE DECODER IS PER SEQUENCE, AND THAT IS THE POINT. The first design shared one fitted basis across
# the batch. It is simpler -- the rows become a trivial relabelling of tokens -- but it CHANGES THE
# ANSWER: a rank-r1 basis that has to span B sequences must cover B times the key diversity, so every
# sequence is approximated more coarsely than it would be alone. Measured directly, a batch slice then
# disagreed with that sequence's own single-sequence run by 5.8 in absolute score. Batching is
# supposed to be a serving change, not an accuracy change, so each sequence keeps its own decoder and
# they are stacked here with zero padding to a common rank (ranks differ per sequence by
# construction -- 227 and 228 were both observed -- and padding BOTH the decoder rows and the
# coefficient columns keeps the product exact, since a zero times a zero is zero either way).
def _pad_rank(t, r1, R, dim, value=0.0):
    """Zero-pad along `dim` from r1 to R. Padding both sides of a product keeps it exact."""
    if R == r1:
        return t
    shape = list(t.shape)
    shape[dim] = R - r1
    pad = torch.full(shape, value, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=dim)


def _concat_rows_many(row_list):
    out = row_list[0]
    for r in row_list[1:]:
        out = _concat_rows(out, r)
    return out


class PooledKStore:
    """M per-sequence K stores pooled into slabs, with stacked decoders."""

    def __init__(self, kls, H, n):
        B = len(kls)
        R = max(k.r1 for k in kls)
        dev = kls[0].coeff_q.device
        self.H, self.n, self.B, self.r1 = H, n, B, R
        # Width taken from the per-sequence decoders, not from the module constant.
        self.dec_q = torch.zeros(B, R, kls[0].dec_q.shape[1], dtype=torch.int8, device=dev)
        self.dec_scale = torch.zeros(B, R, dtype=torch.float16, device=dev)
        cq, csc = [], []
        for b, k in enumerate(kls):
            self.dec_q[b, :k.r1] = k.dec_q
            self.dec_scale[b, :k.r1] = k.dec_scale
            cq.append(_pad_rank(k.coeff_q, k.r1, R, 1))
            # coeff_scale is [n] -- ONE VALUE PER TOKEN, with no rank axis to pad. Padding it with
            # dim=0 pads the TOKEN axis instead, silently giving a sequence n + (R - r1) rows. That
            # is exactly what happened: with ranks [228, 228, 227, 228] the short sequence produced
            # 1025 rows, and materialise died on `4096 vs 4100`. It needs no padding at all -- the
            # padded rank columns of coeff_q are zero, and zero times any scale is zero.
            csc.append(k.coeff_scale)
        self.coeff_q = torch.cat(cq, 0)                 # [B*n, R], row = b*n + i
        self.coeff_scale = torch.cat(csc, 0)            # [B*n]
        self.resid = _concat_rows_many([k.resid for k in kls if k.resid is not None]) \
            if any(k.resid is not None for k in kls) else None

    def decoder_f(self):
        """[B, R, H*d] fp32. One basis per sequence -- see the note above the class."""
        return self.dec_q.to(torch.float32) * self.dec_scale.to(torch.float32).unsqueeze(-1)

    def nbytes(self):
        n = int(self.coeff_q.numel()) + int(self.coeff_scale.numel()) * 2 \
            + int(self.dec_q.numel()) + int(self.dec_scale.numel()) * 2
        return n + (self.resid.nbytes() if self.resid is not None else 0)

    def materialise(self, H, dtype=torch.float16):
        """-> [B, H, n, d]. Prefill-only, so a loop over the batch is fine; the decode path never
        calls this (SKIP_MATERIALISE is on and the kernels read the store directly)."""
        dec = self.decoder_f()                                          # [B, R, H*d]
        cq = (self.coeff_q.float() * self.coeff_scale.float().unsqueeze(1)
              ).reshape(self.B, self.n, self.r1)                        # [B, n, R]
        K = torch.stack([cq[b] @ dec[b] for b in range(self.B)])        # [B, n, H*d]
        d = dec.shape[-1] // H                                          # from the data, not `D`
        K = K.reshape(self.B, self.n, H, d).permute(0, 2, 1, 3)         # [B, H, n, d]
        if self.resid is not None:
            Rm = self.resid.materialise(torch.float32)                  # [B*H*n, d], slab order
            K = K + Rm.reshape(self.B, H, self.n, d)
        return K.to(dtype)


class PooledVStore:
    """V is per-row, so B of them is just a longer row list in the same slab order."""

    def __init__(self, vls, H, n):
        self.H, self.n, self.B = H, n, len(vls)
        self.rows = _concat_rows_many([v.rows for v in vls])

    def nbytes(self):
        return self.rows.nbytes()

    def materialise(self, H, dtype=torch.float16):
        return self.rows.materialise(dtype).reshape(self.B, H, self.n, self.rows.d).to(dtype)


# ------------------------------------------------------------------ the cache, itself
def _rotate_half(x):
    """HF's convention: dim j pairs with dim j + d/2, NOT (2j, 2j+1).

    Getting this wrong still produces an ORTHOGONAL, position-dependent transform, so nothing
    raises, every norm and energy check still passes, and the stored tensor is silently
    mis-rotated. See ROPE_DISPERSION.md."""
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def rope_apply(x, cos, sin):
    """Forward RoPE for the (j, j+d/2) pair layout. cos/sin broadcast against x."""
    return x * cos + _rotate_half(x) * sin


def unrope_apply(y, cos, sin):
    """Inverse RoPE -- rotation by -theta. Derived, not guessed: with a = x[j], b = x[j+d/2], the
    forward map is (a*cos - b*sin, b*cos + a*sin), so the inverse is `y*cos - rotate_half(y)*sin`.
    Verified `unrope(rope(x)) == x` at 6.3e-8."""
    return y * cos - _rotate_half(y) * sin


# ---- the re-rotation, and why it is a BYTE problem and not a maths one
#
# With storage="pre" the store holds position-free keys, so every materialise has to rotate the
# WHOLE key block back. Measured on the real shape (n=8192, H=8, d=128, 32 layers):
#
#     eager fp32 angles + final cast   32.61 ms/step   <- the shipped form
#     bf16 angles + chunk slicing      14.73 ms/step   <- this path
#     floor: read K + write rotated K   2.90 ms/step   <- a bare clone, the target
#
# and the port's DECODE penalty measured in the real loop was +29.51 ms/step, i.e. this function is
# the entire cost. Two things make it 2.2x heavier than it needs to be:
#
#   (a) `x * cos` with fp32 cos PROMOTES a bf16 K to fp32, so every intermediate is twice as wide,
#       and the result then has to be cast back. Keeping the angles in the output dtype removes both.
#   (b) `_rotate_half`'s `cat([-x[..., h:], x[..., :h]])` materialises a full-width copy -- it alone
#       measured 5.99 ms/step, twice a bare `clone()` over the same bytes. Slicing with views and
#       building the two halves directly removes that copy.
ROPE_FAST = {"on": False}


def _re_rotate(x, cos, sin):
    """RoPE with both byte fixes and no change to the algebra.

    `a*ca - b*sa` is `x1*cos1 - x2*sin1` and `b*cb + a*sb` is `x2*cos2 + x1*sin2`, which is exactly
    `x*cos + rotate_half(x)*sin` written against the (j, j+d/2) pairing. The ONLY numerical change
    is the angle precision: a bf16 angle carries ~0.4 % absolute error near 1.0, which measured
    0.99999452 cosine on the rotated tensor against the fp32 form. The rotation itself is exact.

    `cos`/`sin` must already be in `x`'s dtype -- see `TTKVLayer._cos_sin_dt`.
    """
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    ca, cb = cos[..., :h], cos[..., h:]
    sa, sb = sin[..., :h], sin[..., h:]
    return torch.cat([a * ca - b * sa, b * cb + a * sb], dim=-1)


def make_cache(cfg, H=8, NL=32, fold_T=256, storage="post", rope_fn=None):
    """Build a transformers Cache whose per-layer objects hold the PACKED store plus a DENSE tail.

    transformers 5.14 made DynamicCache a container of per-layer caches:
        keys, values = self.layers[layer_idx].update(key_states, value_states, *args, **kwargs)
    so a per-layer class is the whole integration. `update` receives ONLY the new rows, compresses
    them into the store, and returns the materialised full K/V for attention. Nothing dense survives
    the call, which is what makes the capacity claim real rather than arithmetical.

    `fold_T` is the repack interval. Appending straight into the container costs ~20 ms per layer
    per token -- it re-quantises, re-packs and re-cats the whole container for one row -- and that
    was 86% of the decode step. New rows are therefore held dense and folded only every `fold_T`
    tokens, so the per-token cost of maintaining the container drops by ~fold_T. The tail is at
    most fold_T rows dense, which at fold_T=256 is ~1 MB per layer against a packed store that is
    orders of magnitude larger, so it does not touch the capacity claim.

    `storage` selects WHERE the key compression happens:

        "post"  compress the rows `update` was handed, i.e. post-RoPE. The shipped behaviour.
        "pre"   un-rotate to pre-RoPE, compress, re-rotate on the way out. Measured **4.70x ->
                6.52x** joint ratio at unchanged logit alignment, for +2.4% prefill time
                (`probe_prerope_container.py`). RoPE disperses the key block's spectrum, so the
                pre-RoPE object needs ~3.6x less rank for the same error budget -- see
                ROPE_DISPERSION.md and PAPER_CLAIMS C16/C17.

    `rope_fn` supplies the TRUE frequencies and is REQUIRED for storage="pre". The model's inv_freq
    carries a llama3 ramp that `config.rope_theta` does not reproduce, and -- again -- a wrong
    frequency set still produces an orthogonal, position-dependent transform, so a mistake here
    fails silently rather than loudly.

    WARNING, and it is not a formality: with storage="pre" the STORE holds position-free keys, and
    `attend_container` contracts the query directly against that store. A query is post-RoPE, so
    that contraction is `q_post . k_pre` -- wrong, finite, correctly shaped and undetectable further
    downstream. `attend_container` therefore REFUSES a pre-RoPE store rather than reading it.
    """
    assert storage in ("post", "pre"), f"storage must be 'post' or 'pre', got {storage!r}"
    if storage == "pre":
        assert rope_fn is not None, (
            "storage='pre' requires rope_fn: (positions) -> (cos, sin). The true frequencies cannot "
            "be reconstructed from cfg, and a wrong set is silently orthogonal rather than wrong.")
    from transformers.cache_utils import DynamicLayer as _Base

    # ONE ANGLE TABLE PER CACHE, NOT PER LAYER. Every layer called `rope_fn` for itself and cached
    # its own [n, TD] fp32 copy, so at n=32768 the cache held 32 identical copies: measured 1.003
    # GiB of angles against a 0.536 GiB container, 1.87x the store it exists to serve, and 32
    # distinct allocations. `_cos_sin` only ever READS what it returns, so pointing every layer at
    # one tensor changes no arithmetic -- it changes only how many copies exist. Measured reclaim at
    # 32k: allocated 6.974 -> 6.034 GiB. The holder lives here because every layer of one cache
    # shares one `rope_fn` and one device.
    _CS = {"cs": None, "cs16": None}

    if isinstance(cfg["tols"], (list, tuple)):
        assert len(cfg["tols"]) == NL, \
            f"config gives {len(cfg['tols'])} tolerances but the model has {NL} layers"
        TOLS = [float(t) for t in cfg["tols"]]
    else:
        TOLS = [float(cfg["tols"])] * NL

    class TTKVLayer(_Base):
        """One layer of the compressed cache. `update` is the whole integration:

            keys, values = self.layers[layer_idx].update(key_states, value_states, ...)

        so we compress the rows the model just handed us, drop them, and hand back the materialised
        full K/V. What persists is the packed store, never the dense tensor.
        """

        def __init__(self, li=0):
            super().__init__()
            self.cfg = cfg                      # from the enclosing make_cache scope
            self.H = H
            self.li = li
            self.tol = TOLS[li]
            self.kstore = None
            self.vstore = None
            self.tail_k = None                  # dense rows not yet folded into the store
            self.tail_v = None
            self.folds = 0
            self.B = 1
            self.fold_T = max(int(fold_T), 1)
            self.storage = storage
            self._cs = None                     # cached fp32 (cos, sin), grown on demand: a
            #                                     decode step must not recompute angles per layer
            self._cs16 = None                   # the same angles in the OUTPUT dtype, for the
            #                                     re-rotation only (see `_cos_sin_dt`)

        def _cos_sin(self, n, device):
            """(cos, sin) fp32 for positions 0..n-1, cached, on `device`. Positions are the running
            length, not a `cache_position` kwarg: the container is append-only and single-sequence,
            so the running length IS the position, and depending on a kwarg would fail precisely on
            the code path that does not pass it.

            fp32 on purpose: this is what the UN-rotate uses, and it has to invert the rotation the
            model actually applied, not an approximation of it."""
            if _CS["cs"] is None or _CS["cs"][0].shape[0] < n:
                cos, sin = rope_fn(torch.arange(n, device=device))
                _CS["cs"] = (cos.to(device), sin.to(device))
                _CS["cs16"] = None              # the dtype copy is now stale
            self._cs = _CS["cs"]
            return self._cs

        def _cos_sin_dt(self, n, device, dtype):
            """The same angles in `dtype`, for the RE-rotation.

            Kept separate from `_cos_sin` so the un-rotate keeps fp32 precision while the re-rotate
            stops promoting every intermediate to fp32. Measured worth 32.61 -> 18.53 ms/step on
            the real shape, before the chunk-slicing change on top.
            """
            self._cos_sin(n, device)
            if (_CS["cs16"] is None or _CS["cs16"][0].shape[0] < n
                    or _CS["cs16"][0].dtype != dtype):
                _CS["cs16"] = (_CS["cs"][0].to(dtype), _CS["cs"][1].to(dtype))
            self._cs16 = _CS["cs16"]
            return _CS["cs16"]

        def update(self, key_states, value_states, *args, **kwargs):
            # BATCH MUST BE 1 for the APPEND path, but the build path handles B>1. Appending a
            # token to a pooled container lands at row bz*(H*cap) + h*cap + n, which is INTERIOR to
            # the pooled tensor -- the single-sequence append works only because a new row goes at
            # the very end. "Interior" is not a plain cat, so it needs capacity slabs and a per-
            # sequence write cursor. Until then: run with fold_T beyond the decode window, so new
            # tokens stay in the dense tail (which attention already handles and which is batched).
            assert key_states.shape[1] == self.H and value_states.shape[1] == self.H, (
                f"expected [b, {self.H}, q, d]; got {tuple(key_states.shape)} / "
                f"{tuple(value_states.shape)}")
            # Materialise in the dtype the model handed us. This model runs attention in bf16 while
            # its 4-bit compute dtype is fp16, so hardcoding fp16 raised
            #     Expected query, key, and value to have the same dtype
            dt = key_states.dtype
            k = key_states.float()                       # [B, H, q, d]
            v = value_states.float()
            q_len = k.shape[2]
            if storage == "pre" and q_len > 0:
                # Un-rotate FIRST, so what the store holds is position-free. Everything below
                # (`_build`, `_push_tail`, `_append`) then operates on pre-RoPE rows untouched.
                start = self.get_seq_length()
                cos, sin = self._cos_sin(start + q_len, k.device)
                k = unrope_apply(k, cos[start:start + q_len], sin[start:start + q_len])
            if self.kstore is None:
                self._build(k, v, dt)                    # the first chunk (prefill) becomes the store
            else:
                self._push_tail(k, v)
            if SKIP_MATERIALISE["on"] and q_len == 1:
                # Container attention reads the store directly, so the dense rebuild is dead work --
                # but ONLY for a single-token step, because `_patched_sdpa` routes to
                # `attend_container` only when q_len == 1. Any other q_len (a chunked prefill, or a
                # speculative/multi-token decode step) falls through to stock SDPA, which would then
                # attend to a ONE-TOKEN K/V: no error and no shape mismatch, just wrong logits. The
                # placeholder is therefore confined to the case it is correct for, and every other
                # step gets the real materialised block.
                # This still assumes the container attention is INSTALLED. With the flag on and no
                # install, a q_len == 1 step is wrong the same way -- the flag belongs with the
                # install, it is not an independent switch.
                return key_states[:, :, :1, :], value_states[:, :, :1, :]
            Kd = self.kstore.materialise(self.H, dt)
            Vd = self.vstore.materialise(self.H, dt)
            if Kd.dim() == 3:            # a single-sequence store returns [H, n, d]
                Kd, Vd = Kd.unsqueeze(0), Vd.unsqueeze(0)
            if self.tail_k is not None:
                Kd = torch.cat([Kd, self.tail_k.to(dt)], dim=2)
                Vd = torch.cat([Vd, self.tail_v.to(dt)], dim=2)
            if storage == "pre":
                # Re-rotate the WHOLE materialised block in one call. Store rows occupy positions
                # 0..n_store-1 and tail rows follow contiguously, which is exactly what makes a
                # single rotation over the concatenation correct -- and it holds only because the
                # container is append-only and single-sequence.
                if ROPE_FAST["on"]:
                    cos, sin = self._cos_sin_dt(Kd.shape[2], Kd.device, dt)
                    Kd = _re_rotate(Kd, cos, sin)
                else:
                    # `.to(dt)` is NOT cosmetic: cos/sin are float32, so the product promotes K to
                    # float32 and stock SDPA then rejects it with
                    #     Expected query, key, and value to have the same dtype ... key.dtype: float
                    cos, sin = self._cos_sin(Kd.shape[2], Kd.device)
                    Kd = rope_apply(Kd, cos.unsqueeze(0).unsqueeze(0),
                                    sin.unsqueeze(0).unsqueeze(0)).to(dt)
            return Kd, Vd

        def _push_tail(self, k, v):
            """Hold new rows DENSE until the tail is worth folding.

            Appending straight into the container costs ~20 ms per layer per token -- it
            re-quantises, re-packs and re-cats the whole container for one row -- and that measured
            at 86% of the decode step. The tail is already dense, so attention needs no repack for
            it at all; the container is touched once per `fold_T` tokens instead of once per token.

            k/v arrive as [B, H, q, d]; the cat is along the token axis, which is dim 2 for every B.
            """
            self.tail_k = k if self.tail_k is None else torch.cat([self.tail_k, k], dim=2)
            self.tail_v = v if self.tail_v is None else torch.cat([self.tail_v, v], dim=2)
            if self.tail_k.shape[2] >= self.fold_T:
                if self.tail_k.shape[0] > 1:
                    raise NotImplementedError(
                        "folding a POOLED container needs capacity slabs: a new token for every "
                        "sequence lands at row bz*(H*cap) + h*cap + n, interior to the pooled tensor, "
                        "so it cannot be the plain cat the single-sequence append relies on. Run "
                        "with fold_T larger than the decode window (the dense tail carries the new "
                        "tokens and attention is batched), or implement slabs.")
                self._append(self.tail_k[0], self.tail_v[0])
                self.tail_k = None
                self.tail_v = None
                self.folds += 1

        def _build(self, k, v, dtype):
            """k/v: [B, H, q, d]. B=1 keeps the single-sequence store EXACTLY as it was, so a
            batching change can never alter the single-sequence result."""
            cfg = self.cfg
            B, q = k.shape[0], k.shape[2]
            d = k.shape[-1]        # the MODEL's head dim; hardcoding 128 built a d/8-byte bitmap for
            self.B = B             # a row that is not d wide, which fails loudly here and silently
            if B == 1:             # in any place that only slices
                self.kstore = build_k_layer(k[0], self.tol, cfg["qbits"],
                                            cfg["res_frac"], cfg["res_bits"], d=d)
                self.vstore = build_v_layer(v[0], cfg["k_lanes"], cfg["vbits"], d=d)
            else:
                # One store per sequence, then pooled. The decoders are NOT shared -- see the note
                # above PooledKStore for why that is a correctness requirement and not a preference.
                self.kstore = PooledKStore(
                    [build_k_layer(k[b], self.tol, cfg["qbits"], cfg["res_frac"],
                                   cfg["res_bits"], d=d) for b in range(B)], self.H, q)
                self.vstore = PooledVStore(
                    [build_v_layer(v[b], cfg["k_lanes"], cfg["vbits"], d=d) for b in range(B)],
                    self.H, q)

        def _append(self, k, v):
            """Append q_len new rows. K: project onto the frozen decoder and store the coefficients
            -- one r1-sized solve per token, no refit, which is the streaming property. V: pack the
            new rows into the existing container."""
            dec = self.kstore.decoder_f()                            # [r1, H*d], m = h*d + j
            d = dec.shape[1] // self.H                              # from the store, not the module
            ql = k.shape[1]
            x = k.permute(1, 0, 2).reshape(ql, -1).to(torch.float32)     # [q, H*d], SAME layout
            r1 = dec.shape[0]
            A = dec @ dec.T + 1e-3 * torch.eye(r1, device=dec.device, dtype=torch.float32)
            g = torch.linalg.solve(A, dec @ x.T).T                       # [q, r1]
            cscale = g.abs().amax(1, keepdim=True) / (2 ** (self.cfg["qbits"] - 1) - 1)
            cscale = torch.where(cscale == 0, torch.ones_like(cscale), cscale)
            cq = torch.clamp(torch.round(g / cscale), -(2 ** (self.cfg["qbits"] - 1)),
                             2 ** (self.cfg["qbits"] - 1) - 1).to(torch.int8)
            self.kstore.coeff_q = torch.cat([self.kstore.coeff_q, cq], 0)
            self.kstore.coeff_scale = torch.cat([self.kstore.coeff_scale,
                                                 cscale.to(torch.float16).reshape(-1)], 0)
            if self.kstore.resid is not None:
                Kd_new = (cq.float() * cscale) @ dec                     # [q, H*d]
                R = (x - Kd_new).reshape(ql, self.H, d).permute(1, 0, 2).reshape(self.H * ql, d)
                kr = self.kstore.resid.k
                mask = torch.zeros_like(R, dtype=torch.bool)
                mask.scatter_(1, torch.topk(R.abs(), kr, dim=1).indices, True)
                self.kstore.resid = _insert_rows_headmajor(
                    self.kstore.resid,
                    PackedRows.build(R, mask, self.cfg["res_bits"], d, k=kr, check=False), self.H)
            mv = v.reshape(self.H * v.shape[1], d).float()
            mask = torch.zeros_like(mv, dtype=torch.bool)
            mask.scatter_(1, torch.topk(mv.abs(), self.cfg["k_lanes"], dim=1).indices, True)
            self.vstore.rows = _insert_rows_headmajor(
                self.vstore.rows,
                PackedRows.build(mv, mask, self.cfg["vbits"], d,
                                 k=self.cfg["k_lanes"], check=False), self.H)

        def get_seq_length(self, *a, **kw):
            """Store rows PLUS tail rows. Reporting only the store would desynchronise the model's
            cache_position and masks from the K/V it is actually attending to."""
            n = 0
            if self.kstore is not None:
                # PooledKStore holds B*n coefficient rows, so it carries its own per-sequence token
                # count; a single-sequence store's row count IS its token count.
                n = int(self.kstore.n) if hasattr(self.kstore, "n") else int(self.kstore.coeff_q.shape[0])
            elif self.tail_k is not None:
                n = 0
            if self.tail_k is not None:
                n += int(self.tail_k.shape[2])
            return n

        def get_max_length(self, *a, **kw):
            return None

        def nbytes(self):
            n = (self.kstore.nbytes() if self.kstore else 0) + \
                (self.vstore.nbytes() if self.vstore else 0)
            if self.tail_k is not None:
                n += (self.tail_k.numel() + self.tail_v.numel()) * 2
            return n

    # Cache, not DynamicCache: `layer_class_to_replicate` is a keyword of the BASE class.
    # DynamicCache.__init__ has its own signature and hardcodes DynamicLayer.
    from transformers.cache_utils import Cache

    class TTKVCache(Cache):
        """The cache the model is handed. Its `layers` are TTKVLayer objects, so what persists
        between steps is the packed store, not the dense tensors."""

        def __init__(self, **kw):
            super().__init__(layer_class_to_replicate=TTKVLayer, **kw)

        def update(self, key_states, value_states, layer_idx, *args, **kwargs):
            """Mirror the base implementation but create each layer WITH its index. Both shipped
            configs run a uniform tol of 0.20, so nothing today reads the index per layer; it is
            threaded through so that a scheduled arm can be built without touching this call path
            again (a schedule was tried and did NOT beat uniform -- TTKV_PRIMER ch. 9)."""
            while len(self.layers) <= layer_idx:
                self.layers.append(TTKVLayer(len(self.layers)))
            return self.layers[layer_idx].update(key_states, value_states, *args, **kwargs)

        def get_seq_length(self, layer_idx=0):
            return 0 if not self.layers else self.layers[layer_idx].get_seq_length()

        def get_max_length(self, *a, **kw):
            return None

        def packed_nbytes(self):
            return sum(l.nbytes() for l in self.layers if hasattr(l, "nbytes"))

    return TTKVCache


def _concat_rows(a, b):
    """Append rows: a plain cat on all three arrays. Correct ONLY because pack_bits_rows pads each
    row to whole bytes -- without that, the bit field would straddle rows and this would be
    silently wrong rather than slow.

    This appends `b` as one BLOCK, so it is correct only for a single head's row list (and for the
    pooled slabs, where each sequence's block genuinely is contiguous). For a multi-head container
    use `_insert_rows_headmajor`.
    """
    assert a.d == b.d and a.bits == b.bits and a.k == b.k, \
        f"containers differ: d {a.d}/{b.d}, bits {a.bits}/{b.bits}, k {a.k}/{b.k}"
    out = PackedRows(a.rows + b.rows, a.d, a.k, a.bits)
    out.bitmap = torch.cat([a.bitmap, b.bitmap], 0)
    out.packed = torch.cat([a.packed, b.packed], 0)
    out.scale = torch.cat([a.scale, b.scale], 0)
    return out


def _insert_rows_headmajor(a, b, H):
    """Insert `b`'s rows into `a` per head, preserving the (head, token) order every reader assumes.

    `_concat_rows` appends `b` as one BLOCK after `a`, which is right only for a single head's row
    list. Every multi-head container here is (head, token)-ordered with row = h*n + i -- both the
    build path (`build_k_layer` / `build_v_layer` reshape K/V's [H, n, d] to [H*n, d]) and a fold's
    new rows, which arrive head-major as h*q + i. So a block append lands the new rows at
    `H*n_old + h*q + i` instead of `h*(n_old+q) + i`. The row COUNT is unchanged and no shape check
    catches it, but every reader addresses h*n+i -- `PackedKLayer.materialise` and
    `PackedVLayer.materialise` reshape to [H, n, d], and `ttkv_cuda.resid_add` / `v_out` compute
    the same row offset -- so the store is silently permuted. The standing signal for this class of
    mistake is an ANTI-correlated metric, not noise. `self_test_fold` is the regression test.

    Both sides are viewed as [H, n, ...] and joined on the TOKEN axis, which reproduces a one-shot
    build row for row. That is legal because `pack_bits_rows` pads each row to a whole number of
    bytes, so a row is a contiguous object in all three arrays.
    """
    assert a.d == b.d and a.bits == b.bits and a.k == b.k, \
        f"containers differ: d {a.d}/{b.d}, bits {a.bits}/{b.bits}, k {a.k}/{b.k}"
    assert a.rows % H == 0 and b.rows % H == 0, \
        f"row counts {a.rows}/{b.rows} are not whole token blocks at H={H}"
    n_old, q_len = a.rows // H, b.rows // H
    out = PackedRows(a.rows + b.rows, a.d, a.k, a.bits)
    out.bitmap = torch.cat([a.bitmap.reshape(H, n_old, -1),
                            b.bitmap.reshape(H, q_len, -1)], dim=1).reshape(a.rows + b.rows, -1)
    out.packed = torch.cat([a.packed.reshape(H, n_old, -1),
                            b.packed.reshape(H, q_len, -1)], dim=1).reshape(a.rows + b.rows, -1)
    out.scale = torch.cat([a.scale.reshape(H, n_old),
                           b.scale.reshape(H, q_len)], dim=1).reshape(-1)
    return out


def _mask_of(p):
    """The stored mask, device-only."""
    return unpack_bitmap_rows(p.bitmap, p.d)


def self_test(H=8, n=512, d=D):
    """Container round-trip on real-shaped random data. Validates packing/unpacking and reports
    measured bytes against the hand ledger. NOT a ratio test: white noise is not compressible."""
    from ttkv_codec import CONFIGS
    torch.manual_seed(0)
    print(f"packed-cache self test: H={H} n={n} d={d}")
    K = torch.randn(H, n, d) * 0.2
    V = torch.randn(H, n, d) * 0.2
    for nm, cfg in CONFIGS.items():
        tol = cfg["tols"] if isinstance(cfg["tols"], float) else cfg["tols"][0]
        kl = build_k_layer(K, tol, cfg["qbits"], cfg["res_frac"], cfg["res_bits"])
        vl = build_v_layer(V, cfg["k_lanes"], cfg["vbits"])
        # round trip: compare against the SAME codec's dense reference
        Kd = kl.materialise(H)
        Vd = vl.materialise(H)
        rows = H * n
        kb, vb = kl.nbytes(), vl.nbytes()
        jx = (2 * d * 2) / (kb / rows + vb / rows)
        print(f"  {nm:11s} r1={kl.r1:3d} K={kb:9d} B  V={vb:9d} B  joint={jx:5.2f}x  "
              f"Kerr={float((Kd-K).abs().mean()):.4f} Verr={float((Vd-V).abs().mean()):.4f}")
    # packing must be lossless for the quantised values, at both config bit depths
    for bits, k in ((3, 45), (4, 64), (3, 32)):
        x = torch.randint(0, 2 ** bits, (37, k))
        rt = unpack_bits_rows(pack_bits_rows(x, bits), k, bits)
        assert bool((rt == x).all()), f"{bits}-bit pack/unpack is not lossless at k={k}"
    print("  bit packing round-trips exactly, and rows are byte-aligned (appends are a cat)")
    # an append must equal a rebuild -- this is the property streaming decode depends on. This
    # fixture is a SINGLE head's row list, which is the case `_concat_rows` is correct for; the
    # multi-head case is `self_test_fold`.
    A = PackedRows.build(torch.randn(50, d), _topk_mask(torch.randn(50, d), 32), 3, d)
    B = PackedRows.build(torch.randn(20, d), _topk_mask(torch.randn(20, d), 32), 3, d)
    C = _concat_rows(A, B)
    assert C.rows == 70 and int(C.packed.shape[0]) == 70 and int(C.bitmap.shape[0]) == 70, \
        "concat produced the wrong row count"
    assert bool(torch.equal(C.materialise()[50:], B.materialise())), \
        "appended rows do not materialise back to what was stored"
    print("  concat == rebuild for the appended rows")


def self_test_fold(d=D):
    """A fold must land its new rows in the SAME (head, token) order a one-shot build produces.

    `_append` adds one fold's worth of rows to an existing store. Those rows are built head-major
    (row h*q + i) while the store they join is addressed h*n + i, so the join has to insert PER
    HEAD. A block append is invisible to a shape or row-COUNT check -- it permutes (head, token) --
    so the check has to be on the VALUES.

    V is stateless per row (top-k and one ruler per row, no shared basis), which makes the
    invariant testable EXACTLY: build V in one shot, and separately build V[:n_old] and then fold
    V[n_old:], and the two containers must agree bit for bit. This is the fold form of the
    acceptance criterion "a sequence is bit-identical to its own single pass".

    The helper is checked directly, and then a whole `make_cache` fold is run end to end, because
    the helper being right does not prove `_append` calls it.
    """
    torch.manual_seed(0)
    H, n_old, q_len = 8, 5, 3
    n = n_old + q_len
    k_lanes, bits = 45, 3
    V = torch.randn(H, n, d) * 0.2

    one_shot = build_v_layer(V, k_lanes, bits, d)
    old = build_v_layer(V[:, :n_old], k_lanes, bits, d)
    new = build_v_layer(V[:, n_old:], k_lanes, bits, d)
    folded = _insert_rows_headmajor(old.rows, new.rows, H)

    shot_dense = one_shot.materialise(H)
    assert folded.rows == one_shot.rows.rows == H * n
    assert bool(torch.equal(folded.materialise().reshape(H, n, d), shot_dense)), (
        "the folded store does not match a one-shot build: the fold reordered (head, token)")

    # The check only has teeth if the block-append form really differs; otherwise it would pass
    # against the very bug it exists to catch.
    blocky = _concat_rows(old.rows, new.rows)
    assert not bool(torch.equal(blocky.materialise().reshape(H, n, d), shot_dense)), (
        "the fixture no longer distinguishes a block append from a per-head insert")

    # And the layout is (head, token) for a K-style row list too, checked without the codec: tag
    # each row h*100 + token and read the tag back out of the materialised row. bits=2 with a
    # constant row means the codec reproduces the tag exactly (q == 1, scale == tag).
    def tagged(n_tok, offset):
        t = torch.empty(H * n_tok, d)
        for h in range(H):
            for i in range(n_tok):
                t[h * n_tok + i] = float(h * 100 + offset + i)
        return t

    a = PackedRows.build(tagged(n_old, 0), torch.ones(H * n_old, d, dtype=torch.bool), 2, d,
                         k=d, check=False)
    b = PackedRows.build(tagged(q_len, n_old), torch.ones(H * q_len, d, dtype=torch.bool), 2, d,
                         k=d, check=False)
    got = _insert_rows_headmajor(a, b, H).materialise()
    for h in range(H):
        for i in range(n):
            assert float(got[h * n + i, 0]) == float(h * 100 + i), (
                f"row {h * n + i} holds tag {float(got[h * n + i, 0])}, expected {h * 100 + i}")

    # And through the REAL path. A cache that folds once must agree with a one-shot build of the
    # same sequence -- this is the check that catches the bug where it actually lived, in
    # `_append`, rather than in the helper it calls.
    try:
        from transformers.cache_utils import DynamicLayer  # noqa: F401
    except Exception:                                        # noqa: BLE001 (older transformers)
        print("  fold test (end to end): SKIPPED, this transformers has no DynamicLayer")
        return
    cfg = CONFIGS["campaign4x"]
    fold_T, n0 = 4, 5
    cache = make_cache(cfg, H=H, NL=1, fold_T=fold_T, storage="post")()
    V0 = torch.randn(1, H, n0, d) * 0.2
    K0 = torch.randn(1, H, n0, d) * 0.2
    cache.update(K0, V0, 0)
    vs = [V0]
    for _ in range(fold_T):
        v = torch.randn(1, H, 1, d) * 0.2
        cache.update(torch.randn(1, H, 1, d) * 0.2, v, 0)
        vs.append(v)
    lyr = cache.layers[0]
    assert lyr.folds == 1, f"expected exactly one fold, got {lyr.folds}"
    vfull = torch.cat(vs, 2)
    n = vfull.shape[2]
    assert lyr.vstore.rows.rows == H * n, (
        "the fold changed the row COUNT -- which is exactly why reordering (head, token) is "
        "invisible to a shape check and has to be caught on the values")
    assert bool(torch.equal(lyr.vstore.rows.materialise().reshape(H, n, d),
                            build_v_layer(vfull[0], cfg["k_lanes"], cfg["vbits"], d)
                            .materialise(H))), \
        "V after a fold does not match a one-shot build -- the fold reordered (head, token)"
    kpre = build_k_layer(K0[0], cfg["tols"], cfg["qbits"], cfg["res_frac"], cfg["res_bits"], d=d)
    assert bool(torch.equal(lyr.kstore.resid.materialise().reshape(H, n, d)[:, :n0],
                            kpre.resid.materialise().reshape(H, n0, d))), \
        "the fold moved the prefill rows of the K residual"

    print("  fold inserts per head: a folded store equals a one-shot build, row for row")


if __name__ == "__main__":
    self_test()
    self_test_fold()
