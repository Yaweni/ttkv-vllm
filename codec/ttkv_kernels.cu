// Fused TTKV container kernels.
//
// WHAT THESE REPLACE, AND WHY IT IS WORTH A KERNEL
// The torch path materialises D, a dense [H*n, hd] fp32 block, then scatters the codes into it and
// hands it to a batched matmul:
//     D = torch.zeros(H*n, hd);  D.masked_scatter_(mask, code*scale);  bmm(q_group, D^T)
// The profiler put `aten::masked_scatter_` at 44.4 ms/token over 64 calls -- the single largest
// item in the decode loop, larger than every remaining dequant, copy and reshape combined. Two
// thirds of a 16.8 MB buffer is a memset, and ~100 of every 128 lanes are structurally zero.
//
// The whole thing collapses because the sparsity is known at compile-of-the-problem time: the
// bitmap says which lanes exist, so the contraction can run over LANES instead of over hd. No D
// is built, nothing is scattered, and the zero lanes are never touched.
//
//     residual:  s[p, i] += sum_t  q[p, lane_t(i)] * ((code[i,t] - off) * rowscale[i])
//     decoder :  QT[p, a] = dec_scale[a] * sum_j dec_q[a, h(p)*hd + j] * q[p, j]
//
// EXACTNESS
// The lane order is not a free choice. `pack_bits_rows` writes MSB-first and `PackedRows.materialise`
// assigns the t-th code of a row to the t-th SET BIT of that row's bitmap, so iterating j upward and
// incrementing t on each set bit reproduces the same pairing. Getting this backwards would still
// produce plausible-looking scores, and the project's standing signal for that class of mistake is
// an ANTI-correlated metric, not noise.
//
// HQ and HD arrive as compile-time defines (TTKV_HQ, TTKV_HD) so the inner loop over query heads
// unrolls to constants. That is the reason this is built with NVRTC rather than a fixed cubin: the
// module is recompiled per head configuration and cached.

#define TQ TTKV_HQ
#define TD TTKV_HD
#define TG TTKV_G

// Tokens per KRT tile. MUST match KRT_TILE in ttkv_cuda.py -- it sets both the shared-memory size
// and the grid. 64 keeps the key tile at 64*129*4 = 33 KB, which fits two blocks per SM on sm_86
// while leaving the rotation a whole (j, j+TD/2) pair to each thread.
#define KT TTKV_KT

// Thread tiling for the reconstruction: KM tokens x KD lanes per thread. The block must be exactly
// (KT/KM)*(TD/KD) threads -- if a block were short the barriers below would be reached by a partial
// block, which is undefined and in practice deadlocks.
#define KM TTKV_KM
#define KD TTKV_KD
#define KX (TD / KD)     // lane groups per row
#define KY (KT / KM)     // token groups per tile

// MSB-first extraction of the t-th `bits`-wide value from a packed row, matching pack_bits_rows.
// Two bytes are read because a 7+3 split straddles a boundary; the bound check matters only for the
// last value of a row, and only because the row is byte-padded rather than byte-aligned to it.
__device__ __forceinline__ unsigned int ttkv_get_bits(const unsigned char* __restrict__ row,
                                                      int t, int bits, int PB) {
    int p = t * bits;
    int byte = p >> 3;
    int sh = p & 7;
    unsigned int hi = (byte + 1 < PB) ? (unsigned int)row[byte + 1] : 0u;
    unsigned int acc = ((unsigned int)row[byte] << 8) | hi;
    return (acc >> (16 - sh - bits)) & ((1u << bits) - 1u);
}

// QT[b, p, a] = dec_scale[a] * sum_j dec_q[a, h(p)*hd + j] * q[b, p, j]
//
// This is the Q-absorption, and it is why the decoder never has to be expanded into a per-head copy:
// `P_rep` in the torch path gathers the decoder to [r1, HQ, hd] once per layer per token -- 1.8 MB of
// pure index traffic -- to then contract it. Reading the decoder row directly with the head offset
// baked in gets the same number with none of the copy. One thread per rank index; the 32 query heads
// of a block read the same decoder row, so it is served from cache.
//
// BATCH ARRIVES AS A GRID AXIS, not as a Python loop. That is the whole reason batching is viable:
// the launch count per token is unchanged (3 kernels x 32 layers) no matter how large B is, so the
// ~8.7 us/launch toll does not multiply by B. A loop over batch would be 1536 launches/token at B=16
// -- about 13.5 ms/token of pure launch overhead.
//
// THE DECODER IS PER-BATCH, and that was not the first design. Pooling the batch into the TT-SVD fit
// (one shared rank-r1 basis for all sequences) is what makes the rows a simple relabelling of
// tokens, but it changes the ANSWER: the basis then has to span B sequences' worth of key diversity
// at the same rank, so each sequence is approximated more coarsely than it would be alone. Measured
// directly -- with a shared fit, a batch slice disagreed with that sequence's own B=1 run by 5.8 in
// absolute score, while the kernels still matched the pooled torch reference to cosine 1.0. That is
// a batching-induced accuracy change, which is not acceptable for a serving change. Hence B decoders.
extern "C" __global__ void ttkv_qt(const float* __restrict__ q,             // [B, TQ, TD]
                                   const signed char* __restrict__ dec_q,   // [B, r1, H*TD]
                                   const float* __restrict__ dec_scale,     // [B, r1]
                                   float* __restrict__ QT,                  // [B, TQ, r1]
                                   int n_rank, int G, int H) {
    const int bz = blockIdx.z;                      // batch element
    const int p = blockIdx.x;                       // query head
    const int hv = p / G;                           // its kv head
    const int a = blockIdx.y * blockDim.x + threadIdx.x;
    if (a >= n_rank) return;
    const signed char* drow = dec_q + ((long)bz * n_rank + a) * (H * TD) + (long)hv * TD;
    const float* qrow = q + (long)(bz * TQ + p) * TD;
    float acc = 0.f;
    #pragma unroll 8
    for (int j = 0; j < TD; ++j) acc += (float)drow[j] * qrow[j];
    QT[(long)(bz * TQ + p) * n_rank + a] = acc * dec_scale[(long)bz * n_rank + a];
}

// The K residual: s[b, p, i] += sum over the row's present lanes of q[b,p,lane] * (code-off)*scale.
//
// ONE BLOCK PER (token tile, KV HEAD, BATCH), and the block touches ONLY that kv head's query group.
// This is the mistake to not make again: the obvious version of this loop accumulates over all HQ
// query heads, which adds every kv head's residual to every query head -- 8 contributions where the
// math wants 1. It still runs, still produces numbers of the right magnitude, and reads as
// cosine 0.224 against the torch path rather than as an error. Llama's grouping is p = h*G + g, so
// for kv head hv the heads that read it are exactly [hv*G, hv*G + G).
//
// BATCH IS A GRID AXIS and each sequence owns a CONTIGUOUS SLAB: `row = bz*(H*n) + hv*n + i`. That
// layout is chosen so pooling B per-sequence containers is a plain `cat` along dim 0 -- no row
// permutation, no gather. It also means a block for (bz, hv) reads exactly one sequence's rows, so
// the reduction over cached tokens can never cross a sequence boundary.
//
// `n` remains the PER-SEQUENCE token count. Equal-length sequences are required; variable lengths
// would need a per-sequence row cursor instead of a uniform n. Each block's output is bit-identical
// to the B=1 run on that sequence, which is the sharpest available correctness test.
//
// q is transposed into shared memory as qs[lane][g] for just this batch element's G heads, so the
// inner loop reads G CONSECUTIVE floats rather than values strided by hd -- the strided form is what
// made the earlier gather experiments collapse. Accumulators live in registers and g fully unrolls.
extern "C" __global__ void ttkv_resid_add(const float* __restrict__ q,             // [B, TQ, TD]
                                          const unsigned char* __restrict__ packed,
                                          const unsigned char* __restrict__ bitmap,
                                          const float* __restrict__ row_scale,
                                          float* __restrict__ s,                   // [B, TQ, n]
                                          int n, int PB, int bits) {
    extern __shared__ float qs[];                   // [TD][TG], this kv head's group only
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;     // token WITHIN this sequence
    const int hv = blockIdx.y;                      // kv head
    const int bz = blockIdx.z;                      // batch element

    if (i < n) {
        for (int idx = tid; idx < TD * TG; idx += blockDim.x) {
            const int j = idx / TG, g = idx - j * TG;
            qs[idx] = q[(long)(bz * TQ + hv * TG + g) * TD + j];
        }
    }
    __syncthreads();
    if (i >= n) return;

    const long row = (long)bz * ((long)gridDim.y * n) + (long)hv * n + i;
    const float sc = row_scale[row];
    const float off = (float)(1 << (bits - 1));
    const unsigned char* bm = bitmap + row * (TD >> 3);
    // NB: the packed row stride PB is bytes-per-row, and rows are byte-padded, so this is a plain
    // multiply -- no bit offset accumulates across rows.
    const unsigned char* pk = packed + row * PB;

    float acc[TG];
    const long base = (long)bz * TQ * n + (long)hv * TG * n + i;
    #pragma unroll
    for (int g = 0; g < TG; ++g) acc[g] = s[base + (long)g * n];

    int t = 0;
    // BYTE-MAJOR, not bit-major. The old form tested one bit per iteration under `#pragma unroll 1`,
    // which is exactly the directive that stops the compiler hoisting the byte load and folding the
    // shift -- so it re-loaded the same byte 8 times and paid a variable shift each time. Here the
    // byte is loaded once, its shifts are constants, and an empty byte skips 8 tests outright.
    const int nb = TD >> 3;
    #pragma unroll 1
    for (int b = 0; b < nb; ++b) {
        const unsigned int byte = (unsigned int)bm[b];
        if (!byte) continue;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (byte & (1u << (7 - k))) {
                const int j = b * 8 + k;
                const unsigned int u = ttkv_get_bits(pk, t, bits, PB);
                const float val = ((float)u - off) * sc;
                const float* qrow = qs + j * TG;
                #pragma unroll
                for (int g = 0; g < TG; ++g) acc[g] += val * qrow[g];
                ++t;
            }
        }
    }
    // ASSIGN, not `+=`: acc was seeded from s above, so accumulating here would add s back twice --
    // 2*s + residual. A zero-initialised s hides that completely, which is why the first version of
    // this kernel passed its isolated test and failed in the loop at cosine 0.085.
    #pragma unroll
    for (int g = 0; g < TG; ++g) s[base + (long)g * n] = acc[g];
}

// ------------------------------------------------------------------ V output
// o[b, g, j] = sum_i p[b, hv*G+g, i] * VD[hv, b*n+i, j], with VD the row's present lanes and 0
// elsewhere.
//
// This is the SAME densify-then-matmul that the K residual needed, but the shaped is transposed:
// the output is [G, hd] (tiny, and accumulated across ALL tokens) while the reduction runs over
// tokens. So the dense V block cannot stay in registers -- it has to be materialised somewhere.
//
// The trick is that "somewhere" does not have to be global memory. The torch path built a
// [H*n, hd] fp32 buffer in global (16.8 MB of memset + scatter + read back). Here the densification
// happens one 64-token TILE at a time in SHARED memory and is consumed immediately, so the global
// buffer never exists and the round trip never happens.
//
// BATCH IS A GRID AXIS, and this is the kernel where it MATTERS semantically: the reduction over
// tokens must not cross a sequence boundary, so each block owns exactly one (batch, kv head, token
// tile). Rolling the batch into the token axis would silently mix sequences here.
//
// Per block: zero a [TILE, hd] tile, scatter the tile's present lanes into it, then a small GEMM
// against the tile's attention weights. The only global traffic is p (read) and o (one atomic per
// output per block).
#define TV_TILE 64
#define TV_BLOCK 256
#if TD != 128
#error "ttkv_v_out assumes TD == 128: it splits each row across 4 threads x 32 lanes"
#endif

extern "C" __global__ void ttkv_v_out(const float* __restrict__ p,   // [B, TQ, p_stride] slice
                                      const unsigned char* __restrict__ packed,
                                      const unsigned char* __restrict__ bitmap,
                                      const float* __restrict__ row_scale,
                                      float* __restrict__ o,          // [B, TQ, TD], zeroed
                                      int n, int p_stride, int PB, int bits) {
    extern __shared__ float smem[];
    float* vd = smem;                        // [TV_TILE][TD], the tile's dense V
    float* ps = smem + TV_TILE * TD;         // [TG][TV_TILE], this tile's attention weights

    const int tid = threadIdx.x;
    const int hv = blockIdx.y;
    const int bz = blockIdx.z;
    const int t0 = blockIdx.x * TV_TILE;

    for (int idx = tid; idx < TV_TILE * TD; idx += TV_BLOCK) vd[idx] = 0.f;
    __syncthreads();

    // scatter: four threads per row, 32 lanes (4 bitmap bytes) each. `t` must start at the number of
    // set bits BEFORE this thread's chunk, which is a popcount of the preceding bytes -- not a
    // running counter, because the chunks run in parallel.
    {
        const int r = tid >> 2;
        const int q = tid & 3;
        const int i = t0 + r;
        if (i < n) {
            const long row = (long)bz * ((long)gridDim.y * n) + (long)hv * n + i;
            const unsigned char* bm = bitmap + row * (TD >> 3);
            const unsigned char* pk = packed + row * PB;
            const float sc = row_scale[row];
            const float off = (float)(1 << (bits - 1));
            int t = 0;
            for (int b = 0; b < q * 4; ++b) t += __popc((unsigned int)bm[b]);
            const int j0 = q * 32;
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const unsigned int byte = (unsigned int)bm[q * 4 + b];
                if (!byte) continue;
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    if (byte & (1u << (7 - k))) {
                        const unsigned int u = ttkv_get_bits(pk, t, bits, PB);
                        vd[r * TD + j0 + b * 8 + k] = ((float)u - off) * sc;
                        ++t;
                    }
                }
            }
        }
    }

    for (int idx = tid; idx < TG * TV_TILE; idx += TV_BLOCK) {
        const int g = idx / TV_TILE, r = idx - g * TV_TILE;
        const int i = t0 + r;
        ps[idx] = (i < n) ? p[(long)(bz * TQ + hv * TG + g) * p_stride + i] : 0.f;
    }
    __syncthreads();

    // reduce. Threads are laid out so that a warp reads vd[r][j] at consecutive j -- stride 1, no
    // bank conflicts -- while ps[g][r] is a broadcast.
    const int NG = TV_BLOCK / TD;
    const int j = tid % TD;
    const int gp = tid / TD;
    #pragma unroll
    for (int g = gp; g < TG; g += NG) {
        float a = 0.f;
        #pragma unroll 4
        for (int r = 0; r < TV_TILE; ++r) a += ps[g * TV_TILE + r] * vd[r * TD + j];
        atomicAdd(&o[(long)(bz * TQ + hv * TG + g) * TD + j], a);
    }
}

// ============================================================================================
// KRT -- the pre-RoPE score path. Reconstruct, rotate, contract; never materialise the block.
//
// WHY THE ABSORBED PATH CANNOT DO THIS. `ttkv_qt` folds the decoder into the QUERY
// (QT[p,a] = sum_j dec[a, h(p)*TD+j] * q[p,j]) so the token loop reduces to a rank-r1 dot product
// against the stored coefficients. That is legal only because every stored row shares one rotation:
// with POST-RoPE storage, rotating q once is the same operation as rotating every key, so hoisting
// it out of the loop is free. With PRE-RoPE storage it is not free -- each stored row carries its
// own position and therefore its own angle -- so q cannot be absorbed, and the contraction has to
// touch the reconstructed key. Q-absorption and pre-RoPE storage are mutually exclusive; there is
// no version of this where both survive.
//
// This kernel is the argument that the alternative is affordable. The reconstruction is a
// [KT, r1] x [r1, TD] product per head done once per tile INTO SHARED, and the 128-lane residual
// scatter is amortised over all TG query heads that share the kv head. The comparator is not the
// absorbed kernel but the torch path it replaces, which materialises a dense [H*n, TD] fp32 block
// and scatters into it -- 44 ms/token of `masked_scatter_` over 64 calls, measured.
//
// THE ORDER OF OPERATIONS IS THE CONTRACT. Two of the four steps fail quietly:
//   1. reconstruct   k = (sum_a cq[t,a]*dec_q[a, h*TD+j]) * cscale[t]
//      `cscale` scales the DECODER part and must NOT scale the residual -- the torch path builds
//      them separately and adds. Folding cscale across both shifts every row by the residual's
//      magnitude, which looks like a rank problem rather than an arithmetic one.
//   2. residual      add the lanes TT dropped, at their own positions (bitmap note below).
//   3. rotate        each row by ITS OWN ABSOLUTE position, pairing j with j+TD/2 -- NOT 2j,2j+1.
//      The pair choice is load-bearing: `_rotate_half` concatenates halves, so a thread owning
//      (j, j+TD/2) rotates in registers with no partner exchange and no second barrier.
//   4. contract      s[p, i0+t] = sum_j q[p,j] * Ks[t][j]
//
// LANE ORDER, the same trap `ttkv_resid_add` documents: `pack_bits_rows` writes MSB-first and the
// t-th code of a row belongs to the t-th SET BIT of that row's bitmap. Iterate j upward counting
// set bits and the pairing reproduces. Reversed, it still yields scores of the right magnitude and
// shows up as an anti-correlated metric rather than as noise.
//
// SHARED LAYOUT. Ks is [KT][TD+1]. The +1 pad is not decoration: the contraction reads Ks[t][j]
// with consecutive threads walking t at fixed j, so an unpadded row stride of 128 floats lands all
// 32 lanes of a warp on one bank. 129 is coprime with 32.
//
// ONE BLOCK PER (tile, KV HEAD), so a block writes only its own query group's scores: no atomics,
// and no cross-head reduction that would silently add every kv head to every query head.
// ============================================================================================
#define KB 256

#if (TD % 2) != 0
#error "ttkv_krt pairs j with j+TD/2: TD must be even"
#endif
#if (KT * (TD / 2)) % KB != 0 || (KT * TD) % KB != 0
#error "ttkv_krt's loops assume the tile divides evenly into KB threads (no barrier may diverge)"
#endif
#if (KT % KM) != 0 || (TD % KD) != 0
#error "the thread tiling must divide the tile exactly"
#endif
#if KY * KX != KB
#error "KM x KD must fill the block exactly -- a short block would diverge the barriers"
#endif
#if KX % 8 != 0
#error "the residual prefix sharing assumes a lane group spans whole bitmap bytes"
#endif

extern "C" __global__ void ttkv_krt(
        const signed char* __restrict__ coeff_q,   // [n, r1]
        const float* __restrict__ coeff_scale,     // [n]
        const float* __restrict__ dec_f,           // [r1, H*TD], fp32, ALREADY scaled by dec_scale
        const unsigned char* __restrict__ bitmap,  // [H*n, TD/8]
        const unsigned char* __restrict__ packed,  // [H*n, PB]
        const float* __restrict__ row_scale,       // [H*n]
        const float* __restrict__ cos,             // [n, TD]
        const float* __restrict__ sin,             // [n, TD]
        const float* __restrict__ q,               // [B, TQ, TD], already rotated
        float* __restrict__ s,                     // [B, TQ, n]
        int n, int n_rank, int G, int H, int PB, int bits, int has_res, int mode) {

    __shared__ float Ks[KT][TD + 1];

    const int tid = threadIdx.x;
    const int hv = blockIdx.y;
    // BATCH IS A GRID AXIS, NOT A LOOP -- the same convention `ttkv_qt`, `ttkv_resid_add` and
    // `ttkv_v_out` already use, so launch count per token is unchanged at 96 regardless of B and
    // the ~8.7 us/launch toll does not multiply. KRT was written after that convention and never
    // got it, which is the entire reason the pre-RoPE path was refused at B>1.
    //
    // bz == 0 reduces EVERY offset below to the single-sequence expression exactly, so the B=1 path
    // that all the recorded pre-cont numbers were measured on is unchanged to the byte.
    const int bz = blockIdx.z;
    const int i0 = blockIdx.x * KT;
    const int T = min(KT, n - i0);
    const int HD2 = TD >> 1;
    const int tx = tid % KX;            // lane group: this thread owns lanes tx + k*KX
    const int ty = tid / KX;            // token group: this thread owns tokens ty*KM + m
    const int tbase = ty * KM;
    const int dstride = H * TD;

    // ---- 1+2. reconstruct and patch, straight into shared. Nothing dense reaches global memory.
    //
    // KM TOKENS x KD LANES PER THREAD, and the two axes buy different things. This loop is the
    // kernel's dominant cost and it is INSTRUCTION bound, not bandwidth bound: measured, 0.679 of
    // 1.154 ms per layer. One output per thread spent a coefficient load, a scale load and a
    // decoder load on a SINGLE MAC. The token axis amortises the decoder load -- one row of the
    // decoder serves KM tokens -- and the lane axis amortises the coefficient load, which is one
    // byte per (token, rank) pair no matter how many lanes read it.
    //
    // KD lanes strided by KX, NOT KD contiguous lanes. Two reasons, and both are load-bearing:
    // the decoder read stays coalesced across the 16 lane groups, and the shared store
    // Ks[t][tx + k*KX] walks consecutive banks instead of landing four threads to a bank.
    if (mode & 1) {
        const signed char* cp[KM];
        float csf[KM];
        #pragma unroll
        for (int m = 0; m < KM; ++m) {
            // Clamped, not skipped: the last tile is partial, so token KM-1 of the last token
            // group can run past the store. Reading a duplicate row is harmless because the
            // rotation and contraction below already skip every t >= T; reading out of bounds is
            // not harmless.
            const int tc = min(i0 + tbase + m, n - 1);
            // coeff_q row is (batch, token) = bz*n + i. NOTE this is NOT the residual's formula
            // two steps below -- see the comment on `row`.
            cp[m] = coeff_q + (long)(bz * n + tc) * n_rank;
            csf[m] = coeff_scale[bz * n + tc];
        }
        // The decoder is PER SEQUENCE: `PooledKStore.decoder_f()` returns [B, n_rank, H*TD], so the
        // batch stride is n_rank * dstride. At bz=0 this is the old `dec_f + hv*TD + tx`.
        const float* dp = dec_f + (long)bz * n_rank * dstride + (long)hv * TD + tx;

        float acc[KM][KD];
        #pragma unroll
        for (int m = 0; m < KM; ++m)
            #pragma unroll
            for (int k = 0; k < KD; ++k) acc[m][k] = 0.f;

        #pragma unroll 2
        for (int a = 0; a < n_rank; ++a, dp += dstride) {
            float g[KM];
            // coeff_scale is per TOKEN and the token is fixed for this thread, so folding it in
            // here instead of after the reduction removes KM multiplies per rank step and makes
            // the expression match torch's `(coeff_q * coeff_scale) @ decoder_f()` term for term.
            #pragma unroll
            for (int m = 0; m < KM; ++m) g[m] = (float)cp[m][a] * csf[m];
            #pragma unroll
            for (int k = 0; k < KD; ++k) {
                const float dk = dp[k * KX];
                #pragma unroll
                for (int m = 0; m < KM; ++m) acc[m][k] += g[m] * dk;
            }
        }
        #pragma unroll
        for (int m = 0; m < KM; ++m) {
            float* Krow = Ks[tbase + m];
            #pragma unroll
            for (int k = 0; k < KD; ++k) Krow[tx + k * KX] = acc[m][k];
        }

        if (has_res && (mode & 2)) {
            #pragma unroll
            for (int m = 0; m < KM; ++m) {
                const int tk = i0 + tbase + m;
                if (tk >= n) continue;
                // DIFFERENT ROW FORMULA FROM coeff_q ABOVE, and this is the trap in this change.
                // The residual is a pooled PackedRows whose slab order is
                //     (batch, head, token) = bz*(H*n) + hv*n + i,
                // while coeff_q is (batch, token) = bz*n + i. Both are correct for their own store
                // and mixing them reads another sequence's row -- bounded, in range, correctly
                // shaped, and wrong. Same failure class as the _pad_rank token-axis bug.
                const long row = (long)bz * H * n + (long)hv * n + tk;
                const unsigned char* bm = bitmap + row * (TD >> 3);
                const float rs = row_scale[row];
                const float off = (float)(1 << (bits - 1));
                const int ji = tx >> 3, jr = tx & 7;   // KX = 16 is two whole bitmap bytes
                int ci = 0;
                for (int b = 0; b < ji; ++b) ci += __popc((unsigned)bm[b]);
                float* Krow = Ks[tbase + m];
                // The KD lanes sit (KX/8) bitmap BYTES apart and share jr, so each lane's prefix
                // count is the previous lane's plus one popcount per intervening byte. Computed
                // once here rather than recomputed from byte 0 for every lane.
                #pragma unroll
                for (int k = 0; k < KD; ++k) {
                    const int bb = ji + k * (KX >> 3);
                    if (k) {
                        #pragma unroll
                        for (int b = 0; b < (KX >> 3); ++b)
                            ci += __popc((unsigned)bm[bb - (KX >> 3) + b]);
                    }
                    const unsigned int cur = (unsigned)bm[bb];
                    if ((cur >> (7 - jr)) & 1u) {
                        const unsigned int u = ttkv_get_bits(
                            packed + row * PB, ci + __popc(cur >> (8 - jr)), bits, PB);
                        Krow[tx + k * KX] += ((float)u - off) * rs;
                    }
                }
            }
        }
    }
    __syncthreads();

    // ---- 3. rotate by absolute position. One thread owns the pair (j, j+TD/2), so both halves sit
    //         in registers: no exchange, no extra barrier. Pairs are disjoint, so a writer and a
    //         reader never touch the same word even without the barrier above.
    if (mode & 4) {
    for (int idx = tid; idx < KT * HD2; idx += KB) {
        const int t = idx / HD2;
        const int j = idx - t * HD2;
        if (t >= T) continue;
        const long a0 = (long)(i0 + t) * TD + j;
        const float x  = Ks[t][j];
        const float xp = Ks[t][j + HD2];
        const float c = cos[a0], sn = sin[a0];
        // THE HALVES CARRY THE SAME ANGLE, so the second pair of loads is redundant. Checked on
        // the real tensors the kernel is handed rather than assumed: `_cos_sin` builds the [n, TD]
        // table by duplicating the [n, TD/2] ramp, and cos[:, :TD/2] == cos[:, TD/2:] exactly. So
        // every element of the angle table is fetched TWICE per block, once at j and once at
        // j + TD/2, and the rotate pass pays for the second fetch. Halving it is bit-identical by
        // construction; the identity gate checks that claim instead of trusting it.
        const float ca = c, sa = sn;
        Ks[t][j]       = x * c - xp * sn;
        Ks[t][j + HD2] = xp * ca + x * sa;
    }
    }
    __syncthreads();

    // ---- 4. contract. Consecutive threads take consecutive tokens at one query head: the q row is
    //         a broadcast and the s writes are contiguous. Ks[t][j] walks t at fixed j, i.e. stride
    //         TD+1 across the padded rows.
    if (mode & 4) {
    for (int idx = tid; idx < TG * KT; idx += KB) {
        const int t = idx % KT;
        const int g = idx / KT;
        if (t >= T) continue;
        const int p = hv * TG + g;
        const float* qrow = q + (long)(bz * TQ + p) * TD;
        float acc = 0.f;
        #pragma unroll 8
        for (int j = 0; j < TD; ++j) acc += qrow[j] * Ks[t][j];
        s[(long)(bz * TQ + p) * n + i0 + t] = acc;
    }
    }
}
