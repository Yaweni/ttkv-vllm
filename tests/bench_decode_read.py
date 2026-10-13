"""What the DECODE READ costs per layer-call, and whether it is work or waiting.

The read is the other half of the per-step overhead: `impl._decode_one` gathers each request's slots
out of the paged cache, splits the K||V blob, builds the angle rows, reconstructs-and-rotates every
cached key against the query (`krt`), softmaxes, and decodes V back out (`v_out`). Seven calls per
layer, 28 layers a step, and the engine's phase table puts the whole thing at ~0.65 ms a layer-call --
but a phase on a deep queue measures the DRAIN, so the table cannot say whether that is arithmetic,
launch overhead, or waiting. This does, the same way `bench_write_path.py` does for the write:
host wall with the queue deep against host wall with it EMPTY, GPU time from stream events, and a
per-phase split. It also sweeps context, because the one number the whole design rests on is the
MARGINAL cost per cached token -- our claim is ~0.2 us against stock's ~0.6, and it has only ever
been measured to 4.3k tokens.

THE GEOMETRY IS THE SHIPPED MODEL'S. H=8, d=128, HQ=24 -- Llama-3.2-3B, G=3, which is NOT a power of
two, so `krt_triton` pads the query group axis and takes a contiguous copy on the way out. That
padding is inside the timed path here on purpose: it is on the decode path of the model we actually
serve. `HQ=32` (G=4) is measured alongside it to price the padding on its own.

The dense arm is a bandwidth yardstick, not a competitor: `scaled_dot_product_attention` over an
fp16 cache of the same length, which is what vLLM's attention costs in bytes (4 KB a token a layer
against our 912 B). Same card, same lengths, so the crossing is a measurement rather than a model.

USAGE
    PYTHONPATH=/opt/ttkv/integration python tests/bench_decode_read.py
"""
from __future__ import annotations

import pathlib
import sys

import torch

_HERE = pathlib.Path(__file__).resolve()
for _cand in (_HERE.parents[1] / "codec", _HERE.parent, _HERE.parents[1], _HERE.parents[2] / "integration",
              pathlib.Path("/opt/ttkv/integration")):
    if _cand.is_dir():
        sys.path.insert(0, str(_cand))

from ttkv_codec import CONFIGS  # noqa: E402
from vllm_ttkv import decode as dc  # noqa: E402
from vllm_ttkv import encode as enc  # noqa: E402
from vllm_ttkv import prof  # noqa: E402
from vllm_ttkv.krt_triton import krt_from_slots  # noqa: E402
from vllm_ttkv.layout import layout_for_config  # noqa: E402
from vllm_ttkv.v_out_triton import v_out_from_slots  # noqa: E402

# The instruments are `bench_write_path`'s, imported rather than copied: one definition of "work vs
# wait" for both benches, so a change to how that is measured cannot drift between them.
from bench_write_path import _gpu_ms, _host_ms, _host_ms_empty, _phases  # noqa: E402

H, D = 8, 128                      # kv heads and head dim: Llama-3.2-3B
HQ = 24                            # G=3 -- not a power of two, so `krt` pads
LAYERS = 28
BLOCK = 16                         # vLLM's page size, so slots are the engine's flat ids
CFG = "safe"
R1 = 112
N_DECODE = 1082                    # the harness's 60-repeat prompt
SWEEP = (1082, 4322, 8192, 16384, 32768)


def _need_cuda():
    if not torch.cuda.is_available():
        print("  SKIPPED: no CUDA device")
        return False
    return True


def _cos_sin_cache(max_len: int, d: int = D, theta: float = 10000.0):
    """The model's rotary table, in the shape `decode.cos_sin_from_cache` takes."""
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32, device="cuda") / d))
    freqs = torch.outer(torch.arange(max_len, device="cuda").float(), inv)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def _store(lay, cfg, n_tok, v_dtype=torch.bfloat16):
    """A fitted decoder plus a store of `n_tok` rows -- exactly what a decode step reads.

    Every row is projected against the decoder that the first 64 rows fitted, which is the shipped
    semantics at decode: the fit belongs to the request's first chunk, and every token after it is
    encoded against that frozen basis.
    """
    torch.manual_seed(7)
    rank = 24
    K = ((torch.randn(n_tok, rank, device="cuda") * 0.2)
         @ (torch.randn(rank, H * D, device="cuda") * 0.2)
         ).reshape(n_tok, H, D).permute(1, 0, 2).contiguous()
    K = K + 0.02 * torch.randn(H, n_tok, D, device="cuda")
    fitted = enc.fit_into_budget(K[:, :64].contiguous(), lay, cfg)
    fk = enc.encode_frozen(K, fitted.decoder, lay)
    V = (torch.randn(H, n_tok, D, device="cuda") * 0.3).to(v_dtype)
    C = lay.compressed_bytes_per_token()
    # vLLM's own page shape, which is what the engine hands this path: [blocks, block_size, C].
    n_blocks = -(-(n_tok + 1) // BLOCK)
    cache = torch.zeros(n_blocks, BLOCK, C // 2, dtype=torch.bfloat16, device="cuda")
    slots = torch.arange(n_tok, dtype=torch.int32, device="cuda")
    enc.pack_and_write(cache, slots, fk, V, lay, check=False)
    torch.cuda.synchronize()
    return cache, slots, fk.decoder.decoder_f


def _read_call(lay, cache, slots, decoder_f, csc, q, scale, phases=False):
    """`impl._decode_one`'s calls, in its order, on a store that is already written."""
    C = lay.compressed_bytes_per_token()
    n = int(slots.numel())
    pos = torch.arange(n, device="cuda")

    def body():
        blob = dc.gather_slots(cache, slots, C, check=False)
        k_slots, v_slots = dc.split_blob(lay, blob)
        cos, sin = dc.cos_sin_from_cache(csc, pos, D, max_position=n - 1)
        scores = krt_from_slots(lay, k_slots, decoder_f, cos, sin, q)
        p = torch.softmax(scores * scale, dim=-1)
        return v_out_from_slots(lay, v_slots, p)

    def phased():
        with prof.phase("gather"):
            blob = dc.gather_slots(cache, slots, C, check=False)
        with prof.phase("split_blob"):
            k_slots, v_slots = dc.split_blob(lay, blob)
        with prof.phase("cos_sin"):
            cos, sin = dc.cos_sin_from_cache(csc, pos, D, max_position=n - 1)
        with prof.phase("krt"):
            scores = krt_from_slots(lay, k_slots, decoder_f, cos, sin, q)
        with prof.phase("softmax"):
            p = torch.softmax(scores * scale, dim=-1)
        with prof.phase("v_out"):
            return v_out_from_slots(lay, v_slots, p)

    return (phased if phases else body), n


def _dense_call(n, hq):
    """`scaled_dot_product_attention` over an fp16 cache of the same length, in the BYTES stock pays.

    Stock keeps `H` kv heads, not `hq`, so that is the k/v this builds: 2*H*D*2 = 4 KB a token a
    layer, against our slot. The query keeps the model's head count and the group is expanded inside
    the kernel.
    """
    k = torch.randn(1, H, n, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, H, n, D, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(1, hq, 1, D, dtype=torch.bfloat16, device="cuda")

    def call():
        try:
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        except TypeError:                       # older torch: expand the groups by hand
            g = hq // H
            return torch.nn.functional.scaled_dot_product_attention(
                q, k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1))

    return call


def _one_geometry(cfg, lay, n_tok, hq, phases=False):
    cache, slots, decoder_f = _store(lay, cfg, n_tok)
    csc = _cos_sin_cache(n_tok + 8)
    q = torch.randn(hq, D, device="cuda") * 0.1
    return _read_call(lay, cache, slots, decoder_f, csc, q, D ** -0.5, phases=phases)


def _shape(lay, cfg):
    """Work or wait, and which call it is -- at the harness's decode length, G=3 and G=4."""
    print(f"\n=== the read's shape, {N_DECODE} cached tokens (Llama-3.2-3B geometry) ===")
    for hq in (HQ, 32):
        call, n = _one_geometry(cfg, lay, N_DECODE, hq)
        iters = max(20, 100_000 // n)
        host = _host_ms(call, iters)
        empty = _host_ms_empty(call, max(4, iters // 4))
        gpu = _gpu_ms(call, iters)
        tag = f"HQ={hq} (G={hq // H})"
        print(f"  {tag:16s} host deep {host:6.3f}  host empty {empty:6.3f}  gpu {gpu:6.3f}"
              f"  {'-> work' if empty > 0.6 * host else '-> waiting'}")
    print()
    call, _n = _one_geometry(cfg, lay, N_DECODE, HQ, phases=True)
    for name, ms in sorted(_phases(call, 60).items(), key=lambda kv: -kv[1]):
        print(f"    {name:32s} {ms:7.3f} ms/call   {ms * LAYERS:6.1f} ms/step over {LAYERS} layers")


def _marginal(lay, cfg):
    """ms a layer-call against context, ours and dense: the number the 2x claim rests on."""
    print("\n=== per cached token, ours against a dense fp16 cache of the same length ===")
    print(f"  {'tokens':>8s} {'ours ms':>9s} {'ours us/tok':>12s} {'dense ms':>9s} {'dense us/tok':>13s}"
          f" {'dense/ours':>11s}")
    rows = []
    for n_tok in SWEEP:
        call, n = _one_geometry(cfg, lay, n_tok, HQ)
        iters = max(20, 100_000 // n)
        ours = _host_ms(call, iters)
        dense = _host_ms(_dense_call(n, HQ), iters)
        rows.append((n, ours, dense))
        print(f"  {n:8d} {ours:9.3f} {ours * 1e3 / n:12.3f} {dense:9.3f} {dense * 1e3 / n:13.3f}"
              f" {dense / ours:11.2f}")
    if len(rows) >= 2:
        n0, o0, d0 = rows[0]
        n1, o1, d1 = rows[-1]
        print(f"\n  marginal over the sweep: ours {(o1 - o0) * 1e3 / (n1 - n0):.3f} us/token/layer, "
              f"dense {(d1 - d0) * 1e3 / (n1 - n0):.3f} us/token/layer")
        print(f"  per STEP at {n1} tokens over {LAYERS} layers: ours {o1 * LAYERS:.1f} ms, "
              f"dense {d1 * LAYERS:.1f} ms (the model's own weights are ~20 ms on this card)")
    return rows


def _krt_only(lay, cache, slots, decoder_f, csc, n):
    """The pieces that do NOT depend on the tiling, computed once so the sweep times `krt` alone."""
    blob = dc.gather_slots(cache, slots, lay.compressed_bytes_per_token(), check=False)
    k_slots, _v_slots = dc.split_blob(lay, blob)
    cos, sin = dc.cos_sin_from_cache(csc, torch.arange(n, device="cuda"), D, max_position=n - 1)
    return k_slots, cos, sin


def _tile_sweep(lay, cfg, n_tok, hq=HQ):
    """Does the read's cost follow the TILE COUNT or the arithmetic? The two are separable.

    `_krt_pow2` launches a grid of `(ceil(n/kt), H)` programs, and each program reads the decoder --
    `[r1, H*d]` fp32, ~460 KB at H=8, d=128, r1=112. The ARITHMETIC is invariant in `kt`: the same
    rows, the same `r1*d` reconstruction per row, the same dot. So if the cost is decoder traffic it
    falls with the tile count (`1/kt`); if it is the reconstruction itself, `kt` does nothing.

    The scores are checked equal across the sweep, because a tiling knob that changed the answer
    would make every number here meaningless -- and a silent change is exactly the failure this
    kernel's other conventions are guarded against.

    WHAT IT FOUND, and the hypothesis it killed: the decoder story was wrong. `kt=32/64` cost 1.39/1.33
    ms at 16384 tokens; `kt=128` cost 14.6, `kt=256` 59.0. Smaller tiles mean MORE decoder re-reads
    and cost nothing; taller tiles mean fewer and cost everything, because `KT` sizes a token axis
    held in the block. So the tiling is already at its optimum and neither `kt` nor `br` is the lever
    (br=128 at kt=128 asks for 196 KB of shared memory against the card's 101 KB). What is left in
    `krt` at kt=64 is the reconstruction itself: 16384 rows x 8 heads x r1*d = 1.9 GFLOP in 1.33 ms
    is ~10% of this card's fp32 peak. That is the number to attack, and it is an implementation
    number -- the algebra is fixed by storing pre-rotary keys.
    """
    cache, slots, decoder_f = _store(lay, cfg, n_tok)
    csc = _cos_sin_cache(n_tok + 8)
    q = torch.randn(hq, D, device="cuda") * 0.1
    k_slots, cos, sin = _krt_only(lay, cache, slots, decoder_f, csc, n_tok)
    iters = max(3, 60_000 // n_tok)
    print(f"\n  --- krt tiles at {n_tok} tokens (grid = ceil(n/kt) x H = {lay.H} heads) ---")
    print(f"  {'kt':>6s} {'programs':>9s} {'ms/call':>9s} {'ms/step':>8s} {'us/token/layer':>15s}"
          f" {'decoder MB':>11s} {'agree':>6s}")
    ref = None
    for kt in (32, 64, 128, 256):
        call = (lambda kt=kt: krt_from_slots(lay, k_slots, decoder_f, cos, sin, q, kt=kt, br=32))
        got = call()
        if ref is None:
            ref = got
        ok = torch.allclose(got, ref, rtol=1e-4, atol=1e-4)
        ms = _host_ms(call, iters)
        programs = -(-n_tok // kt) * lay.H
        # Each program handles ONE head, so what it re-reads is that head's decoder slice
        # `[r1, d]` -- not the whole `[r1, H*d]`. Getting this wrong by 8x is what made the
        # decoder-traffic story look plausible before the sweep measured it.
        mb = programs * lay.r1 * lay.d * 4 / 1e6
        print(f"  {kt:6d} {programs:9d} {ms:9.3f} {ms * LAYERS:8.1f} {ms * 1e3 / n_tok:15.3f}"
              f" {mb:11.0f} {str(ok):>6s}")
    print(f"  {'br':>6s} {'kt':>6s} {'ms/call':>9s} {'agree':>6s}   (br tiles the reduction axis;"
          f" br=128 at kt=128 asks for 196 KB of shared memory and the card has 101 KB)")
    for br in (16, 32, 64):
        call = (lambda br=br: krt_from_slots(lay, k_slots, decoder_f, cos, sin, q, kt=128, br=br))
        got = call()
        ok = torch.allclose(got, ref, rtol=1e-4, atol=1e-4)
        print(f"  {br:6d} {128:6d} {_host_ms(call, iters):9.3f} {str(ok):>6s}")
    base = _host_ms(lambda: krt_from_slots(lay, k_slots, decoder_f, cos, sin, q, kt=128, br=32), iters)
    for hq2 in (hq, 32):
        q2 = torch.randn(hq2, D, device="cuda") * 0.1
        ms = _host_ms(lambda q2=q2: krt_from_slots(lay, k_slots, decoder_f, cos, sin, q2, kt=128,
                                                   br=32), iters)
        print(f"  HQ={hq2:<3d} G={hq2 // lay.H}  kt=128  br=32              {ms:9.3f}"
              f"  {ms / base:6.2f}x the HQ={hq} arm")


def main() -> int:
    if not _need_cuda():
        return 0
    cfg = CONFIGS[CFG]
    lay = layout_for_config(cfg, r1=R1, H=H, d=D)
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"layout: r1={lay.r1} res_k={lay.res_k}@{lay.res_bits}b v_lanes={lay.v_lanes}@{lay.v_bits}b "
          f"slot={lay.compressed_bytes_per_token()} B   (dense fp16 = {2 * H * D * 2} B)")
    _shape(lay, cfg)
    _marginal(lay, cfg)
    for n_tok in (16384, 32768):
        _tile_sweep(lay, cfg, n_tok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
