"""How much accuracy does FOLDING cost, as a function of the first chunk size?

THE DECISION THIS SETTLES
    Under fold, the store's decoder is fitted once -- on the first prefill chunk -- and every later
    token is encoded by projecting onto it. Measured, that leaves the decoder under-ranked, because
    `r1` grows with `n`: per-layer mean 59.7 at n=512, 87.3 at 2048, 100.7 at 16384 on prose. A
    decoder fitted on 2048 tokens is therefore ~14% short of what 16k wants, and on 512 tokens it is
    ~40% short.

    Rank shortfall is only a proxy. The number that decides the design is the RECONSTRUCTION ERROR
    it causes, because that is what reaches the attention scores -- and the residual patch is what
    has to absorb it, from a lane budget that is a fixed FRACTION of d and therefore cannot grow.

WHY IT CAN BE MEASURED WITHOUT vLLM
    Folding is arithmetic on a key block: fit a decoder on a prefix, project the whole block onto
    it, quantise, patch the residual. `build_k_layer` already does all of that except the "frozen
    decoder" part, and `PackedKLayer.decoder_f()` exposes the decoder it fitted. So both arms run
    through the SHIPPED codec rather than through a reimplementation -- which matters here, because
    the residual scheme is subtle enough that a second copy would measure something else.

WHAT IT CANNOT TELL YOU
    The absolute error, only the GAP. Real serving sees the error through attention, and the
    relevant scale is whether a score difference is large enough to change a token. This reports
    relative Frobenius error and per-row cosine; the attn-level test is the e2e layer's job.

USAGE
    python tools/measure_fold_gap.py --model meta-llama/Llama-3.2-3B \
        --context 16384 --chunks 512 2048 4096 8192 --corpus <longbench dir>
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import sys

import torch

_HERE = pathlib.Path(__file__).resolve()
for _c in (_HERE.parents[1] / "codec", _HERE.parents[1], _HERE.parents[1] / "integration", pathlib.Path("/opt/ttkv/integration")):
    if _c.is_dir():
        sys.path.insert(0, str(_c))

TEXT_FIELDS = ("context", "text", "input", "document", "content")


def _one_sample(corpus: str, need_tokens: int) -> str:
    import pyarrow.parquet as pq

    paths = sorted(glob.glob(str(pathlib.Path(corpus) / "**" / "*.parquet"), recursive=True))
    buf = []
    for p in paths:
        t = pq.read_table(p)
        col = next((c for c in TEXT_FIELDS if c in t.column_names), None)
        if col is None:
            continue
        for i in range(t.num_rows):
            v = t.column(col)[i].as_py()
            if isinstance(v, str):
                buf.append(v)
            if sum(len(x) for x in buf) >= need_tokens * 4:
                return "\n\n".join(buf)
    return "\n\n".join(buf)


def _encode_frozen(K: torch.Tensor, dec_f: torch.Tensor, qbits: int, res_frac: float,
                   res_bits: int) -> torch.Tensor:
    """Encode K [H, n, d] onto a FROZEN decoder. Mirrors `build_k_layer`, minus the fit.

    The row conventions are copied deliberately rather than invented: `build_k_layer` builds
    `kflat = c @ dec` with column index `m = h*dd + j`, compares it against `K.reshape(H, n, dd)`,
    and keeps the residual in that (head, token) order. Re-deriving the layout here instead of
    copying it is how a measurement ends up reporting a misalignment as an accuracy loss.
    """
    H, n, d = K.shape
    r1 = dec_f.shape[0]
    M = K.permute(1, 0, 2).reshape(n, H * d).float()          # token-major, m = h*d + j

    c = M @ torch.linalg.pinv(dec_f)                          # [n, r1] least squares
    cscale = c.abs().amax(1, keepdim=True) / (2 ** (qbits - 1) - 1)
    cscale = torch.where(cscale == 0, torch.ones_like(cscale), cscale)
    cq = torch.clamp(torch.round(c / cscale), -(2 ** (qbits - 1)), 2 ** (qbits - 1) - 1)

    Kd = ((cq * cscale) @ dec_f).reshape(n, H, d).permute(1, 0, 2).contiguous()   # [H, n, d]
    R = (K.reshape(H, n, d).float() - Kd).reshape(H * n, d)

    if res_frac > 0:
        kr = max(int(d * res_frac), 1)
        mask = torch.zeros_like(R, dtype=torch.bool)
        mask.scatter_(1, torch.topk(R.abs(), kr, dim=1).indices, True)
        rmax = torch.where(mask, R.abs(), torch.zeros_like(R)).amax(1, keepdim=True)
        scl = torch.where(rmax == 0, torch.ones_like(rmax), rmax) / (2 ** (res_bits - 1) - 1)
        q = torch.clamp(torch.round(R / scl), -(2 ** (res_bits - 1)), 2 ** (res_bits - 1) - 1)
        Kd = Kd + (q.float() * scl).reshape(H, n, d) * mask.reshape(H, n, d)
    return Kd


def _err(K: torch.Tensor, K_hat: torch.Tensor) -> tuple[float, float]:
    """(relative Frobenius error, mean per-row cosine). Row = one (head, token) key vector."""
    a = K.reshape(-1, K.shape[-1]).float()
    b = K_hat.reshape(-1, K.shape[-1]).float()
    rel = float((a - b).norm() / a.norm().clamp_min(1e-30))
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
    return rel, float(cos)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--context", type=int, default=16384)
    ap.add_argument("--chunks", type=int, nargs="+", default=[512, 2048, 4096, 8192])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cfg", default="safe", help="allocation: safe | tight | campaign4x")
    ap.add_argument("--layers", type=int, default=0, help="0 = all layers")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ttkv_codec import CONFIGS
    from ttkv_packed_cache import build_k_layer

    cfg = CONFIGS[args.cfg]
    tols, qbits = cfg["tols"], cfg["qbits"]
    res_frac, res_bits = cfg["res_frac"], cfg["res_bits"]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    text = _one_sample(args.corpus, args.context)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(dev).eval()
    c = model.config
    H = int(c.num_key_value_heads)
    d = int(getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads)
    atts = [m for m in model.modules() if hasattr(m, "k_proj") and hasattr(m, "v_proj")]
    nl = len(atts) if args.layers <= 0 else min(args.layers, len(atts))

    ids = tok(text, return_tensors="pt", truncation=True, max_length=args.context).input_ids.to(dev)
    n = int(ids.shape[1])
    print(f"{args.model}  H={H} d={d} layers={nl}  n={n}  cfg={args.cfg} "
          f"(tol={tols} qbits={qbits} res_frac={res_frac} res_bits={res_bits})")

    cap: dict = {}
    handles = []
    for li, at in enumerate(atts[:nl]):
        def _hook(_m, _i, out, li=li):
            cap[li] = out.detach()[0].reshape(n, H, d).permute(1, 0, 2).contiguous().cpu().float()
        handles.append(at.k_proj.register_forward_hook(_hook))
    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()

    chunks = [c_ for c_ in args.chunks if c_ < n]
    print(f"\n{'chunk':>7} {'r1 fit':>7} {'r1 full':>8} {'relF norm':>10} {'cos':>8}   (fold vs one-shot)")
    rows = {}
    for ch in [0] + chunks:          # 0 = one-shot, the reference
        r1s, rels, coss = [], [], []
        if ch == 0:
            for li in range(nl):
                K = cap[li]
                lay = build_k_layer(K, tols, qbits=qbits, res_frac=res_frac, res_bits=res_bits)
                r1s.append(lay.r1)
                rel, cos = _err(K, lay.materialise(H, torch.float32))
                rels.append(rel); coss.append(cos)
            print(f"{'one-shot':>7} {sum(r1s)/len(r1s):>7.1f} {sum(r1s)/len(r1s):>8.1f} "
                  f"{sum(rels)/len(rels):>10.5f} {sum(coss)/len(coss):>8.4f}   <- reference")
            rows[ch] = (sum(r1s) / len(r1s), rels, coss, None)
            continue

        for li in range(nl):
            K = cap[li]
            # 1. fit on the first chunk ONLY -- this is the fold design
            lay_chunk = build_k_layer(K[:, :ch], tols, qbits=qbits,
                                      res_frac=res_frac, res_bits=res_bits)
            dec_f = lay_chunk.decoder_f()
            r1s.append(lay_chunk.r1)
            # 2. encode the WHOLE sequence onto that frozen decoder
            rel, cos = _err(K, _encode_frozen(K, dec_f, qbits, res_frac, res_bits))
            rels.append(rel); coss.append(cos)
        print(f"{ch:>7} {sum(r1s)/len(r1s):>7.1f} {rows[0][0]:>8.1f} "
              f"{sum(rels)/len(rels):>10.5f} {sum(coss)/len(coss):>8.4f}")
        rows[ch] = (sum(r1s) / len(r1s), rels, coss, None)

    print()
    base = rows[0]
    print("cost of folding, relative to the one-shot reference:")
    for ch in chunks:
        r1m, rels, coss, _ = rows[ch]
        print(f"  chunk={ch:5d}: rank {100*r1m/base[0]:5.1f}% of full | "
              f"err {sum(rels)/len(rels)/ (sum(base[1])/len(base[1])):5.2f}x reference | "
              f"cos {sum(coss)/len(coss):.4f} vs {sum(base[2])/len(base[2]):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
