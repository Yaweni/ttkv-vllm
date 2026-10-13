"""Calibrate the per-layer `r1` table that vLLM needs BEFORE it can allocate the cache.

THE CIRCULARITY THIS BREAKS
    vLLM sizes the KV cache at init, so `customize_spec` has to publish each layer's slot width
    before any sequence exists. But `r1` is a property of the DATA -- it is the rank the TT-SVD
    needed for that prompt's keys. So it cannot be measured at prefill and used to size the cache in
    the same run. The only honest ways out are a calibration pass whose table ships as an artifact,
    or capping the rank (which is a format change: a cap of 64 under-ranks `tight` by 1.75x). This
    tool produces the artifact.

WHAT IT MUST NOT DO
    Reimplement the rank rule. It calls `ttkv_codec.tt_svd` -- the same function `build_k_layer`
    calls -- with the same `tol`, on the same tensor layout (`[n, d, H]`, tokens first). A second
    copy of the truncation rule that agreed today would drift later and fail SILENTLY, because a
    wrong `r1` is still a perfectly valid rank: the store still reconstructs, it is just coarser or
    fatter than it needs to be.

WHY MAX OVER SAMPLES, AND NOT A MEAN
    `r1` varies by layer AND by text. Measured on a single 16k prompt, per-layer `r1` spans
    6..166; measured across corpora at 16k, prose needs 85 where RULER filler needs 21. A mean
    under-ranks roughly half the traffic. Max-over-samples never under-ranks the calibration set.
    It can still under-rank unseen text -- that is what the corpus choice is for, which is why the
    serving profile was settled first (prose, not RULER filler).

CAPTURE IS PRE-ROPE, from `k_proj`
    That is the storage mode the plugin ships, so the rank measured here is the rank the cache will
    actually need. `v_proj` is deliberately not captured: V's lane count is an absolute per-row
    budget, not something the data decides.

    The forward runs in bf16 for speed. That cannot move the rank: bf16 rounding is ~2e-3 relative,
    an energy share of ~4e-6, against a truncation threshold of `tol**2` = 0.04.

USAGE
    python tools/calibrate_r1.py --model meta-llama/Llama-3.2-3B \
        --context 16384 --samples 4 --corpus <longbench snapshot dir> \
        --out calibration/llama-3.2-3b-16384.json
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
import time

import torch

_HERE = pathlib.Path(__file__).resolve()
for _c in (_HERE.parents[1] / "codec", _HERE.parents[1], _HERE.parents[1] / "integration", pathlib.Path("/opt/ttkv/integration")):
    if _c.is_dir():
        sys.path.insert(0, str(_c))

# Fields to try, in order, when pulling long prose out of a parquet. LongBench puts the document in
# `context`; pg19/wikitext put it in `text`.
TEXT_FIELDS = ("context", "text", "input", "document", "content")


def _read_samples(corpus: str, field: str | None, need_tokens: int, samples: int):
    """Concatenate rows until each sample has at least `need_tokens` of text, `samples` times.

    Concatenating whole documents rather than slicing one is deliberate: the point is a realistic
    long-context key distribution, and a 16k span cobbled from many short documents is exactly what
    a 16k serving request looks like when it is a multi-document prompt.
    """
    import pyarrow.parquet as pq

    paths = sorted(glob.glob(str(pathlib.Path(corpus) / "**" / "*.parquet"), recursive=True))
    if not paths:
        paths = sorted(glob.glob(corpus))
    if not paths:
        raise SystemExit(f"no parquet under {corpus!r}")

    out, buf = [], []
    for p in paths:
        table = pq.read_table(p)
        names = table.column_names
        col = field if field in names else next((c for c in TEXT_FIELDS if c in names), None)
        if col is None:
            raise SystemExit(f"{p}: none of {TEXT_FIELDS} in {names}")
        for i in range(table.num_rows):
            v = table.column(col)[i].as_py()
            if not isinstance(v, str):
                continue
            buf.append(v)
            # cheap proxy for token count; the tokenizer trims to exactly `need_tokens` anyway
            if sum(len(x) for x in buf) >= need_tokens * 4:
                out.append("\n\n".join(buf))
                buf = []
                if len(out) >= samples:
                    return out, [pathlib.Path(x).name for x in paths]
    if buf:
        out.append("\n\n".join(buf))
    if not out:
        raise SystemExit("corpus produced no text")
    return out, [pathlib.Path(x).name for x in paths]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--context", type=int, default=16384)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--corpus", required=True, help="directory searched recursively for *.parquet")
    ap.add_argument("--field", default=None, help="text column; auto-detected if omitted")
    ap.add_argument("--tol", type=float, default=0.20, help="the SHIPPED tolerance")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ttkv_codec import tt_svd

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    texts, used = _read_samples(args.corpus, args.field, args.context, args.samples)
    print(f"corpus: {len(texts)} sample(s) from {len(used)} parquet file(s)")

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa",
    ).to(dev).eval()

    c = model.config
    H = int(c.num_key_value_heads)
    d = int(getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads)
    atts = [m for m in model.modules() if hasattr(m, "k_proj") and hasattr(m, "v_proj")]
    nl = len(atts)
    print(f"model: H={H} d={d} layers={nl}  KV width={H * d}")

    # per_layer[li] = [r1 for each sample]; the max is taken at the end.
    per_layer: list[list[int]] = [[] for _ in range(nl)]
    r2_layer: list[list[int]] = [[] for _ in range(nl)]

    for si, text in enumerate(texts):
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=args.context).input_ids.to(dev)
        n = int(ids.shape[1])
        if n < args.context:
            print(f"  sample {si}: only {n} tokens (wanted {args.context}); using what there is")
        cap: dict = {}
        handles = []
        for li, at in enumerate(atts):
            def _hook(_m, _i, out, li=li):
                cap[li] = out.detach()[0].reshape(n, H, d).permute(1, 0, 2).contiguous().cpu().float()
            handles.append(at.k_proj.register_forward_hook(_hook))
        with torch.no_grad():
            model(ids)
        for h in handles:
            h.remove()

        t0 = time.perf_counter()
        for li in range(nl):
            # EXACTLY the runtime call: tokens first, [n, d, H], same tol.
            _cores, ranks = tt_svd(cap[li].permute(1, 2, 0).contiguous(), args.tol)
            per_layer[li].append(int(ranks[0]))
            r2_layer[li].append(int(ranks[1]) if len(ranks) > 1 else 1)
        r_now = [per_layer[l][-1] for l in range(nl)]
        del cap
        print(f"  sample {si}: n={n} r1 min={min(r_now)} max={max(r_now)}  "
              f"({time.perf_counter() - t0:.1f}s of SVD)", flush=True)

    r1_max = [max(v) for v in per_layer]
    out = {
        "model": args.model,
        "context": args.context,
        "tol": args.tol,
        "heads": H,
        "head_dim": d,
        "kv_width": H * d,
        "layers": nl,
        "corpus": {"files": used, "samples": len(texts), "field": args.field or "auto"},
        "aggregation": "per-layer max over samples",
        "r1": r1_max,
        "r2": [max(v) for v in r2_layer],
        "r1_per_sample": per_layer,
    }
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {p}")
    print(f"  r1: min={min(r1_max)} max={max(r1_max)} mean={sum(r1_max) / len(r1_max):.1f}")
    print("  the spread across layers is why this stays per-layer rather than one global rank")
    return 0


if __name__ == "__main__":
    sys.exit(main())
