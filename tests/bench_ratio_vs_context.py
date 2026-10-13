"""Why did the e2e fixture read 3.5x when the campaign says 5-7x? Three separate reasons.

The 3.514x in `test_e2e_forward.py` is a TEST FIXTURE number and it was never a shipped config. It
differs from the campaign figures for three independent reasons, and they compound:

1. WRONG ALLOCATION. The fixture derives `res_frac=0.5, res_bits=4, lanes=0.5*d, vbits=4`, which is
   the `campaign4x` allocation -- the WEAKEST of the three by some margin (its own label is 4.252x
   against `safe` 5.12x and `tight` 7.09x). `campaign4x` is archived and explicitly not a release
   point.
2. WRONG CONTEXT LENGTH. It runs at n=48. The decoder is `r1 * H*d` bytes PER LAYER, shared by every
   token in that layer, so its cost per token is `r1*H*d/n` -- it nearly vanishes at 16k and is a
   large fraction of the bill at 48. The campaign numbers are at 16384.
3. WRONG GEOMETRY. Qwen2-0.5B is d=64, H=2, where the per-row overheads (an 8 B bitmap and a 2 B
   ruler against a 128 B dense row) bite harder than at d=128 where they were tuned.

WHAT THIS SCRIPT DOES
    Sweeps context length at a fixed geometry for all three SHIPPED allocations, derived from their
    own fractions rather than carried as absolute lane counts -- which is the fix `layout.py`
    documents and the reason a config can otherwise flatter itself at a wider head. It reports r1 and
    the decoder's share of the bytes alongside the ratio, so the n-dependence is visible rather than
    asserted.

    `--real` captures actual K and V from cached weights (`k_proj`/`v_proj` hooks, one forward at the
    longest length, prefix-sliced for the rest) and its table is the one to quote. Without it the
    store is synthetic, the fitted `r1` runs 2-3x the real value, and every ratio is a lower bound --
    see `synth_kv`. Random token ids either way, which matches the provenance of the 3.514x fixture.

ASCII only. Not a test; it is not collected by pytest.
"""
from __future__ import annotations

import pathlib
import sys

import torch

_HERE = pathlib.Path(__file__).resolve()
for _c in (_HERE.parents[1] / "codec", _HERE.parent, _HERE.parent / "integration", pathlib.Path("/opt/ttkv/integration")):
    if _c.is_dir():
        sys.path.insert(0, str(_c))

import ttkv_packed_cache as pc  # noqa: E402
from ttkv_codec import CONFIGS  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Why bf16 for the capture. In float32, SDPA cannot use a flash kernel, so it falls back to math and
# materializes [H, n, n] scores -- 24 GiB at n=16384, which OOMs a 16 GiB card (measured). bf16 selects
# flash and needs none of it. The precision cost is nil for a RATIO: bf16 rounding is ~2e-3 relative,
# an energy share of ~4e-6, against a truncation threshold of 0.2**2 = 0.04. It cannot move where the
# cumulative-energy curve crosses the threshold.
CAPTURE_DTYPE = torch.bfloat16

# (label, model_id or None for synthetic, H, d, NL, rope_theta). The first two are the two geometries
# that matter -- Llama-3.1-8B is where the campaign numbers were made, Qwen2-0.5B is what the e2e
# fixture uses. The last two are the SAME geometries with real cached weights, and they are the ones
# to quote. `meta-llama/Llama-3.2-3B` is the cheap stand-in for Llama-3.1-8B: same H=8, d=128 and so
# the same KV width of 1024, at 6 GB instead of 15 GB.
GEOMS = (
    ("Llama-3.1-8B  (campaign geometry, synthetic)", None, 8, 128, 32, 5.0e5),
    ("Qwen2-0.5B    (the e2e fixture, synthetic)", None, 2, 64, 24, 1.0e6),
    ("Llama-3.2-3B  (real weights, KV width 1024)", "meta-llama/Llama-3.2-3B", 8, 128, 28, 5.0e5),
    ("Qwen2-0.5B    (real weights, the fixture)", "Qwen/Qwen2-0.5B", 2, 64, 24, 1.0e6),
)
RANK = 320                        # synthetic token subspace
NOISE = 0.004                     # noise floor on top of it
NS = (48, 512, 4096, 16384)
ALLOCS = ("safe", "tight", "campaign4x")
FIXTURE = (0.5, 4)                # the e2e fixture's own allocation: lanes 0.5*d, 4-bit codes


def rope_fn_from(theta, d):
    """Frequencies only. The real Llama ramp lives in `inv_freq`; for a RATIO sweep the ramp's exact
    shape matters less than that a rotation with the right dispersion is applied, and a synthetic
    ramp of the same theta dispenses with needing a model."""
    inv = 1.0 / (float(theta) ** (torch.arange(0, d, 2, dtype=torch.float32) / d))

    def fn(positions):
        f = torch.outer(positions.float().cpu(), inv)
        e = torch.cat([f, f], dim=-1)
        return e.cos().to(positions.device), e.sin().to(positions.device)

    return fn


def cfg_for(name, d):
    """A SHIPPED allocation, or the e2e fixture's, re-expressed by FRACTION for this head dim.

    `res_frac` is already a fraction. `k_lanes` is an ABSOLUTE count in the codec (81 / 45 / 64 at
    d=128), so it is divided by 128 here -- carrying 81 lanes into a 64-wide row is not a narrower
    config, it is a nonexistent one.
    """
    if name == "e2e-fixture":
        lanes, bits = FIXTURE
        return dict(tols=0.20, qbits=8, res_frac=0.5, res_bits=bits,
                    k_lanes=max(1, round(d * lanes)), vbits=bits, campaign=0.0)
    c = CONFIGS[name]
    return dict(tols=c["tols"], qbits=c["qbits"], res_frac=c["res_frac"], res_bits=c["res_bits"],
                k_lanes=max(1, round(c["k_lanes"] * d / 128)), vbits=c["vbits"],
                campaign=c["campaign"])


def synth_kv(H, d, n, seed=0):
    """A synthetic store, used only when the real one is not requested.

    MEASURED CAVEAT, and the reason `--real` exists. Random-Gaussian tokens have no low intrinsic
    dimension, so the fitted `r1` comes out far above what real weights give at the same geometry and
    length -- 277 pre / 838 post here, against 42 / 150 for the same arm on Llama-3.2-3B's real K,
    and against 21-85 pre / 129-314 post recorded in `ROPE_DISPERSION.md`. A larger `r1` inflates the
    coefficient row and depresses every ratio, so EVERY number in the synthetic table is a LOWER
    BOUND. That is why the first run of this script read 1.98x where the real fixture reads 3.514x at
    the same geometry, the same length and the same allocation. Use it for the shape of the
    n-dependence; do not quote it.
    """
    torch.manual_seed(seed)
    K = ((torch.randn(n, RANK) * 0.2) @ (torch.randn(RANK, d * H) * 0.2)
         ).reshape(n, H, d).permute(1, 0, 2).contiguous()
    K = K + NOISE * torch.randn(H, n, d)
    V = torch.randn(H, n, d) * 0.3 + 0.1 * torch.sign(torch.randn(H, n, d))
    return K, V


def _rope_table(model, maxn, d, theta):
    """`(cos, sin)` for positions `0..maxn-1` on CPU, from the model's OWN rotary module.

    Read off the live module rather than rebuilt from `config.rope_theta`, because the llama3 ramp is
    not reconstructible from theta and a wrong frequency set is still orthogonal and position
    dependent -- so it fails SILENTLY. A precomputed table also sidesteps a device mismatch between a
    CPU-side store and a GPU-side rotary module.

    Returns a third element saying whether the ramp actually differs from a plain-theta one. For a
    llama3 model it MUST differ (the low frequencies are scaled); if it does not, the table was built
    the wrong way and every pre-rotary number below is suspect. That check is printed, not asserted,
    so a non-llama model does not fail the run.
    """
    rot = next((m for nm, m in model.named_modules() if "rotary" in nm.lower()), None)
    if rot is None:
        raise RuntimeError("no rotary module in the model")
    pos = torch.arange(maxn, device=DEV).view(1, -1)

    cos = sin = inv = None
    for dt in (CAPTURE_DTYPE, torch.float32):
        try:
            dummy = torch.zeros(1, maxn, d, device=DEV, dtype=dt)
            c, s = rot(dummy, pos)[:2]
            cos, sin = c[0].float().cpu(), s[0].float().cpu()
            break
        except Exception:                                                        # noqa: BLE001
            continue
    if cos is None:
        inv = rot.inv_freq.to(DEV).float()
        f = torch.outer(pos[0].float(), inv)
        e = torch.cat([f, f], dim=-1)
        cos, sin = e.cos().cpu(), e.sin().cpu()
    if inv is None:
        inv = getattr(rot, "inv_freq", None)
        if inv is not None:
            inv = inv.to(DEV).float()

    plain = 1.0 / (float(theta) ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    if inv is None:
        note = "unverified (no inv_freq buffer to compare against)"
    else:
        # ELEMENT-WISE relative deviation, not absolute-over-max. The llama3 scaling only touches a
        # narrow mid-band of frequencies, and those have tiny absolute values, so normalising by the
        # largest entry makes a real custom ramp look plain (it read 9.85e-04 that way, which is 21%
        # of the local value). Qwen2 has no rope_scaling, so 0.0 is the correct reading there.
        v = inv.cpu()[: len(plain)]
        rel = float(((v - plain).abs() / plain).max())
        note = (f"max relative deviation from a plain theta={theta:g} ramp = {rel:.3f} "
                f"({'custom/llama3 ramp, as expected' if rel > 1e-3 else 'plain ramp, as expected'})")
    return cos, sin, note


def capture_real(model_id, maxn, theta, seed=0):
    """Real PRE-ROPE K and V off every layer's `k_proj`/`v_proj`, plus the model's rope table.

    Hooks rather than a cache, because a capture is config-independent: ONE forward yields the K/V
    that every allocation and both storage modes are then measured on, which is what makes the columns
    comparable. `k_proj` is a bare linear layer, so its output is exactly what the container needs
    before it un-rotates; `to_postrope` applies the model's own frequencies afterwards.

    One forward at the longest length, then prefix-slices for the shorter ones. That is exact, not an
    approximation: attention is causal, so token i's K/V cannot depend on tokens after i, and RoPE is
    applied per position, so positions 0..n-1 of the long forward are identical to a short one.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=CAPTURE_DTYPE, local_files_only=True, attn_implementation="sdpa",
    ).to(DEV).eval()
    c = model.config
    H = int(c.num_key_value_heads)
    d = int(getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads)
    atts = [m for m in model.modules() if hasattr(m, "k_proj") and hasattr(m, "v_proj")]

    table = _rope_table(model, maxn, d, theta)
    got: dict = {}

    def _mk(li, kind):
        def _hook(_mod, _inp, out):
            # to CPU inside the hook: at n=16384 the captured tensors are ~1.9 GB in bf16 and the GPU
            # is already holding the weights.
            got[(li, kind)] = out.detach()[0].reshape(maxn, H, d).permute(1, 0, 2).contiguous().cpu()
        return _hook

    handles = []
    for li, at in enumerate(atts):
        handles.append(at.k_proj.register_forward_hook(_mk(li, "k")))
        handles.append(at.v_proj.register_forward_hook(_mk(li, "v")))
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(100, 30000, (1, maxn), generator=g).to(DEV)
    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()
    assert len(got) == 2 * len(atts), f"captured {len(got)} of {2 * len(atts)} projections"
    del model
    torch.cuda.empty_cache()
    kv = [(got[(li, "k")].float(), got[(li, "v")].float()) for li in range(len(atts))]
    del got
    return kv, H, d, table


def to_postrope(kv, rope_fn):
    """The rows a model hands a cache. `storage="pre"` is what makes the container un-rotate them.

    Applied ONCE per capture, so that `storage` is the only variable left between the two arms.
    Feeding the pre-rotary arm already-un-rotated rows un-rotates them twice and makes both arms
    identical -- that is the bug that made the first version of this script report post == pre to the
    digit, and it is why this is a separate, shared step rather than folded into `measure`.
    """
    _, n, d = kv[0][0].shape
    cos, sin = rope_fn(torch.arange(n))
    h = d // 2
    out = []
    for K_pre, V in kv:
        x = K_pre.float()
        out.append((x * cos + torch.cat([-x[..., h:], x[..., :h]], dim=-1) * sin, V.float()))
    return out


def measure(H, d, kv_post, cfg, storage, rope_fn):
    """`storage` varies and NOTHING ELSE. Returns `(ratio, (r1_lo, r1_mean, r1_hi), decoder%, coeff%)`.

    r1 is reported as a spread, not a mean, because real per-layer r1 runs 93-616 on a single prompt.
    That variance is the reason a uniform page cannot hold the store, so hiding it behind one number
    would hide the problem.
    """
    n = kv_post[0][0].shape[1]
    NL = len(kv_post)
    cache = pc.make_cache(cfg, H, NL, fold_T=n, storage=storage,
                          rope_fn=rope_fn if storage == "pre" else None)()
    for li, (K, V) in enumerate(kv_post):
        cache.update(K.unsqueeze(0), V.unsqueeze(0), li)

    packed = sum(l.nbytes() for l in cache.layers)
    dense = NL * H * n * d * 2 * 2
    ks = [l.kstore for l in cache.layers if l.kstore is not None]
    r1s = [k.r1 for k in ks] or [0]
    dec = sum(k.dec_q.numel() + k.dec_scale.numel() * 2 for k in ks)
    coeff = sum(k.coeff_q.numel() + k.coeff_scale.numel() * 2 for k in ks)
    return dense / packed, (min(r1s), sum(r1s) / len(r1s), max(r1s)), dec / packed, coeff / packed


def _geometry_kv(label, mid, H, d, theta, real):
    """`(kv_pre_by_n, rope_fn)` for one row of `GEOMS`, or `None` when the mode does not want it."""
    if (mid is not None) != real:
        return None
    if mid is None:
        K, V = synth_kv(H, d, max(NS))
        kv_by_n = {}
        for n in NS:
            pair = (K[:, :n].contiguous(), V[:, :n].contiguous())
            # NL identical layers, matching the geometry's real layer count. Identical layers make the
            # ratio NL-invariant, but repeating them keeps the printed NL honest and the arms alike.
            kv_by_n[n] = [pair] * NL
        return kv_by_n, rope_fn_from(theta, d), f"synthetic plain ramp (theta={theta:g})"
    kv_all, H2, d2, (cos_t, sin_t, rope_note) = capture_real(mid, max(NS), theta)
    assert (H2, d2) == (H, d), f"{mid} is H={H2} d={d2}, but GEOMS says H={H} d={d}"

    def rf(positions):
        return cos_t[positions], sin_t[positions]

    return ({n: [(K[:, :n].contiguous(), V[:, :n].contiguous()) for K, V in kv_all] for n in NS},
            rf, rope_note)


def main(real=False):
    print(__doc__.split("WHAT THIS SCRIPT DOES")[0].strip())
    print(f"\nArm: {'REAL weights (quotable)' if real else 'synthetic (lower bound; see synth_kv)'}")
    print("`campaign4x` and the e2e fixture are the same allocation -- the identity is asserted below,")
    print("because that is the whole of reason 1. `coeff%` and `decoder%` are the shares of the")
    print("packed bill, and they are what makes a ratio geometry- and length-dependent.\n")

    for label, mid, H, d, NL, theta in GEOMS:
        got = _geometry_kv(label, mid, H, d, theta, real)
        if got is None:
            continue
        kv_by_n, rf, note = got
        fx = cfg_for("e2e-fixture", d)
        c4 = cfg_for("campaign4x", d)
        same = all(fx[k] == c4[k] for k in ("tols", "qbits", "res_frac", "res_bits", "k_lanes", "vbits"))
        print(f"===== {label}: H={H} d={d} NL={len(kv_by_n[max(NS)])}  KV width H*d = {H * d} =====")
        print(f"      rope: {note}")
        print(f"      campaign4x == e2e fixture at this d: {'YES' if same else 'NO'}  {c4}")
        post_by_n = {n: to_postrope(kv_by_n[n], rf) for n in NS}
        for storage in ("post", "pre"):
            print(f"--- storage={storage} ---")
            print(f"{'alloc':<12} {'n':>6} {'r1 lo/mean/hi':>15} {'coeff%':>7} {'decoder%':>9} {'ratio':>8}")
            for name in ALLOCS:
                cfg = cfg_for(name, d)
                for n in NS:
                    r, r1, ds, cs = measure(H, d, post_by_n[n], cfg, storage, rf)
                    print(f"{name:<12} {n:>6} {r1[0]:>5} {r1[1]:>5.0f} {r1[2]:>4} {100 * cs:>6.1f}% "
                          f"{100 * ds:>8.1f}% {r:>7.2f}x", flush=True)
                print()
        print()


if __name__ == "__main__":
    main(real="--real" in sys.argv)
