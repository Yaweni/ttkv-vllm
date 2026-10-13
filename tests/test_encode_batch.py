"""Does the BATCHED projection write the same bytes as the per-request one?

`encode_frozen_batch` exists to cut launches, not to change the format, so the only acceptable
result is byte equality with `encode_frozen` on the same inputs. That is a stronger gate than it
looks: the two paths reach the same coefficients through different dense kernels (`bmm` on a stacked
`[G, n, H*d]` against a `@` on one request at a time), and the coefficient row is QUANTIZED. A
one-ulp difference in the last mantissa bit of a coefficient does not stay a one-ulp difference --
it moves a rounded code by one step, which changes `packed` bytes, which changes the reconstruction.
That is the failure this file is here to catch, and it cannot be caught by an accuracy tolerance.

WHY THE TEST BUILDS REAL FITS RATHER THAN RANDOM DECODERS. A decoder has a structure -- it is a
rank budget's orthonormal basis with a `pinv` derived from it -- and random matrices would let the
batched path agree for the wrong reason (or disagree for one). The fixtures come through
`fit_into_budget`, the same call the first chunk makes.

The guards are tested as hard as the arithmetic, for the usual reason: a batch whose members have
different ranks, or a decoder list that does not match the stack, is silently broadcastable in
torch and produces plausible garbage.

Skips cleanly without a CUDA device, like the rest of the kernel-dependent suite.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

_HERE = pathlib.Path(__file__).resolve()
for _cand in (_HERE.parents[1] / "codec", _HERE.parents[1], _HERE.parents[2], _HERE.parents[2] / "integration",
              pathlib.Path("/opt/ttkv/integration")):
    if _cand.is_dir():
        sys.path.insert(0, str(_cand))

from ttkv_codec import CONFIGS  # noqa: E402
from vllm_ttkv import encode  # noqa: E402
from vllm_ttkv.layout import layout_for_config, pack_k_slots  # noqa: E402

H, D = 8, 128
HQ = 32                       # G = 4
N = 16
R1 = 32
DEV = "cuda"


def _need_cuda():
    if not torch.cuda.is_available():
        print("  SKIPPED: no CUDA device")
        return False
    return True


def _k(g: int, n: int = N, seed: int = 0) -> torch.Tensor:
    """A low-rank-plus-noise key block, so the residual path is live and the quantizer has range."""
    torch.manual_seed(seed + 100 * g)
    rank = 4
    K = ((torch.randn(n, rank, device=DEV) * 0.2)
         @ (torch.randn(rank, H * D, device=DEV) * 0.2)).reshape(n, H, D).permute(1, 0, 2)
    return (K + 0.02 * torch.randn(H, n, D, device=DEV)).contiguous()


def _fixture(cfgname: str = "safe", n: int = N, g: int = 4, r1: int = R1, seed: int = 0):
    cfg = CONFIGS[cfgname]
    lay = layout_for_config(cfg, r1=r1, H=H, d=D)
    Ks = [_k(i, n, seed) for i in range(g)]
    kdecs = [encode.fit_into_budget(Ks[i], lay, cfg).decoder for i in range(g)]
    return lay, Ks, kdecs


def test_the_batch_reproduces_the_per_request_bytes():
    """THE GATE. Coefficients, rulers and residual rows all have to match exactly, and then the
    packed slots built from them have to match too -- the last one is what the store actually holds,
    so agreeing on the fields and disagreeing on the bytes would still be a broken cache."""
    if not _need_cuda():
        return
    for cfgname in ("safe", "tight", "campaign4x"):
        lay, Ks, kdecs = _fixture(cfgname)
        one = [encode.encode_frozen(K, kd, lay) for K, kd in zip(Ks, kdecs)]
        many = encode.encode_frozen_batch(torch.stack(Ks), kdecs, lay)
        assert len(many) == len(one)
        for i, (a, b) in enumerate(zip(one, many)):
            assert torch.equal(a.coeff_q, b.coeff_q), f"{cfgname} req {i}: coeff_q differs"
            assert torch.equal(a.coeff_scale, b.coeff_scale), \
                f"{cfgname} req {i}: coeff_scale differs"
            assert torch.equal(a.resid_src, b.resid_src), f"{cfgname} req {i}: resid_src differs"
        slots_one = [pack_k_slots(lay, f.coeff_q, f.coeff_scale, f.resid) for f in one]
        slots_many = [pack_k_slots(lay, f.coeff_q, f.coeff_scale, f.resid) for f in many]
        for i, (a, b) in enumerate(zip(slots_one, slots_many)):
            assert torch.equal(a, b), f"{cfgname} req {i}: packed slots differ"
        print(f"  {cfgname:11s} r1={lay.r1:<4d} res_k={lay.res_k:<3d} "
              f"G={len(one)} n={N}  bytes identical on {tuple(slots_one[0].shape)}")


def test_a_batch_of_one_matches_the_per_request_path():
    """The degenerate group. The caller only batches when there is more than one member, but the
    kernel has to be right at G=1 too or the two paths would disagree in a way the caller's
    branch would hide."""
    if not _need_cuda():
        return
    lay, Ks, kdecs = _fixture()
    one = encode.encode_frozen(Ks[0], kdecs[0], lay)
    many = encode.encode_frozen_batch(Ks[0].unsqueeze(0), kdecs[:1], lay)[0]
    assert torch.equal(one.coeff_q, many.coeff_q)
    assert torch.equal(one.resid_src, many.resid_src)


def test_the_batch_does_not_share_a_basis_between_requests():
    """The property the whole design refuses to give up: each request keeps its OWN decoder. If the
    batch quietly used member 0's decoder for everybody, the coefficients would still be the right
    shape and finite -- and request 1's reconstruction would be wrong by that request's own basis
    difference. Perturbing one member's decoder must move only that member's coefficients."""
    if not _need_cuda():
        return
    lay, Ks, kdecs = _fixture()
    base = encode.encode_frozen_batch(torch.stack(Ks), kdecs, lay)
    moved = list(kdecs)
    moved[2] = encode.KDecoder(kdecs[2].decoder_f * 1.0 + 0.05)
    after = encode.encode_frozen_batch(torch.stack(Ks), moved, lay)
    for i in (0, 1, 3):
        assert torch.equal(base[i].coeff_q, after[i].coeff_q), \
            f"request {i} changed when only request 2's decoder moved"
    delta = (base[2].coeff_q.float() - after[2].coeff_q.float()).abs().max().item()
    print(f"  perturbing member 2 moved only member 2, by {delta} code steps")
    assert delta > 0, "perturbing a decoder changed nothing, so the test is not observing it"


@pytest.mark.parametrize("bad", ["shape", "count", "rank"])
def test_the_batch_refuses_what_it_cannot_express(bad):
    if not _need_cuda():
        return
    lay, Ks, kdecs = _fixture()
    stack = torch.stack(Ks)
    if bad == "shape":
        # Drop a HEAD, not a token: `n` is free here (the group is defined by it), H is not.
        arg, decs, needle = stack[:, :-1], kdecs, "but the layout is for"
    elif bad == "count":
        arg, decs, needle = stack, kdecs[:2], "decoders were given"
    else:
        # Four decoders, the last fitted at TWICE the layout's rank. The count has to match or the
        # count guard fires first and this case tests the wrong refusal.
        other = layout_for_config(CONFIGS["safe"], r1=R1 * 2, H=H, d=D)
        arg = stack
        decs = kdecs[:3] + [encode.fit_into_budget(Ks[3], other, CONFIGS["safe"]).decoder]
        needle = "cannot be read at another"
    with pytest.raises(ValueError) as e:
        encode.encode_frozen_batch(arg, decs, lay)
    assert needle in str(e.value), str(e.value)
