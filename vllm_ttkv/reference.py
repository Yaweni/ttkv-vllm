"""Reference pre-rotary decode, in plain torch, against the packed slots.

This is the **correctness oracle for `csrc/ttkv_krt`**. It is deliberately not fast: it builds the
dense rows it is allowed to build because nothing here is the serving path. Its job is to be
obviously right, and to be the thing a reviewer can check without reading CUDA.

WHAT IT DOES, AND WHY EACH STEP IS THERE
----------------------------------------
    reconstruct   K_pre = coefficients @ decoder                     (raw, position-free)
    patch         K_pre += residual codes * per-row ruler            (the lanes the fit missed)
    rotate        K_post = RoPE(K_pre, THAT ROW's own position)      <- the whole point
    contract      scores = q_post . K_post
    softmax, then V summed with the same weights

The rotation is the step that makes this different from every other attention path in vLLM. A
post-rotary store can absorb the decoder into the query, because the decoder is the same for every
row. A raw store cannot: the rotation is a function of the KEY's position, so `R_i` sits between
the reconstruction and the contraction and cannot be hoisted out of the token loop. Writing
`q_post . k_pre` instead -- i.e. skipping the rotation -- yields finite, correctly-shaped,
plausible numbers, which is why the source repo's `attend_container` refuses a pre-rotary store
rather than decoding it (`ttkv_attn.py::_refuse_pre_storage`).

THE ORACLE IS A DIFFERENT INDEX ORDER ON PURPOSE
------------------------------------------------
`attend_raw_store` works in `[token, head, dim]` with the softmax on dim 0; `oracle_attend` works in
`[head, group, dim]` with the softmax on the last dim, expanding GQA explicitly. They are the same
mathematics written two ways, so agreement is a real check of the (head, token) ordering and of the
rotation broadcast rather than a tautology. That matters because the failure mode this project keeps
meeting is a silent permutation -- it reads as an *anti-correlated* metric, not as noise.
"""
from __future__ import annotations

import torch

try:
    import ttkv_packed_cache as pc
except ImportError as exc:                                    # pragma: no cover
    raise ImportError(
        "vllm_ttkv.reference needs the TT-KV codec (`ttkv_packed_cache`). It is the single "
        "definition of the bit-packing and of `PackedRows`, so this module reuses it rather "
        "than reimplementing the format: put the integration directory on sys.path, or vendor "
        "the codec into this package.") from exc

from vllm_ttkv.layout import unpack_k_slots, unpack_v_slots


def rope_tables(positions: torch.Tensor, d: int, theta: float = 10000.0):
    """`(cos, sin)`, each `[len(positions), d]`, in the `(j, j + d/2)` pairing HF uses.

    A TEST STAND-IN for the model's table, and it must not be used in a real run: Llama-3.1's
    `inv_freq` carries a llama3 frequency ramp that `config.rope_theta` does not reproduce, and a
    wrong frequency set is still an orthogonal, position-dependent transform -- so it fails
    silently rather than loudly. In a vLLM run the tables come from the model's own rotary module
    (the way HPC passes `cos_sin_cache=self.rotary_emb.cos_sin_cache`).
    """
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32,
                                       device=positions.device) / d))
    freqs = torch.outer(positions.float(), inv)               # [n, d/2]
    emb = torch.cat([freqs, freqs], dim=-1)                   # [n, d]
    return emb.cos(), emb.sin()


def dense_k_from_slots(lay, k_slots: torch.Tensor, decoder_f: torch.Tensor):
    """`[H, n, d]`, **RAW**: coefficients @ decoder, plus the residual lanes the fit missed.

    No rotation is applied. The caller decides each stored row's position, and that decision is the
    entire difference between this and a post-rotary decode.
    """
    n, H, d = k_slots.shape[0], lay.H, lay.d
    coeff_q, coeff_scale, (bm, pk, sc) = unpack_k_slots(lay, k_slots)
    kflat = (coeff_q.float() * coeff_scale.float().unsqueeze(1)) @ decoder_f   # [n, H*d]

    resid = pc.PackedRows(H * n, d, lay.res_k, lay.res_bits)
    resid.bitmap, resid.packed, resid.scale = bm, pk, sc
    return (kflat.reshape(n, H, d).permute(1, 0, 2)
            + resid.materialise(torch.float32).reshape(H, n, d)).contiguous()


def dense_v_from_slots(lay, v_slots: torch.Tensor):
    """`[H, n, d]`. V is not rotated and has no basis, so a slot is the whole story."""
    n, H, d = v_slots.shape[0], lay.H, lay.d
    vbm, vpk, vsc = unpack_v_slots(lay, v_slots)
    vrows = pc.PackedRows(H * n, d, lay.v_lanes, lay.v_bits)
    vrows.bitmap, vrows.packed, vrows.scale = vbm, vpk, vsc
    return vrows.materialise(torch.float32).reshape(H, n, d).contiguous()


def dense_from_slots(lay, k_slots, v_slots, decoder_f):
    """`(K_pre [H, n, d], V [H, n, d])` -- what the slots hold, in the codec's own order."""
    return (dense_k_from_slots(lay, k_slots, decoder_f),
            dense_v_from_slots(lay, v_slots))


def raw_scores(lay, k_slots, decoder_f, query, cos, sin, scale=None):
    """Scores `[n, HQ]` before softmax, with each stored row rotated by ITS OWN position.

    Exposed separately, and not only for convenience: the scores are the direct observable of
    whether the per-row rotation happened. The softmax OUTPUT can be surprisingly insensitive to it
    -- with flat weights, a perturbation in the scores barely moves a weighted average of zero-mean
    rows, which is exactly what a first attempt at a negative-control test ran into. This is also
    the quantity `ttkv_krt` produces, so the kernel port compares against it directly.
    """
    K_pre = dense_k_from_slots(lay, k_slots, decoder_f)                    # [H, n, d]
    HQ, d = query.shape
    scale = d ** -0.5 if scale is None else scale
    # per-ROW position: cos is [n, d], inserted at dim 1 so it spans the head axis
    K_post = pc.rope_apply(K_pre.permute(1, 0, 2), cos.unsqueeze(1), sin.unsqueeze(1))  # [n,H,d]
    p_all = torch.arange(HQ, device=query.device) // (HQ // lay.H)
    return torch.einsum("npd,pd->np", K_post[:, p_all, :], query) * scale


def attend_raw_store(lay, k_slots, v_slots, decoder_f, query, cos, sin, scale=None):
    """Reference decode. `query` is `[HQ, d]` and POST-rotary. Returns `[HQ, d]`.

    `cos`/`sin` are `[n, d]`, one angle pair per STORED row. Rotating by anything other than that
    row's own position is the failure this function exists to make visible; `tests/test_reference.py`
    asserts both that a wrong position changes the answer and that omitting the rotation changes it
    a lot.
    """
    HQ = query.shape[0]
    H = lay.H
    s = raw_scores(lay, k_slots, decoder_f, query, cos, sin, scale)        # [n, HQ]
    p_attn = torch.softmax(s, dim=0)
    V = dense_v_from_slots(lay, v_slots)
    p_all = torch.arange(HQ, device=query.device) // (HQ // H)
    return torch.einsum("np,npd->pd", p_attn, V.permute(1, 0, 2)[:, p_all, :])


def oracle_attend(lay, k_slots, v_slots, decoder_f, query, cos, sin, scale=None):
    """The same attention, written the other way round: `[head, group, dim]`, softmax on the last
    dim, GQA expanded explicitly. See the module docstring on why this is a real oracle."""
    K_pre, V = dense_from_slots(lay, k_slots, v_slots, decoder_f)
    H, n, d = K_pre.shape
    HQ = query.shape[0]
    G = HQ // H
    scale = d ** -0.5 if scale is None else scale

    # cos is [n, d], inserted at dim 0 so it spans the head axis -- deliberately a different
    # broadcast from `attend_raw_store`, so agreement tests the indexing rather than the call
    K_post = pc.rope_apply(K_pre, cos.unsqueeze(0), sin.unsqueeze(0))       # [H, n, d]
    q = query.reshape(H, G, d)
    scores = torch.einsum("hgd,hnd->hgn", q, K_post) * scale
    p_attn = torch.softmax(scores, dim=-1)
    return torch.einsum("hgn,hnd->hgd", p_attn, V).reshape(HQ, d)
