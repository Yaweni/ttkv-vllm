"""The attention math for the pre-rotary path, in plain torch + Triton. NO vLLM IMPORT.

WHY THIS IS A SEPARATE MODULE FROM `impl.py`
    Two reasons, and the second is the load-bearing one.

    1. `impl.py` is an ADAPTER: it unpacks vLLM's metadata, splits decodes from prefills, and
       writes into the output buffer. That is plumbing whose bugs are shaped like plumbing bugs
       (a wrong slice, a dropped request). This module is the math whose bugs are silent.
    2. The math has to be testable where vLLM is not installable. It is verified against
       `reference.attend_raw_store` -- an independent, obvious, slow implementation of the same
       attention -- which is the only reason to believe the kernels.

THE ONE THING THAT MAKES THIS DIFFERENT FROM EVERY OTHER ATTENTION PATH IN vLLM
    The store holds keys BEFORE rotation. A post-rotary store can hoist the decoder into the
    query, because the decoder is the same for every row:

        post-rotary:  q_post . k_post_i = (q_post @ D^T) . c_i        <- decoder absorbed, valid

    A raw store cannot, because the rotation is a function of the KEY's own position:

        raw:          q_post . k_pre_i  = (R_p q_pre) . k_pre_i       <- R_i sits in between

    so `R_i` cannot leave the token loop. Writing the absorbed form anyway produces finite,
    correctly-shaped, plausible scores -- which is why the source repo's `attend_container`
    REFUSES a pre-rotary store instead of decoding it (`ttkv_attn.py::_refuse_pre_storage`), and
    why `krt` exists at all.
"""
from __future__ import annotations

import torch

from vllm_ttkv.layout import TTKLayout

__all__ = [
    "HOST_READBACKS",
    "byte_rows",
    "cos_sin_from_cache",
    "gather_slots",
    "split_blob",
    "block_slots",
    "block_slots_host",
    "check_row_host",
    "repeat_kv",
    "check_layout_matches_decoder",
    "attend_prefill",
]

# Every place in this file that reads a DEVICE value back on the host, counted rather than timed.
# Each one drains the pipeline, so their number is a first-order cost of the decode path and the
# thing to drive down -- the phase timings in `impl` show where they land, not how many there are.
HOST_READBACKS = [0]


def cos_sin_from_cache(cos_sin_cache: torch.Tensor, positions: torch.Tensor,
                       head_dim: int, rotary_dim: int | None = None,
                       max_position: int | None = None):
    """`(cos, sin)`, each `[n, head_dim]` fp32, in the frame `krt` expects.

    THE CONVENTION IS THE WHOLE POINT OF THIS FUNCTION, so it is spelled out. vLLM's
    `cos_sin_cache` is `[max_position, rotary_dim]` holding `cat([cos, sin], -1)`, i.e. cos in the
    FIRST half and sin in the second, each of width `rotary_dim/2`, indexed by `inv_freq` position.
    vLLM's rope op then applies the `rotate_half` form:

        o_lo = x_lo * cos - x_hi * sin
        o_hi = x_hi * cos + x_lo * sin

    `krt` applies the `(j, j + d/2)` pairing form, which is the SAME function as long as the table
    it is handed is DUPLICATED across the halves:

        y_lo = k_lo * cos[lo] - k_hi * sin[lo]
        y_hi = k_hi * cos[hi] + k_lo * sin[hi]

    with `cos[lo] == cos[hi]`. So the duplication below is not slack in the table -- it is the
    translation between the two spellings, and dropping it produces a rotation by the wrong angle
    for half the row, which is still an orthogonal position-dependent transform and therefore
    still returns well-formed numbers.

    The lookup is by the row's own ABSOLUTE position, which is exactly what makes this the
    pre-rotary path: the store is position-free, so the position is supplied here, at read time.

    WHY THE TABLE IS USED AS THE MODEL BUILT IT, bf16 AND ALL. vLLM casts the table to the model
    dtype at construction (`RotaryEmbedding.__init__`: `cache = cache.to(dtype)`,
    `rotary_embedding/base.py:61`), so on a bf16 model the ANGLES ARE bf16 -- roughly 0.4% off the
    fp32 ones. Rebuilding them in fp32 from `rope_parameters` would look like a precision
    improvement and would be the opposite of one: the query arriving here was rotated by the model
    with THIS table, so a key rotated by a slightly different angle differs from the model's own
    key by `delta * |q||k| * sin(theta)` -- first order in the angle mismatch, and larger than the
    store's own reconstruction error (~3e-4 measured end to end). Rotating q and k with the SAME
    rounded angle leaves their inner product almost untouched, because the error is common to both.
    The `.float()` below is only about kernel dtype; it does not recover digits the table lost.
    """
    rot = cos_sin_cache.shape[-1] if rotary_dim is None else rotary_dim
    if rot != head_dim:
        raise NotImplementedError(
            f"rotary_dim={rot} != head_dim={head_dim} (partial rotary). `krt` splits the row into "
            f"halves at d/2 and pairs (j, j + d/2), which is only the rotation when the rotation "
            f"spans the whole row; a partial-rotary model needs a kernel with an explicit "
            f"rotary_dim, not a different table. Refusing rather than rotating the wrong lanes.")
    idx = positions.to(torch.long)
    rows = cos_sin_cache.shape[0]
    if max_position is not None:
        # HOST-SIDE BOUND, for the caller that knows the largest position it is asking for (`impl`
        # passes `seq_len - 1` for its `arange`). The device check below reads a device value back
        # on the host, which drains the pipeline -- measured at ~0.25 ms, and this function used to
        # run once per layer per step. The bound is the same statement either way: the highest
        # position asked for must be inside the table.
        if max_position >= rows:
            raise IndexError(
                f"position {max_position} is past the model's cos_sin_cache, which holds {rows} "
                f"entries. The table is sized for max_model_len; a store longer than that would "
                f"silently read the wrong angle (or, here, not read at all).")
    else:
        HOST_READBACKS[0] += 1
        if idx.numel() and int(idx.max()) >= rows:
            raise IndexError(
                f"position {int(idx.max())} is past the model's cos_sin_cache, which holds "
                f"{rows} entries. The table is sized for max_model_len; a store longer than that "
                f"would silently read the wrong angle (or, here, not read at all).")
    row = cos_sin_cache[idx].to(torch.float32)
    c, s = row[:, :rot // 2], row[:, rot // 2:]
    return (torch.cat([c, c], dim=-1).contiguous(),
            torch.cat([s, s], dim=-1).contiguous())


def byte_rows(page: torch.Tensor, slot_bytes: int, what: str = "the KV cache"):
    """`[slots, W]` uint8 rows of a paged cache tensor, with `W >= slot_bytes`.

    THE LAST DIMENSION OF A vLLM PAGE IS NOT A BYTE COUNT. The page is allocated in the MODEL dtype
    and its width comes from `state_content_bytes`, which is measured in BYTES -- so on the default
    bf16 serving path the last dim is `slot_bytes / 2` ELEMENTS. Measured on the real run: an
    816-byte blob arrived as a 408-wide bf16 page, and reading `shape[-1]` as bytes said the
    allocation was half the size the layout needed. The payload inside a row is our own packed
    bytes, so the row has to be looked at as bytes; nothing about it is bf16.

    `W > slot_bytes` is accepted rather than refused, because vLLM rounds a page up to whole
    elements and a blob with an odd byte count would otherwise be unservable. Callers take
    `[:, :slot_bytes]` when they want the blob exactly, which is also why this returns the width
    instead of assuming it.

    Shared by the write (`encode.write_slots`) and the read (`gather_slots`): if the two disagreed
    about a row's width they would address different bytes of the same region, and every field after
    the coefficients would be read at the wrong offset without anything raising.
    """
    try:
        u8 = page if page.dtype == torch.uint8 else page.view(torch.uint8)
    except RuntimeError as e:
        raise RuntimeError(
            f"{what} of shape {tuple(page.shape)} and dtype {page.dtype} cannot be viewed as bytes, "
            f"because a dtype view needs the last dimension contiguous. The packed payload is a byte "
            f"string, so a page that cannot be read that way cannot be used. ({e})") from e
    if u8.shape[-1] < slot_bytes:
        raise ValueError(
            f"{what} has a slot width of {u8.shape[-1]} bytes ({page.shape[-1]} {page.dtype} "
            f"elements) but this layout needs {slot_bytes}. The region is sized from "
            f"`state_content_bytes`, so a shortfall means the cache was allocated for a different "
            f"layout -- and a write would land in the next token's row.")
    return u8.reshape(-1, u8.shape[-1])


def gather_slots(cache_layer: torch.Tensor, slots: torch.Tensor,
                 slot_bytes: int | None = None, check: bool = True) -> torch.Tensor:
    """`[n, slot_bytes]` uint8, the rows `slots` names, in the order given.

    `cache_layer` is vLLM's LBNHC page for one layer -- `[num_blocks, 1, block_size, slot_bytes]`
    once `customize_spec` has collapsed `num_head_slots` to 1, but accepted here with the head
    axis already squeezed so a test can hand over either form.

    A gather rather than a paged read, deliberately, and it is a real cost: the kernels index
    token-major (`row = it * SLOT + ...`), which is what makes a row's address independent of the
    sequence length, and that in turn is what makes the layout fold-free. Paging the READS would
    give that up. Note that a request's slots are NOT contiguous in general -- the block table
    interleaves them with other requests' -- so this is an indexed read, not a slice.

    `check=False` skips the device bound check, for a caller that has ALREADY validated the same
    `slots` on the host (`impl._Step` reads the block table back once per step and checks the block
    indices there). The check is not optional in general: `slots.max()` is a device readback, and
    this runs once per layer.
    """
    c = cache_layer
    if c.dim() == 4:
        if c.shape[1] != 1:
            raise ValueError(
                f"expected num_head_slots=1 (the packed slot is per TOKEN, with the head axis "
                f"folded into its content), got a cache branch of shape {tuple(c.shape)}")
        c = c[:, 0]
    if c.dim() != 3:
        raise ValueError(f"expected a [num_blocks, block_size, C] page, got {tuple(c.shape)}")
    # `slot_bytes` is the blob the LAYOUT wants; without it the page's own byte width is used, which
    # is what the tests do when they build a cache as a plain uint8 tensor.
    want = c.shape[-1] if slot_bytes is None else slot_bytes
    flat = byte_rows(c, want, "the cache page for this layer")
    if check:
        HOST_READBACKS[0] += 1
        if slots.numel() and int(slots.max()) >= flat.shape[0]:
            raise IndexError(
                f"slot {int(slots.max())} is outside the cache, which holds {flat.shape[0]} slots")
    return flat[slots, :want].contiguous()


def check_row_host(row: list[int], seq_len: int, block_size: int,
                   n_slots: int | None = None) -> None:
    """The three refusals a block table row has to pass, from a row that is already on the host.

    Split out of `block_slots_host` so that the DEVICE build can use the same checks: the caller has
    the host copy anyway (it needs the row's first block as the request's decoder key), so validating
    from it costs nothing, while rebuilding the slot list on the host costs an O(seq_len) Python loop
    and a second host-to-device copy per step. Same refusals, one place.

    `n_slots` is the cache's own capacity (`num_blocks * block_size`). When given, a block index at
    or beyond it is refused here, which is what lets `gather_slots` run unchecked on the result.
    """
    if seq_len < 0:
        raise ValueError(f"seq_len={seq_len} is negative")
    if seq_len == 0:
        return
    n_blocks = -(-seq_len // block_size)                      # ceil
    if n_blocks > len(row):
        raise IndexError(
            f"seq_len={seq_len} needs {n_blocks} blocks of {block_size} but the block table row "
            f"holds {len(row)}. Reading this would take the request's slots from whatever block was "
            f"there -- another request's data, with no error.")
    used = row[:n_blocks]
    top = max(used)
    if min(used) < 0:
        raise IndexError(
            f"the block table row for this request holds {min(used)} inside its used prefix, which "
            f"is not a block index. vLLM pads a row with a sentinel beyond the request's blocks; a "
            f"negative value INSIDE the used prefix means the request's allocation is not what this "
            f"table says.")
    if n_slots is not None and top * block_size >= n_slots:
        raise IndexError(
            f"block {top} is outside the cache, which holds {n_slots // block_size} blocks of "
            f"{block_size}. A gather from a block past the end reads another request's rows and "
            f"returns finite, well-shaped attention about the wrong tokens.")


def block_slots_host(row: list[int], seq_len: int, block_size: int, device,
                     n_slots: int | None = None) -> torch.Tensor:
    """`block_slots`, from a block table row that is ALREADY on the host.

    Same convention -- slot id is `block * block_size + offset` -- and the same refusals, but
    computed with host arithmetic. This is the version to use when the caller wants the slot ids as
    HOST data; when it wants them on the device and holds the row tensor too, `block_slots` is
    cheaper: everything here beyond the checks is an O(seq_len) `range`/`extend` loop and a
    host-to-device upload of `seq_len` longs, and both are per step
    (`tests/bench_step_plan.py` measures the pair).
    """
    check_row_host(row, seq_len, block_size, n_slots)
    if seq_len == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    n_blocks = -(-seq_len // block_size)                      # ceil
    out = []
    for b in row[:n_blocks]:
        out.extend(range(b * block_size, b * block_size + block_size))
    return torch.tensor(out[:seq_len], dtype=torch.long, device=device)


def repeat_kv(x: torch.Tensor, G: int) -> torch.Tensor:
    """`[T, Hkv, d]` -> `[T, Hkv*G, d]`, the GQA expansion `scaled_dot_product_attention` needs."""
    if G == 1:
        return x
    T, Hkv, d = x.shape
    return x[:, :, None, :].expand(T, Hkv, G, d).reshape(T, Hkv * G, d)


def check_layout_matches_decoder(lay: TTKLayout, decoder_f: torch.Tensor) -> None:
    """The layout's `r1` and the fitted decoder's row count MUST agree, and nothing else checks it.

    `r1` decides `coeff_bytes`, and every field after the coefficient block is found by adding
    widths to it. So a slot written at one rank and read at another does not fail -- it reads the
    coefficient bytes as residual bitmaps and returns numbers of the right shape and the wrong
    value. Measured on real layers, per-layer rank ranges 6..166, so a stale calibration table
    against a freshly fitted decoder is not a corner case; it is the default outcome of getting
    the order of operations wrong.
    """
    got = int(decoder_f.shape[0])
    if lay.r1 != got:
        raise ValueError(
            f"the layout says r1={lay.r1} but the fitted decoder has r1={got} rows. The slot width "
            f"is derived from r1, so these must be the same number: a mismatch shifts every field "
            f"after the coefficients and reads bitmaps as codes. This happens when the cache was "
            f"sized from a calibration table and the decoder was refitted at a different tolerance "
            f"-- pick one rank and use it for both.")


def split_blob(lay: TTKLayout, blob: torch.Tensor):
    """`[n, k + v]` -> `(k_slots [n, k], v_slots [n, v])`, both contiguous.

    K and V share one cache region (see `TTKLayout.v_offset`), so the read path gets them as
    slices of the same gathered blob. The `.contiguous()` calls are LOAD-BEARING rather than
    tidiness: every kernel takes its row stride as a compile-time constant equal to the slot
    width, and a non-contiguous slice has a stride of `k + v`. Handing that over would make the
    kernel walk off the end of each row and read the next token's bytes -- same shapes, wrong
    values, no error. Measured cost of the copy: ~70 us per decode step at n=16k over 28 layers,
    against a 22.9 ms step.
    """
    k = lay.k_slot_bytes
    if blob.shape[-1] != k + lay.v_slot_bytes:
        raise ValueError(
            f"a token's blob is {blob.shape[-1]} bytes but the layout asks for "
            f"{k} + {lay.v_slot_bytes}. The region is sized from `compressed_bytes_per_token`, so "
            f"a mismatch means the cache was allocated with a different layout than the one this "
            f"layer is being read with.")
    return blob[:, :k].contiguous(), blob[:, k:].contiguous()


def block_slots(block_row: torch.Tensor, seq_len: int, block_size: int,
                check: bool = True) -> torch.Tensor:
    """The flat slot ids for a request's first `seq_len` tokens, in position order.

    `block_row` is `block_table[req]`, `[max_num_blocks]`, and a slot id is `block * block_size +
    offset` -- vLLM's own flat indexing, which is what `slot_mapping` carries too, so the write
    path and the read path agree by construction rather than by two matching conventions.

    A request's slots are NOT contiguous: the block table interleaves them with every other live
    request's. That is exactly why the store is read by a gather instead of a slice, and why the
    store's row addressing is token-major -- an address that depended on the sequence length would
    have to be rebuilt on every append, and could not be gathered at all.

    `check=False` skips the row-length bound, and it is not a way to opt out of being safe: it is for
    a caller that has ALREADY run `check_row_host` on this row's host copy, where the bound was the
    same comparison. The bound needs `idx.max()` on the host, i.e. a device readback, and this
    function exists to avoid those -- `impl._Step` reads the row back once per step for the decoder
    key and validates there, so re-deriving the same bound per row would be a readback per request
    bought with nothing.
    """
    if seq_len < 0:
        raise ValueError(f"seq_len={seq_len} is negative")
    if seq_len == 0:
        return block_row.new_zeros(0, dtype=torch.long)
    dev = block_row.device
    pos = torch.arange(seq_len, device=dev)
    idx = pos // block_size
    if check and int(idx.max()) >= block_row.shape[0]:
        raise IndexError(
            f"seq_len={seq_len} needs {int(idx.max()) + 1} blocks of {block_size} but the block "
            f"table row holds {block_row.shape[0]}. Reading this would take the request's slots "
            f"from whatever block was there -- another request's data, with no error.")
    return (block_row[idx].to(torch.long) * block_size + (pos % block_size))


def check_row_counts(k_rows: int, v_rows: int, cos_rows: int, seq_len: int) -> None:
    """Every row population must agree, and this is the invariant the dense TAIL used to carry.

    The original design kept a dense tail of un-folded rows and had to reconcile `store + tail`
    against the request's length. The tail is gone (measured: encoding one token against the
    frozen decoder costs 0.009 ms, so there is nothing to batch up, and folding existed only
    because appending to the torch container re-quantised the whole store). What is NOT gone is
    the need to notice when the rows do not add up: a store that is one row short drops a key,
    which changes the softmax denominator and every weight in it, silently.

    The current token is INCLUDED: the store write happens earlier in the same step, before
    attention reads it.
    """
    if not (k_rows == v_rows == cos_rows == seq_len):
        raise RuntimeError(
            f"row counts disagree with the request: K={k_rows}, V={v_rows}, angles={cos_rows}, "
            f"seq_len={seq_len}. These are four views of the same token range, so a mismatch means "
            f"a key is missing from either the attention or the softmax normaliser -- which does "
            f"not raise anywhere downstream, it just changes the answer.")


def attend_prefill(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                   scale: float) -> torch.Tensor:
    """First-chunk prefill: plain causal SDPA over the dense q/k/v the model handed us.

    `query` `[T, HQ, d]`, `key`/`value` `[T, Hkv, d]` -> `[T, HQ, d]`.

    THERE IS NOTHING TO SPECIAL-CASE HERE, and that is the point. On a first chunk every key the
    request needs is in this very batch, and the model has already rotated it -- so the cheapest
    correct thing is stock attention, exactly as TurboQuant does it ("Prefill: standard SDPA on
    uncompressed K/V, then quantize"). The store was written moments ago and it is raw, so reading
    it here would mean rotating it back to rotate it forward.

    CALLER CONTRACT, and it is a correctness contract rather than a performance one: this function
    only sees the CURRENT batch of keys. A continuation chunk (`seq_len > q_len`, i.e. the request
    has prior context) must therefore NOT be routed here, because that would attend over the chunk
    alone and drop every earlier key -- a wrong answer produced without an error. `impl._prefill`
    enforces that before calling.
    """
    G = query.shape[1] // key.shape[1]
    if query.shape[1] != key.shape[1] * G:
        raise ValueError(
            f"num_heads={query.shape[1]} is not a multiple of num_kv_heads={key.shape[1]}")
    q = query.transpose(0, 1).unsqueeze(0)                # [1, HQ, T, d]
    k = repeat_kv(key, G).transpose(0, 1).unsqueeze(0)
    v = repeat_kv(value, G).transpose(0, 1).unsqueeze(0)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    return out.squeeze(0).transpose(0, 1).contiguous()    # [T, HQ, d]
