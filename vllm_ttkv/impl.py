"""`AttentionImpl` for TT-KV: attention read from the packed store, over RAW K.

THE ONE THING TO UNDERSTAND HERE
--------------------------------
Q-absorption -- the trick that makes the post-rotary path cheap, folding the tensor-train
decoder into the query so the decoder is never expanded per head -- is **unavailable** on this
path. It requires each stored row to already carry its own position, and raw rows do not:

    post-rotary store :  q_post . k_post_i  = (g @ QT) . c_i      <- absorbed, valid
    raw store         :  q_post . k_pre_i   = ?                   <- R_i intervenes

`q_post . k_pre_i` is `(R_p q_pre) . k_pre_i`, and `R_i` -- the rotation at the KEY's position
-- sits between the reconstruction and the contraction, so it cannot be hoisted out of the token
loop. Writing it anyway is the specific mistake the source repo refuses to make: the wrong
contraction is finite, correctly shaped and plausible. See `ttkv_attn.py::_refuse_pre_storage`.

So the kernel surface for this port is `krt` + `v_out`. `qt` and `resid_add` belong to the
absorbed path and are dead here.

WHAT THIS CLASS IS, AND WHAT IT IS NOT
    It owns the WRITE and the READ. The write is here because this is the only place in the step
    holding all three of the raw K (stashed by `patch.py`'s rotary wrapper), vLLM's slot mapping,
    and the per-request split; the slot arithmetic itself lives in `vllm_ttkv/encode.py`. The read
    is here as an ADAPTER: it unpacks vLLM's metadata, splits decodes from prefills, resolves each
    request's slots through its block table, and writes into the output buffer. The read MATH lives
    in `vllm_ttkv/decode.py` (plus the two kernels), which is importable and testable without vLLM --
    deliberately, because the bugs in the math are silent while the bugs in this file are not.

THE TAIL IS GONE, AND `_tail_scores` IS NOW A REFUSAL
    The design this file was sketched against kept a dense tail of un-folded rows and folded them
    every `fold_T` tokens, because appending one row to the torch container re-quantised and
    re-packed the whole store: ~20 ms/layer/token, 86% of the decode step. That cost does not
    exist here -- a row goes to its slot through `slot_mapping`, and encoding a single token
    against the frozen decoder measures 0.009 ms (0.25 ms over 28 layers, against a 22.9 ms
    decode). So there is nothing to batch up, and the tail is empty by construction.

    That is worth more than it sounds: the dense tail was a SECOND row population with a different
    rotation state (already post-RoPE, must never be rotated again), reaching the kernel through a
    ragged per-request length. Deleting it deletes the varlen ragged-tail path entirely. What
    replaces it is `decode.check_row_counts`, which asserts that the four views of a request's
    token range -- K rows, V rows, angle rows, `seq_len` -- all agree. That invariant is the real
    content of the old tail bookkeeping, and unlike the tail it cannot silently be off by one.
"""
from __future__ import annotations

import logging
import os
import time

import torch
from vllm.v1.attention.backend import AttentionImpl

from vllm_ttkv import decode as dc
from vllm_ttkv import prof
from vllm_ttkv.krt_triton import krt_from_slots, krt_multi_from_slots
from vllm_ttkv.rope_store import (calibration_info, decoder_for_layer, decoder_for_request,
                                  ensure_store, layout_for_layer, register_decoder, request_key,
                                  request_key_host, store_for_layer)
from vllm_ttkv.v_out_triton import v_out_from_slots, v_out_multi_from_slots

try:                                                      # present in backend.py, but not load-bearing
    from vllm.v1.attention.backend import AttentionType
except ImportError:                                       # pragma: no cover
    AttentionType = None                                  # type: ignore[assignment]

logger = logging.getLogger("vllm.ttkv")

# One line per layer per step, off by default. Bring-up on a new model needs it: the per-layer
# decisions (which requests decode, what the batch split was, whether a store existed) are all
# visible only from inside `forward`, and the failure they explain is usually three layers later.
_TRACE = bool(os.environ.get("VLLM_TTKV_TRACE"))

# Per-phase attribution lives in `prof` so that `encode.py` can add its own phases: the write is
# `impl` -> `encode` -> the codec, and a lump called `write.project` is not something anyone can
# optimise -- what is actionable is which STEP of the encode the launches are in.
_PROFILE = prof.ENABLED
_phase = prof.phase
_sync = prof.sync
_COUNT = prof.COUNT


class _Step:
    """Everything about a step that does NOT depend on the layer, computed and validated ONCE.

    WHY THIS EXISTS, and it is a measured reason rather than tidiness. vLLM hands one metadata object
    to a group of layers (measured: 448 layer-calls over 112 metadata objects -- 4 layers per object,
    7 objects per step), so any per-layer code that reads a DEVICE value back on the host does it 28
    times per step. Each readback drains the pipeline, and the first profile of this run put the cost
    at ~6 readbacks per layer-call, i.e. ~150 per step:

        ms per layer-call    phase                 readbacks it paid for
        2.00                 write.pack_and_write  the slot ids, twice
        1.34                 write.project
        1.08                 decode_total          gather + the angle table
        0.66                 plan                  cu_seqlens + seq_lens
        0.24                 block_slots           its own idx.max()

    None of those values is layer-dependent. The batch split, each request's slot ids, the angle rows
    and the slot mapping the write uses are all properties of the BATCH. So they are computed here,
    once, and the per-layer code reads them off this object.

    THE VALIDATION MOVED, IT DID NOT GO AWAY, and it is stronger where it landed. Reading the block
    table and the slot mapping back on the host is what makes their bounds checkable with no further
    readback at all -- the check becomes arithmetic on host values -- and those are the same bounds
    the kernels would otherwise have to be trusted with: a slot id outside the cache addresses
    another request's row, and PAD_SLOT_ID means a token with no slot at all. `gather_slots` and
    `write_slots` still refuse by default; they take `check=False` only from here.

    THE SERIAL MAKES THE CACHE SAFE. A bundle is reused only while `md.ttkv_serial` matches the one
    it was built for, so it cannot outlive its step even if vLLM recycles metadata objects.

    WHAT MOVED HERE LAST, AND WHY IT IS A WAIT RATHER THAN WORK. `tests/bench_step_plan.py` measures
    every call in this build at two queue depths, and the result is unambiguous: each of these
    readbacks costs 24-63 us with the queue empty and the whole queue depth (11-13 ms, measured) with
    it full, while reading ONE int costs exactly what reading 271 does. The transfer is nothing; the
    drain is everything. So they are worth doing once and not worth micro-optimising, and two things
    followed: the write's per-request slot check moved into this build (it was one drain per
    layer-call), and the slot lists are now built on the DEVICE from the block-table row instead of
    by an O(seq_len) host loop plus a second upload -- the device form is flat in context where that
    one is ~47 ns a token, which at ~110k tokens is ~37 ms a step. The A/B in the README puts the
    pair at -4.7% (1082 tokens) and -1.2% (4322).
    """

    __slots__ = ("serial", "plan", "block_size", "n_slots", "slot_mapping", "table", "_rows",
                 "_slots", "_angles")

    def __init__(self, md, cache, serial):
        self.serial = serial
        with _phase("plan.split"):
            self.plan = _BatchPlan.of(md)
            self.block_size = TTKVAttentionImpl._block_size(cache)
            self.n_slots = int(cache.shape[0]) * self.block_size

        # THE TWO READBACKS THAT REPLACE ~150. The block table is read back once for the whole step
        # and every slot list is derived from it on the host, so all 28 layers' gathers are covered
        # by this one transfer plus its host-side bounds check.
        with _phase("plan.blocks"):
            self.table = md.block_table
            rows = md.block_table.tolist()
            if len(rows) != self.plan.n_reqs:
                raise RuntimeError(
                    f"the block table has {len(rows)} rows but the batch has {self.plan.n_reqs} "
                    f"requests, so a request would be addressed with another's blocks.")
            self._rows = rows
        self._slots: dict[int, torch.Tensor] = {}
        self._angles: dict[tuple, tuple] = {}

        with _phase("plan.slotmap"):
            n_tokens = self.plan.qsl[-1]
            sm = getattr(md, "slot_mapping", None)
            if sm is None:
                raise RuntimeError(
                    "the metadata carries no `slot_mapping`, so there is nowhere to write the "
                    "store. It is vLLM's own flat slot ids; `metadata.build` has to carry it.")
            self.slot_mapping = sm[:n_tokens]
            # ONE READBACK FOR EVERY WRITE IN THE STEP. This used to run per request per layer, from
            # `write_slot_ids`, which is one device readback -- and so one pipeline drain -- for each
            # of the 28 layers. The tokens a step writes are exactly `slot_mapping[:n_tokens]`, so
            # checking that slice once refuses the same values, and refusing them EAGERLY is the
            # stronger behaviour: a padded slot is a key that cannot be stored, and the read covers
            # `seq_len` rows whatever the write did.
            validate_slot_ids(self.slot_mapping, 0, n_tokens, self.n_slots)

    def slots(self, r: int, device) -> torch.Tensor:
        """This request's flat slot ids, in position order, validated on the host.

        The CHECKS use the host row this step already holds (the row's first block is the request's
        decoder key, so it had to be read back anyway); the SLOTS are built on the device from the
        row tensor. Building them on the host as well would cost an O(seq_len) Python
        `range`/`extend` loop plus a second host-to-device copy of `seq_len` longs -- the only part
        of the plan path that grows with context, and measured in `tests/bench_step_plan.py`.
        """
        got = self._slots.get(r)
        if got is None:
            with _phase("block_slots"):
                seq_len = self.plan.seq_lens[r]
                dc.check_row_host(self._rows[r], seq_len, self.block_size, self.n_slots)
                got = dc.block_slots(self.table[r].to(device), seq_len, self.block_size, check=False)
            self._slots[r] = got
        return got

    def key(self, r: int) -> int:
        """This request's decoder key, from the block table copy this step already holds."""
        return request_key_host(self._rows[r])

    def write_slot_ids(self, r: int, device) -> torch.Tensor:
        """The slot ids this step writes for request `r`, already validated for the whole step.

        The validation is done in `__init__` on the whole step's slice, because it is a readback and
        a readback is a pipeline drain: per request per layer it was 28 of them a step.
        """
        start, end = self.plan.qsl[r], self.plan.qsl[r + 1]
        return self.slot_mapping[start:end]

    def angles(self, cos_sin_cache, seq_len: int, head_dim: int, device):
        """`(cos, sin)` for positions `0..seq_len-1`, computed once per (table, length) per step.

        The bound is checked on the host here (`max_position`), so no device value is read back: the
        largest position this will ask for is `seq_len - 1` and it is known without asking the GPU.
        """
        key = (id(cos_sin_cache), seq_len, head_dim)
        got = self._angles.get(key)
        if got is None:
            positions = torch.arange(seq_len, device=device)
            got = dc.cos_sin_from_cache(cos_sin_cache, positions, head_dim,
                                        max_position=seq_len - 1)
            self._angles[key] = got
        return got


_STEP_ATTR = "_ttkv_step"
_NO_SERIAL_WARNED = [False]


def validate_slot_ids(slot_mapping, start: int, end: int, n_slots: int) -> None:
    """The write's two refusals, done once per step on the whole step's slice.

    `slot_mapping` is a DEVICE tensor and this reads it back -- that is the point: one transfer per
    step covers every layer, where the per-layer form paid for two drains per layer and the
    per-request form (this, called from `_Step.write_slot_ids`) paid for one drain per layer.
    """
    if end <= start:
        return
    ids = slot_mapping[start:end].tolist()
    lo, hi = min(ids), max(ids)
    if lo < 0:
        raise RuntimeError(
            f"slot_mapping contains {lo}, i.e. vLLM's PAD_SLOT_ID. A padded token has no cache "
            f"slot, so its key cannot be stored -- and since decode reads exactly `seq_len` rows, a "
            f"missing one would silently drop a key from the softmax denominator rather than "
            f"shorten the read. Refusing instead.")
    if hi >= n_slots:
        raise IndexError(
            f"slot {hi} is outside the cache, which holds {n_slots} slots")


def _step_for(md, cache) -> _Step:
    """The per-step bundle for this metadata, reused while its serial matches."""
    serial = getattr(md, "ttkv_serial", 0)
    if not serial:
        # No serial means this metadata did not come from `metadata.build`. Refusing to cache is the
        # safe response, and the warning is there because the cost of losing the cache is 28x the
        # readbacks -- a silent slowdown rather than a wrong answer, which is exactly the kind that
        # goes unnoticed.
        if not _NO_SERIAL_WARNED[0]:
            _NO_SERIAL_WARNED[0] = True
            logger.warning(
                "vllm_ttkv: the attention metadata carries no `ttkv_serial`, so the per-step work "
                "(batch split, slot lists, angle rows) is recomputed for every layer instead of "
                "once. `metadata.build` sets it; a metadata built by something else will not.")
        return _Step(md, cache, 0)
    step = getattr(md, _STEP_ATTR, None)
    if step is None or step.serial != serial:
        step = _Step(md, cache, serial)
        setattr(md, _STEP_ATTR, step)
    return step


class _BatchPlan:
    """Which requests decode, which prefill, and where each one's tokens sit in the flat batch.
    Everything here is derived from `query_start_loc` and `seq_lens` -- both of which the engine
    already produces -- so nothing in this file needs a scheduler change. Nothing is carried
    between steps either: the fold boundary that used to need a counter is gone with the tail, so
    the plan is a pure function of the current metadata.
    """

    __slots__ = ("qsl", "seq_lens", "n_reqs", "decodes", "prefills", "decode_tokens")

    def __init__(self, qsl, seq_lens, decodes, prefills, decode_tokens):
        self.qsl = qsl                       # list[int], len n_reqs + 1
        self.seq_lens = seq_lens             # list[int], len n_reqs
        self.n_reqs = len(seq_lens)
        self.decodes = decodes               # list[int], request indices, decodes first
        self.prefills = prefills
        self.decode_tokens = decode_tokens

    @classmethod
    def of(cls, md) -> "_BatchPlan":
        """Split the batch and locate every request's tokens, from metadata alone.

        THE ONE ASSUMPTION, AND IT IS CHECKED RATHER THAN TRUSTED. vLLM reorders decodes to the
        front of the batch (`reorder_batch_threshold`), and that ordering is what lets this walk
        the decode rows as `qsl[r]` for `r < n_decodes`. If it did not hold, the split would still
        arithmetically succeed -- and the decode branch would attend over a prompt chunk's rows
        while reading a slot mapping written for a different population. So the ordering is
        verified, and a mixed order raises instead of being quietly sorted out.
        """
        dev = md.query_start_loc
        if md.query_start_loc is None or md.seq_lens is None or md.block_table is None:
            raise RuntimeError(
                "the attention metadata is missing one of `query_start_loc` / `seq_lens` / "
                "`block_table`, so the batch cannot be split or addressed. `metadata.build` has to "
                "carry all three off `CommonAttentionMetadata` -- a partially-populated metadata "
                "does not fail where it is built, it fails here.")
        qsl_cpu = getattr(md, "query_start_loc_cpu", None)
        with _phase("plan.qsl"):
            qsl = (qsl_cpu if qsl_cpu is not None else dev).tolist()
            _sync("query_start_loc")
        # EXACT ROW COUNTS COME FROM THE DEVICE TENSOR, NOT FROM A CPU COPY. vLLM's
        # `seq_lens_cpu_upper_bound` is a BOUND -- it exists so a sub-batch can be sized without a
        # sync -- and using it as a row count would read past the end of one request and into
        # whatever the next block holds, i.e. attend over another request's tokens with no error.
        # One D2H per STEP is the price (`_Step` caches this for every layer); the alternative is a
        # silently wrong softmax.
        with _phase("plan.seqlens"):
            seq_lens = md.seq_lens.tolist()
            _sync("seq_lens")
        if len(qsl) != len(seq_lens) + 1:
            raise RuntimeError(
                f"query_start_loc has {len(qsl)} entries but seq_lens has {len(seq_lens)}; a "
                f"cu_seqlens is n_reqs + 1, so these describe different batches")

        q_len = [qsl[i + 1] - qsl[i] for i in range(len(seq_lens))]
        decodes = [r for r, ql in enumerate(q_len) if ql == 1]
        n_decodes = len(decodes)
        if decodes != list(range(n_decodes)):
            raise RuntimeError(
                f"decode requests are not contiguous at the front of the batch (decode rows "
                f"{decodes}, query lengths {q_len}). vLLM's reorder puts them there; without it a "
                f"per-request slot mapping cannot be matched to a request index.")
        return cls(qsl, seq_lens, decodes,
                   [r for r in range(len(seq_lens)) if q_len[r] > 1],
                   qsl[n_decodes] if n_decodes else 0)


class TTKVAttentionImpl(AttentionImpl):
    def __init__(self, num_heads: int, head_size: int, scale: float,
                 num_kv_heads: int | None = None, alibi_slopes=None, sliding_window=None,
                 kv_cache_dtype: str = "auto", logits_soft_cap=None, attn_type=None,
                 kv_sharing_target_layer_name=None, **kwargs):
        # EVERY REFUSAL BELOW WOULD OTHERWISE BE A WRONG ANSWER RATHER THAN A CRASH.
        # ALiBi and a soft cap both change the scores; a sliding window changes which keys are
        # eligible; KV sharing would have this layer read another layer's store, which holds a
        # different decoder. None of those raise on their own -- they just produce attention that
        # is not the model's -- so each one is refused where it is offered.
        if alibi_slopes is not None:
            raise NotImplementedError(
                "TT-KV does not support ALiBi: `krt` has no bias term, so the bias would be "
                "dropped from scores the kernel has already produced.")
        if logits_soft_cap is not None:
            raise NotImplementedError(
                "TT-KV does not support logits_soft_cap; the cap is not applied in `krt`.")
        if sliding_window is not None:
            raise NotImplementedError(
                "TT-KV does not support a sliding window: the read path attends over the whole "
                "store, so a window would silently attend to keys the model excludes.")
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError(
                "TT-KV does not support KV sharing: a shared layer would read another layer's "
                "store, which holds a different fitted decoder (r1 is per layer).")
        if AttentionType is not None and attn_type is not None and attn_type != AttentionType.DECODER:
            raise NotImplementedError(f"TT-KV only supports decoder attention, got {attn_type!r}")

        self.num_heads = num_heads
        # `head_size` here is the MODEL's head dim (128 for Llama-3.1-8B) -- NOT the packed slot
        # width, which is what the KV cache *spec* publishes and is roughly a twentieth of this.
        # Two different quantities that both get called a head size; `_layout` is where they are
        # made to confront each other.
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads or num_heads
        self.scale = float(scale)
        self.kv_cache_dtype = kv_cache_dtype
        if num_heads % self.num_kv_heads:
            raise ValueError(
                f"num_heads={num_heads} is not a multiple of num_kv_heads={self.num_kv_heads}")

    # ------------------------------------------------------------------ the entry point
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None,
                output_scale=None, output_block_scale=None):
        """`query`/`key` are POST-RoPE and are used ONLY on the prefill branch.

        That is not a shortcut: the model forward still rotates before calling attention, so on a
        multi-token step `key` is the rotated prompt and the cheapest correct thing is to attend
        over it directly. The STORE holds raw K, which is what decode reads -- and decode is the
        only branch that has to pay for `krt`.

        `output_scale`/`output_block_scale` are refused rather than ignored: they mean the caller
        expects fused output quantization, and a backend that silently writes unquantized values
        into a buffer sized for quantized ones is a corruption, not a fallback.

        THE RETURN VALUE IS DISCARDED. `unified_attention_with_output` declares
        `mutates_args=["output"]` (`attention.py`), so this method must write into `output` rather
        than return a tensor -- a `return` here is a no-op that looks like an implementation.
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "TT-KV does not support fused output quantization; output_scale and "
                "output_block_scale must be None.")
        if attn_metadata is None:
            if output is None:
                raise ValueError("attn_metadata and output are both None; there is nothing to write")
            return output.fill_(0)

        n_tokens, _, _ = query.shape
        if output is None:
            output = torch.empty(n_tokens, self.num_heads, self.head_size,
                                 dtype=query.dtype, device=query.device)

        # THE CACHE IS VALIDATED FIRST, because `_Step` needs its shape: a per-layer cache tensor
        # arrives as ONE tensor (K and V share the region), and `_cache_tensor` is what refuses the
        # two-tensor form. Asking `_Step` to size itself from an unvalidated object would replace
        # that refusal with an AttributeError on a tuple.
        lay = self._layout(layer)
        cache = self._cache_tensor(kv_cache)

        with _phase("plan"):
            step = _step_for(attn_metadata, cache)
        plan = step.plan
        N = plan.qsl[-1]
        if _PROFILE and not getattr(attn_metadata, "_ttkv_counted", False):
            # vLLM rebuilds the metadata once per step and hands the SAME object to every layer of
            # the group, so its identity is the step boundary -- the same fact that made a per-step
            # write flag wrong.
            _COUNT["steps"] += 1
            attn_metadata._ttkv_counted = True
        if _TRACE:
            logger.info("vllm_ttkv: %s N=%d decodes=%s prefills=%s seq_lens=%s",
                        getattr(layer, "layer_name", None), N, plan.decodes, plan.prefills,
                        plan.seq_lens)
        actual = getattr(attn_metadata, "num_actual_tokens", None)
        if actual is not None and int(actual) != N:
            raise RuntimeError(
                f"query_start_loc says {N} query tokens but num_actual_tokens says {actual}. Both "
                f"describe the same batch, so one of them is stale -- and slicing by the wrong one "
                f"means attending over padding rows, which are not zero.")
        if N > n_tokens:
            raise RuntimeError(
                f"metadata describes {N} query tokens but the query tensor holds {n_tokens}")
        if N == 0:
            return output

        if _PROFILE:
            _COUNT["layers"] += 1

        # THE WRITE COMES FIRST, AND IT COMES FROM HERE. `forward_includes_kv_cache_update = True`
        # tells vLLM to skip its own cache write, so this is the only writer -- and this method is
        # the only place that holds all three things the write needs: the raw K (stashed by
        # `patch.py`'s rotary wrapper), vLLM's slot mapping, and the per-request split. `value` is
        # final as handed over: V is never rotated, so unlike K it needs no capture.
        #
        # UNCONDITIONALLY, AND THAT IS THE FIX FOR A REAL FAILURE. This used to consult a
        # `ttkv_kv_written` flag on the attention metadata, copied from HPC's `hpc_kv_written`
        # handshake -- but vLLM hands ONE metadata object to every layer of a group, so layer 0 set
        # the flag and every other layer skipped its own write while still reading. The run failed
        # at layer 6 with "no TT-KV store for layer 'model.layers.6.self_attn.attn'". A per-step
        # flag cannot express "this LAYER wrote"; the stash can, because `take_raw_k` POPS: it is
        # keyed by layer, written by this layer's own rope call a moment earlier in this same
        # forward, and a second consume raises instead of silently writing twice.
        with _phase("write"):
            self._write_store(layer, lay, cache, value, step, query.device)

        if plan.decodes:
            block_size = self._block_size(cache)
            store = store_for_layer(getattr(layer, "layer_name", None))
            if store is None:
                raise RuntimeError(
                    f"no TT-KV store for layer {getattr(layer, 'layer_name', None)!r}, so there is "
                    f"no angle table to read. The store is created by the writer "
                    f"(`impl._write_store` -> `rope_store.ensure_store`) on this layer's first "
                    f"chunk.")
            layer_name = getattr(layer, "layer_name", None)
            # ONE FALLBACK, ONLY WHERE IT IS UNAMBIGUOUS. The fitted basis is per (request, layer)
            # -- each prefill is its own TT-SVD fit -- so a layer-level decoder applied across a
            # batch is one request's fit used for all of them, which the codec measured as a 5.8
            # shift in absolute score. With a single decode request the two are the same thing, so
            # the fallback is allowed there and refused beyond it.
            fallback = self._single_request_decoder(layer) if len(plan.decodes) == 1 else None
            for r in plan.decodes:
                # `req_key`, NOT `key`: this loop used to assign `key = step.key(r)`, which SHADOWS
                # the `key` parameter of `forward` -- so the prefill branch below was handed a
                # request key (an int) instead of the key tensor, and failed on `key[start:end]`.
                # It never fired while a step was either all-decode or all-prefill; a MIXED batch,
                # which is what chunked prefill with more than one request produces, is what exposes
                # it. `test_a_mixed_batch_of_decodes_and_prefills_uses_both_paths` pins it.
                req_key = step.key(r)
                kdec = decoder_for_request(layer_name, req_key)
                decoder_f = kdec.decoder_f if kdec is not None else fallback
                if decoder_f is None:
                    decoder_f = fallback
                if decoder_f is None:
                    raise RuntimeError(
                        f"no fitted TT-KV decoder for request {req_key} in layer {layer_name!r}, "
                        f"and {len(plan.decodes)} requests are decoding. A decoder belongs to a "
                        f"SEQUENCE, not to a layer: it is that sequence's own TT-SVD fit, and a "
                        f"rank-r1 basis spanning several requests has to cover their combined key "
                        f"diversity. Register it with "
                        f"`rope_store.register_decoder(layer_name, request_key(block_row), "
                        f"decoder_f)` -- keyed by `block_table[row, 0]`, which is what the encoder "
                        f"has in hand and what a kernel could compute too.")
                dc.check_layout_matches_decoder(lay, decoder_f)
                with _phase("decode_total"):
                    row = self._decode_one(lay, store, decoder_f, cache, step, query, r)
                output[plan.qsl[r]] = row.to(output.dtype)

        if plan.prefills:
            self._prefill(layer, lay, cache, step, query, key, value, plan, output)

        return output

    # ------------------------------------------------------------------ the plan
    def _plan(self, md) -> _BatchPlan:
        """The batch split for this metadata. Delegates to `_BatchPlan.of`.

        Kept as a method because it is the seam the ordering test drives directly
        (`test_refuses_a_batch_whose_decodes_are_not_at_the_front`), which is also why the ordering
        check lives in `_BatchPlan.of` rather than in `_Step`.
        """
        return _BatchPlan.of(md)

    # ------------------------------------------------------------------ the decode branch
    def _decode_one(self, lay, store, decoder_f, cache, step, query, r):
        """One request's decode output, `[HQ, d]` fp32, read out of the packed store.

        Rows read == `seq_lens[r]`, and that count INCLUDES the token this step produced: the store
        write for the current step happens earlier in the same forward, before attention. So the
        newest key is read through the same path as every other one -- there is no separate
        "current token" operand, which is exactly what a dense tail exists to avoid.

        EVERYTHING LAYER-INDEPENDENT COMES FROM `step`, which is where the readbacks went: the slot
        list, its bounds check and the angle rows are the same for all 28 layers of the step, and
        recomputing them per layer was the largest single cost in the decode (see `_Step`).
        """
        seq_len = step.plan.seq_lens[r]
        with _phase("block_slots"):
            slots = step.slots(r, query.device)
        with _phase("gather"):
            blob = dc.gather_slots(cache, slots, lay.compressed_bytes_per_token(), check=False)
        with _phase("split_blob"):
            k_slots, v_slots = dc.split_blob(lay, blob)
        with _phase("cos_sin"):
            cos, sin = step.angles(store.cos_sin_cache, seq_len, self.head_size, query.device)
        dc.check_row_counts(k_slots.shape[0], v_slots.shape[0], cos.shape[0], seq_len)

        q = query[step.plan.qsl[r]].to(torch.float32)                  # [HQ, d], post-rotary
        with _phase("krt"):
            scores = self._krt(lay, k_slots, decoder_f, q, cos, sin)   # [HQ, n], UNSCALED
        with _phase("softmax"):
            p = torch.softmax(scores * self.scale, dim=-1)
        with _phase("v_out"):
            return self._v_out(lay, v_slots, p)

    # ------------------------------------------------------------------ the write
    def _write_store(self, layer, lay, cache, value, step, device):
        """Encode this step's raw K and V into the packed store.

        THE PHASE IS CHOSEN BY THE REGISTRY, NOT BY A COUNTER. A request with no registered decoder
        is on its first chunk, so it gets a real TT-SVD fit and keeps the decoder; a request with
        one is past that point, so it projects onto the frozen decoder. That makes the fold
        schedule a pure function of which requests the batch contains -- there is no per-request
        count of tokens that has to survive preemption, recompute or reordering, and no state to get
        out of step with the scheduler.

        `A request with prior context and no decoder` is refused rather than fitted: fitting on one
        token would produce a rank-1 basis for a sequence that needs a real one, and every later
        step would inherit it.
        """
        from vllm_ttkv import encode
        from vllm_ttkv.patch import take_raw_k

        plan = step.plan
        name = getattr(layer, "layer_name", None)
        stash = take_raw_k(name)          # pops: a double-consume is an error, not a stale read
        raw_k = stash["raw_k"]
        # `stash["positions"]` is deliberately unused: the store holds POSITION-FREE keys, which is
        # the entire ratio argument, so the fit and the residual are both blind to position. Only
        # the read path needs a position, and it takes it from `seq_len` at read time.
        if value is None:
            raise RuntimeError(
                f"vllm_ttkv: layer {name!r} got no value tensor, so V cannot be stored. V is never "
                f"rotated, so this is passed through unmodified and there is no capture to fall "
                f"back on.")

        K = encode.as_kv_heads(raw_k, self.num_kv_heads, self.head_size, "the captured raw key")
        V = encode.as_kv_heads(value, self.num_kv_heads, self.head_size, "value")
        if lay.H != self.num_kv_heads:
            raise RuntimeError(
                f"the layout holds one residual row per head and was built for H={lay.H}, but this "
                f"layer has {self.num_kv_heads} KV heads. A mismatch writes heads into the wrong "
                f"rows of every slot -- same shape, permuted heads.")

        N = plan.qsl[-1]
        K, V = K[:N], V[:N]
        cfg = encode.active_config()
        store = ensure_store(name, self._cos_sin_cache(layer, stash), self.num_heads,
                             self.num_kv_heads, self.head_size, lay)

        fitted = projected = 0

        def _write_one(r, fk, start, end):
            with _phase("write.pack_and_write"):
                # `check=False`: this step validated the whole slot mapping once (`write_slot_ids`),
                # so the two device bounds checks here would be 2 more readbacks per layer.
                encode.pack_and_write(cache, step.write_slot_ids(r, device), fk,
                                      V[start:end].permute(1, 0, 2).contiguous(), lay, check=False)

        # THE PROJECTIONS ARE COLLECTED BY TOKEN COUNT AND RUN ONE BATCH PER GROUP; the FITS do not
        # batch and are not collected. Two different reasons for the two halves:
        #
        # A projection is ~20 launches on a one-token tensor in a decode step, so `G` of them cost
        # `G` times the LAUNCHES for the same total work -- that is the cost this removes. A fit is
        # one request's TT-SVD, it happens once per request ever, and it is 135 ms of real
        # arithmetic, so there is nothing to gain by pairing it with a neighbour and a real risk in
        # sharing a basis between requests (measured 5.8 points wrong).
        #
        # Grouping by `end - start` is what makes the batch legal: members must share a token count,
        # because the batch is a stacked `[G, H, n, d]`. A decode step is one group of `n == 1`; a
        # prefill step's requests differ in length and mostly stay per-request.
        groups: dict = {}
        for r in range(plan.n_reqs):
            start, end = plan.qsl[r], plan.qsl[r + 1]
            if end <= start:
                continue
            key = step.key(r)
            kdec = decoder_for_request(name, key)
            # `[t, H, d]` -> `[H, t, d]`: the codec's whole K path is head-major within a token.
            Kr = K[start:end].permute(1, 0, 2)
            if kdec is None:
                if plan.seq_lens[r] != end - start:
                    raise RuntimeError(
                        f"request {key} arrives with seq_len={plan.seq_lens[r]} but only "
                        f"{end - start} tokens in this step, and has no fitted decoder. That is a "
                        f"CONTINUATION step for a request whose prefill never ran -- so there is no "
                        f"basis to project onto, and fitting one on {end - start} tokens would give "
                        f"this sequence a rank that its own first chunk never got.")
                with _phase("write.fit"):
                    fk = encode.fit_into_budget(Kr.contiguous(), lay, cfg)
                register_decoder(name, key, fk.decoder)
                fitted += 1
                _write_one(r, fk, start, end)
            else:
                groups.setdefault(end - start, []).append((r, kdec))

        for _n_r, members in groups.items():
            if len(members) == 1:
                r, kdec = members[0]
                with _phase("write.project"):
                    fk = encode.encode_frozen(
                        K[plan.qsl[r]:plan.qsl[r + 1]].permute(1, 0, 2).contiguous(), kdec, lay)
                projected += 1
                _write_one(r, fk, plan.qsl[r], plan.qsl[r + 1])
                continue
            rows = [(r, kdec, plan.qsl[r], plan.qsl[r + 1]) for r, kdec in members]
            with _phase("write.project.batch"):
                batch_k = torch.stack([K[s:e].permute(1, 0, 2) for _, _, s, e in rows]).contiguous()
                fks = encode.encode_frozen_batch(batch_k, [kd for _, kd, _, _ in rows], lay)
            projected += len(rows)
            # Each request still writes to ITS OWN slots through its own fitted object. The batched
            # call produced equal coefficients, not a merged store.
            for (r, _kd, s, e), fk in zip(rows, fks):
                _write_one(r, fk, s, e)

        logger.debug("vllm_ttkv: %s wrote %d tokens (%d fitted, %d projected) into %d-byte slots",
                     name, N, fitted, projected, lay.compressed_bytes_per_token())

    @staticmethod
    def _cos_sin_cache(layer, stash):
        """The model's OWN RoPE table.

        Not rebuilt from `config.rope_theta`, and that is not pedantry: Llama-3.1's `inv_freq`
        carries a llama3 frequency ramp the config does not reproduce, and a wrong frequency set is
        still orthogonal and still position-dependent -- so it rotates every stored row to the wrong
        angle while every shape check passes. HPC passes `cos_sin_cache=self.rotary_emb.cos_sin_cache`
        for the same reason.

        It comes off the STASH first, because that is the only place it is reachable from: the
        attention layer carries a `layer_name` and no `rotary_emb` (the rope belongs to the model's
        attention block, one level up), so `layer.rotary_emb` is absent on every real model. The
        stash is written by `patch._install_capture` off the very module the model rotates with, so
        it is the true table rather than a re-derivation. The attribute fallback stays for hand-built
        layers in tests.
        """
        csc = stash.get("cos_sin_cache")
        if csc is None:
            csc = getattr(getattr(layer, "rotary_emb", None), "cos_sin_cache", None)
        if csc is None:
            raise RuntimeError(
                f"layer {getattr(layer, 'layer_name', None)!r} has no RoPE table: the rotary "
                f"wrapper stashed none, and `layer.rotary_emb.cos_sin_cache` is absent too. The "
                f"table has to come from the model -- rebuilding it from config is silent-wrong "
                f"rather than approximate.")
        return csc

    # ------------------------------------------------------------------ the prefill branch
    def _prefill(self, layer, lay, cache, step, query, key, value, plan, output):
        """Multi-token requests, by the cheapest CORRECT route -- and which route that is depends on
        whether the request already has context.

        TWO CASES, AND THEY ARE GENUINELY DIFFERENT

        * `seq_len == q_len`: the FIRST chunk. The model handed us this chunk's own post-RoPE
          `key`/`value`, so attending over them directly is the same attention every other backend
          runs (TurboQuant's prefill does the same thing). Reading the store here would mean
          rotating reconstructed raw keys forward only to contract them against queries the model
          already rotated -- strictly more work for the same answer.
        * `seq_len > q_len`: a CONTINUATION chunk (chunked prefill, or a prefix-cache hit). The
          request has `seq_len - q_len` keys in the store from earlier steps, and this branch only
          sees the current batch. Attending over the chunk alone would SILENTLY DROP every one of
          those keys -- a confident wrong answer, which is why this used to be refused outright.

        THE REFUSAL IS GONE BECAUSE THE KERNEL EXISTS. A single-query `krt` contracts one query row
        per (kv head, group) and returns `[HQ, n]`, so it cannot express `q_len` queries over the
        same `n` rows; `krt_multi_from_slots` can, and it reconstructs each stored row ONCE for all
        `q_len` queries rather than once per query. So the continuation is now computed over the
        whole store, and the row count is checked rather than assumed.
        """
        for r in plan.prefills:
            start, end = plan.qsl[r], plan.qsl[r + 1]
            q_len = end - start
            seq_len = int(plan.seq_lens[r])
            if seq_len < q_len:
                raise RuntimeError(
                    f"request {r} has seq_len={seq_len} but {q_len} query tokens in this step, so "
                    f"the store holds FEWER rows than the step has queries. The store is written "
                    f"before attention in the same `forward`, so this cannot be a first chunk -- it "
                    f"means the slot mapping and the sequence length disagree.")
            if seq_len == q_len:
                output[start:end] = self._sdpa(
                    query[start:end].to(torch.float32),
                    key[start:end].to(torch.float32),
                    value[start:end].to(torch.float32)).to(output.dtype)
                continue
            self._prefill_continuation(layer, lay, cache, step, query, plan, r, start, end,
                                      seq_len, output)

    def _prefill_continuation(self, layer, lay, cache, step, query, plan, r, start, end, seq_len,
                              output):
        """A chunk that has EARLIER keys in the store: multi-query `krt` over the whole store.

        The store holds one row per stored token and its addressing is token-major, so the rows this
        request may attend come from `step.slots(r)` and their own positions come from the angle
        table -- and `p0 = seq_len - q_len` is the absolute position of the chunk's first query, so
        causality is `row <= p0 + j`. Nothing here needs the model's `key`/`value`: the values come
        out of the store, which is the whole point of storing them.

        THE READOUT IS QUERY-BLOCKED TOO, and for a measured reason rather than symmetry. The
        per-query `v_out_from_slots` returns `[HQ, d]`, so a chunk of `q_len` queries costs `q_len`
        launches -- 5753 of them a layer on a 4-prompt continuation -- and because its contraction
        is `[GP, KT] x [KT, HALF]` the only dimension that can clear `tl.dot`'s minimum of 16 is the
        GROUP axis, which it pads from `G=4` to 16. Moving the query axis into `M` fixes both at
        once: one launch per chunk, and `M = BQ*GP` clears 16 with no padding at all.
        """
        layer_name = getattr(layer, "layer_name", None)
        store = store_for_layer(layer_name)
        if store is None:
            raise RuntimeError(
                f"no TT-KV store for layer {layer_name!r}, so there is no angle table to read and "
                f"no rows to attend over. It is created by the writer on this layer's first chunk.")
        kdec = decoder_for_request(layer_name, step.key(r))
        decoder_f = kdec.decoder_f if kdec is not None else self._single_request_decoder(layer)
        if decoder_f is None:
            raise RuntimeError(
                f"request {step.key(r)} has seq_len={seq_len} but no fitted decoder in layer "
                f"{layer_name!r}, so there is no basis to read its stored rows against. That means "
                f"its first chunk never reached the writer -- a continuation cannot be decoded "
                f"without the fit that its first chunk produced.")
        dc.check_layout_matches_decoder(lay, decoder_f)

        q_len = end - start
        slots = step.slots(r, query.device)
        blob = dc.gather_slots(cache, slots, lay.compressed_bytes_per_token(), check=False)
        k_slots, v_slots = dc.split_blob(lay, blob)
        cos, sin = step.angles(store.cos_sin_cache, seq_len, self.head_size, query.device)
        dc.check_row_counts(k_slots.shape[0], v_slots.shape[0], cos.shape[0], seq_len)

        # `[q_len, HQ, d]` -> `[HQ, q_len, d]`: the kernel takes the query head-major, the same
        # convention as the scores it returns. Getting this wrong is not a shape error -- the
        # launcher derives the group size from `q.shape[0]`, so a swapped axis changes `G` and
        # returns scores of the right shape for the wrong heads.
        q = query[start:end].permute(1, 0, 2).contiguous().to(torch.float32)
        scores = krt_multi_from_slots(lay, k_slots, decoder_f, cos, sin, q, p0=seq_len - q_len)
        p = torch.softmax(scores * self.scale, dim=-1)                 # [HQ, q_len, seq_len]
        # ONE launch for the whole chunk, not `q_len` of them. The masked positions are already
        # exactly zero here -- softmaxing a `-inf` score gives 0 -- so the readout needs no causal
        # argument, and a zero weight contributes a zero row. `v_out_multi_from_slots` also drops
        # the group padding the single-query kernel needs, because the query axis clears `tl.dot`'s
        # minimum `M` on its own.
        out = v_out_multi_from_slots(lay, v_slots, p)                  # [HQ, q_len, d]
        output[start:end] = out.permute(1, 0, 2).to(output.dtype)      # [q_len, HQ, d]

    # ------------------------------------------------------------------ kernel shims
    def _krt(self, lay, k_slots, decoder_f, q, cos, sin):
        """Post-rotary scores `[HQ, n]` fp32, UNSCALED, straight from the K slots.

        The Triton port is the DEFAULT, not a placeholder: it needs no toolchain (Triton ships with
        vLLM and JITs against whatever is present), whereas an AOT extension forces every user to
        have a matching nvcc, CUDA version and arch at install time. It is also FASTER where real
        layers live -- 0.64x the CUDA kernel at r1=278, measured; the two cross near r1=90 and the
        CUDA kernel wins only below that, which is the short-prefix regime. So `csrc/` is the
        opt-in fast path, and the CUDA body needs no changes to get there: it is plain CUDA C++
        (shared memory, __syncthreads, fp32 FMA, no wmma), and only its delivery changes -- turn
        the `-DTTKV_HQ/-DTTKV_HD/-DTTKV_G/-DTTKV_KT` NVRTC specialisations into template
        parameters, keeping the tiling constant equal to the compiled KT (it sets both the
        shared-memory size and the grid, so a mismatch is not a small error).

        Two Triton traps are load-bearing and both are closed inside `krt_triton` -- do not
        "simplify" them away: `do_not_specialize=['n','r1']` (Triton's divisibility-by-16 argument
        specialization ABORTS the compiler on real ranks) and `input_precision='ieee'` (`tl.dot`
        defaults to TF32 for fp32, a 1700x precision regression that still returns correctly-shaped
        scores).
        """
        return krt_from_slots(lay, k_slots, decoder_f, cos, sin, q)

    def _sdpa(self, query, key, value):
        """First-chunk prefill: stock causal SDPA over the dense q/k/v the model handed us.

        `key`/`value` are the model's own post-RoPE tensors, so this is the same attention every
        other backend runs, and TurboQuant's prefill does exactly the same thing ("Prefill:
        standard SDPA on uncompressed K/V, then quantize"). There is no absorption to special-case
        on this path, and reading the store here would mean rotating raw keys forward only to
        contract them against queries the model already rotated.
        """
        return dc.attend_prefill(query, key, value, self.scale)

    def _tail_scores(self, *_, **__):
        """Removed, deliberately. See the module docstring.

        The dense tail existed to amortise the cost of appending one row to the torch container --
        a re-quantise and re-pack of the whole store, ~20 ms/layer/token. In vLLM a row is written
        to its slot through `slot_mapping` and encoding one token against the frozen decoder costs
        0.009 ms, so a tail would buffer rows for no reason while introducing a second row
        population with a different rotation state and a ragged per-request length.

        The invariant it carried -- that the rows the kernels read are exactly the rows the request
        owns -- lives in `decode.check_row_counts`, asserted on every decode.
        """
        raise RuntimeError(
            "vllm_ttkv: the dense tail does not exist in this design, so nothing should be asking "
            "for tail scores. Rows are encoded into the packed store on the step they are produced "
            "(0.009 ms/token against the frozen decoder) and decode reads every row back through "
            "`krt`. Reaching this means something reintroduced folding -- the metadata no longer "
            "carries a tail length at all, and `decode.check_row_counts` is what carries the "
            "invariant the tail bookkeeping used to protect.")

    def _v_out(self, lay, v_slots, p):
        """Output `[HQ, d]` fp32 from the V slots and the softmax weights.

        V is never dequantised into a dense block. The lane codes are read per stored row and
        consumed immediately, because building `[H*n, d]` was measured at 44.4 ms/token of
        `masked_scatter_` in the torch path -- twice the entire dense decode budget of the cache
        this exists to beat.
        """
        return v_out_from_slots(lay, v_slots, p)

    # ------------------------------------------------------------------ resolution and guards
    def _layout(self, layer):
        lay = getattr(layer, "ttkv_layout", None)
        if lay is None:
            lay = layout_for_layer(getattr(layer, "layer_name", None))
        if lay is None:
            info = calibration_info()
            if info:
                why = (f"A calibration table is loaded (model={info.get('model')}, "
                       f"context={info.get('context')}) but has no entry for this layer index.")
            else:
                why = ("No calibration table is loaded at all -- point VLLM_TTKV_CALIBRATION at a "
                       "tools/calibrate_r1.py artifact.")
            raise RuntimeError(
                f"no TT-KV layout for layer {getattr(layer, 'layer_name', None)!r}. {why}")
        # The layout is a rank for a matrix of a particular shape, so a table measured at one head
        # dim is not merely imprecise at another -- it is a rank for a different matrix. `r1` also
        # sizes the slot, so a mismatch misreads every field after the coefficients.
        if lay.d != self.head_size:
            raise RuntimeError(
                f"the layout was built for head_dim={lay.d} but this layer has "
                f"head_size={self.head_size}. Re-run tools/calibrate_r1.py for this model.")
        return lay

    @staticmethod
    def _cache_tensor(kv_cache):
        if isinstance(kv_cache, (tuple, list)):
            raise RuntimeError(
                f"vllm_ttkv: expected ONE cache tensor but got {len(kv_cache)}. K and V share a "
                f"region because `state_content_bytes` describes the whole per-token cell (dense "
                f"K||V is its default), so `kv_cache[..., :k]` and `kv_cache[..., k:]` are the two "
                f"slots. Taking `kv_cache[0]` here would read K as though it were the combined "
                f"blob.")
        return kv_cache

    @staticmethod
    def _block_size(cache) -> int:
        if cache.dim() == 4:                    # [num_blocks, num_head_slots, block_size, C]
            return int(cache.shape[2])
        if cache.dim() == 3:                    # head slots already squeezed
            return int(cache.shape[1])
        raise ValueError(
            f"expected an LBNHC page ([num_blocks, head_slots, block_size, C]), got "
            f"{tuple(cache.shape)}")

    @staticmethod
    def _single_request_decoder(layer):
        """The layer's own decoder, valid only when the batch holds ONE decode request.

        With one request, "this layer's fit" and "this request's fit" are the same object, so either
        source is correct. `impl.forward` refuses this fallback as soon as a second request decodes,
        because there it stops being a fallback and becomes one sequence's basis applied to another.
        """
        name = getattr(layer, "layer_name", None)
        dec = getattr(layer, "ttkv_decoder_f", None)
        if dec is None:
            dec = decoder_for_layer(name)
        return dec
