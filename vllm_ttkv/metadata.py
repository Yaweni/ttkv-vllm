"""Attention metadata for TT-KV: what the impl needs from the engine, and where each field comes from.

Copied in SHAPE from `HpcAttnMetadata`, which is the in-tree example of a backend and a model-forward
module cooperating through metadata -- but NOT in that one respect, and the difference is load-bearing:
HPC's model-forward module writes raw K and tells the impl so through a per-step flag
(`hpc_kv_written`). TT-KV's impl is itself the writer, and a per-step flag on the metadata CANNOT be
used for it, because vLLM hands ONE metadata object to every layer of a group. With the flag, the
first layer ran the write, set it, and every other layer then skipped its own write while still
reading its own store -- measured, the run failed at layer 6 with "no TT-KV store for layer
'model.layers.6.self_attn.attn'". "Has this LAYER written" is not a per-step fact; the stash answers it
per layer, and `patch.take_raw_k` POPS so a second write in the same step raises.

THE TAIL FIELD IS GONE, AND THAT IS A DELETION RATHER THAN A SIMPLIFICATION
    This file used to carry `ttkv_tail_len`: the count of dense un-folded rows still waiting to be
    folded into the packed store. It existed because appending one row to the torch container
    re-quantised and re-packed the whole store (~20 ms/layer/token, 86% of the decode step), so rows
    were buffered and folded in batches. In vLLM a row goes to its slot through `slot_mapping` and
    encoding one token against the frozen decoder costs 0.009 ms, so there is nothing to buffer. The
    tail is empty by construction, and it was the only reason the read path needed a ragged
    per-request length. `decode.check_row_counts` carries the invariant it was actually protecting.

TWO FIELDS ARE LOAD-BEARING IN A WAY THAT IS NOT OBVIOUS
    `block_table` is not only addressing: `block_table[row, 0]` is also the request's DECODER key
    (`rope_store.request_key`), because the fitted basis is per request and vLLM's
    `CommonAttentionMetadata` carries no request identifiers. Lose this field and the decoders
    become unreachable.

    `seq_lens_cpu_upper_bound` is a BOUND, not a count. It is named that way here, after vLLM's own
    field, to keep it from being used as a row count -- which would read rows past the end of one
    request and into whatever the next block holds. The read path takes its row counts from the
    device `seq_lens` for that reason (`impl._plan`).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.attention.backend import AttentionMetadata, AttentionMetadataBuilder

# Every metadata object gets the next serial. `impl._Step` caches layer-independent per-step work on
# the metadata and reuses it only while the serial matches, so the cache cannot survive into a new
# step -- even if vLLM ever pools and recycles these objects, which would otherwise make a stale
# bundle imminent rather than theoretical.
_SERIAL = [0]


@dataclass
class TTKVAttentionMetadata(AttentionMetadata):
    """The batch description the TT-KV read path needs. Every field is per STEP, not per cache.

    `AttentionMetadata` itself is an empty marker class, so all of these are declared here rather
    than inherited -- and they default to None so the capture-time construction below can build an
    empty one. `impl._plan` refuses a metadata with any of the four missing, so a default is a loud
    failure rather than a subtle one.
    """

    # [num_reqs] int -- the exact context length of each request. THE row count for a decode, and it
    # INCLUDES the token this step produced, because the store write happens first in the same step.
    seq_lens: torch.Tensor | None = None
    # [num_tokens] int64 -- vLLM's own flat slot ids. `block * block_size + offset`, the same
    # convention `decode.block_slots` reads, so the write and the read agree by construction.
    slot_mapping: torch.Tensor | None = None
    # [num_reqs, max_num_blocks] int32. Addressing AND the request's decoder key -- see the module
    # docstring.
    block_table: torch.Tensor | None = None
    # [num_reqs + 1] int32 -- cu_seqlens over queries. The decode/prefill split, and each request's
    # token range, both come from here, so nothing else maintains a request counter.
    query_start_loc: torch.Tensor | None = None

    # CPU-resident copies of the two above. `query_start_loc_cpu` IS exact (a cu_seqlens has no
    # padding) and saves a sync on the split; the seq_lens copy does not exist on purpose -- see the
    # module docstring.
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu_upper_bound: torch.Tensor | None = None

    # The engine's own absolute positions, [num_tokens]. Carried because it is nearly free and
    # useful for cross-checking a store, NOT because the encode needs it: the fit and the residual
    # are both blind to position, which is the whole ratio argument for storing raw K.
    positions: torch.Tensor | None = None

    num_actual_tokens: int = 0
    max_query_len: int = 0
    max_seq_len: int = 0

    # This step's serial, set by `build`. `impl._Step` keys its per-step cache on it; 0 means "not
    # built by `metadata.build`", and the impl then recomputes per layer rather than caching on a
    # guess (it warns once, because the whole point of the cache is that recomputing is 28x the
    # readbacks).
    ttkv_serial: int = 0


class TTKVAttentionMetadataBuilder(AttentionMetadataBuilder):
    """Builds the per-step metadata by COPYING from `CommonAttentionMetadata`.

    NO CUDA GRAPHS, and the base default (`AttentionCGSupport.NEVER`) is the right one rather than
    something to improve on later. The impl is host-interactive by construction: it splits the batch
    by `.tolist()`, gathers each request's slots with an indexed read, and encodes against a decoder
    held in a Python registry. A graph captured over that would bake in one batch's request count
    and lengths -- it would not raise, it would replay the wrong batch. Making this capturable is a
    real piece of work (fixed shapes, no host reads, device-side decoder addressing), not a flag.
    """

    # Ask the runner to put decodes at the front, which `impl._plan` VERIFIES rather than assumes:
    # it reads the per-request query lengths and refuses if a decode is not in the leading run. Same
    # value HPC and TurboQuant use.
    reorder_batch_threshold: int = 1

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "TT-KV does not support speculative decoding. A spec-decode step adds query tokens "
                "whose keys are not yet committed to the store, and the read path's per-request row "
                "count would have to distinguish 'drafted' from 'stored' -- which the prefill branch "
                "already refuses (`MULTI-QUERY krt` is the missing piece), so this refuses one step "
                "earlier and names the reason.")
        self._init_reorder_batch_threshold(reorder_batch_threshold=1,
                                           supports_spec_as_decode=False)

    def build(self, common_prefix_len=None, common_attn_metadata=None, fast_build=False):
        """Copy the six fields off `CommonAttentionMetadata`, and stamp this step's serial.

        No arithmetic, and no derivation -- the fold boundary that used to be derived here went away
        with the dense tail. The serial is one increment and it is what makes `impl._Step`'s cache
        safe to attach here (see `metadata._SERIAL`).
        """
        cam = common_attn_metadata
        if cam is None:
            raise RuntimeError(
                "vllm_ttkv: build() was called without `common_attn_metadata`, so there is nothing "
                "to copy from. Returning an empty metadata instead would move the failure into "
                "`impl._plan`, where the error no longer points at the builder.")
        _SERIAL[0] += 1
        return TTKVAttentionMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            query_start_loc_cpu=getattr(cam, "query_start_loc_cpu", None),
            seq_lens_cpu_upper_bound=getattr(cam, "seq_lens_cpu_upper_bound", None),
            positions=getattr(cam, "positions", None),
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            ttkv_serial=_SERIAL[0],
        )
