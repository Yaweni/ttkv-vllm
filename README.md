# ttkv-vllm — a compressed KV cache for vLLM, stored before RoPE

An out-of-tree vLLM attention backend that replaces the KV cache with a tensor-train store. Each
layer fits a small decoder to the prompt once; every token's key is then stored against that decoder
as a few coefficients, a bitmap, and a residual. Keys are stored **before** the rotation (pre-RoPE),
which is the ratio argument: a pre-rotary key is more compressible than a post-rotary one.

This repository is the vLLM integration: the format, the kernels, the plugin, the tests, and the
measurements — including the ones that went against us.

## Status, in plain English

**It works.** It serves `meta-llama/Llama-3.2-3B` through vLLM end to end on the stock scheduler,
paged cache, and sampling path. Greedy output matches the uncompressed arm for the first 12 of 16
tokens (the cache is lossy, so a divergence after a while is expected rather than a bug). The store is
byte-exact against the reference container. 98 tests pass in the kernel environment (two files skip
without vLLM) and 118 in the vLLM environment; `tests/test_bdd.py` needs the BDD runner and
`tests/test_kernel_krt.py` needs an NVRTC toolchain matching the driver — it passes in the kernel
environment but not in the vLLM one, whose bundled NVRTC emits PTX this driver rejects
(`cuModuleLoadData failed: 222 ... unsupported toolchain`).

**Accuracy is the strong result.** RULER on Llama-3.2-3B, at 4k and 16k context, in fp16 and 4-bit
storage. `safe` is the 4.53x allocation, `tight` the 6.42x one:

| RULER score | 4k fp16 | 4k 4-bit | 16k fp16 | 16k 4-bit |
| --- | --- | --- | --- | --- |
| dense (uncompressed) | 94.68 | 94.53 | 91.34 | 89.99 |
| **safe (4.53x)** | 93.91 | 93.36 | 88.78 | 85.61 |
| tight (6.42x) | 90.16 | 89.92 | 82.58 | 80.39 |
| *excluding the `cwe` task* | | | | |
| dense | 94.26 | 94.15 | 92.76 | 92.58 |
| **safe (4.53x)** | 93.48 | 93.01 | 91.58 | **90.64** |
| tight (6.42x) | 89.40 | 89.40 | 83.00 | 82.33 |

Read that as: at 4.53x compression `safe` stays within ~1 point of uncompressed at 4k, and within
~1.9 points at 16k once one outlier task is excluded. **4-bit storage costs it under a point**
(91.58 -> 90.64), which is the result we did not expect and the one worth the most. At 6.42x, `tight`
holds at 4k but drops ~10 points at 16k — a cliff, not a slope, and the honest limit of the current
tuning.

**Speed is the open problem, and it is real.** Decode is **3.8x slower** than stock at 1082 tokens, and
the gap WIDENS with context. Measured decode steps: 34.6 / 56.5 / 87.5 / 124.4 ms at 1082 / 4322 /
8642 / 14402 tokens, against stock's 19.4 / 21.2 / 21.9 / 23.5 — so 1.8x, 2.7x, 4.0x, and **5.3x
slower**. Ours grows at **6.7 us per cached token per step**, stock's at 0.31, and that slope, not any
fixed overhead, is what long context becomes. Prefill is ~26x slower at 1082 tokens and ~4.5x at 14402
(ours ~3.5 s fixed plus ~0.48 us a token; stock's ~0.16 us a token).

**And the widening slope is inherent to the format, not to our kernels.** Reconstructing a key costs
`r1*d` multiply-adds per cached row per step where a dense read costs `d`, so the arithmetic ratio is
about `r1/(1+G)` = 110/4 ~ 27x — and it does **not** amortise with batch size, because each request's
rows are reached by exactly one query token per step. It amortises only with many query tokens per
cached row — prefill and chunked prefill — because a row is reconstructed once and reused by every
query that attends it. But the dots are identical to dense's, so even there we never become cheaper on
arithmetic: the 27x penalty fades toward a rounding error, and what remains is the byte advantage
(912 B a token a layer against 4096), which wins only if the kernel is read-bound. So **parity is
reachable in the query-dense regime** where single-stream decode is 5.3x — and that is where the port
was weakest, since the prefill path is still the torch `attend_prefill`.

**The first piece of that work is done, and it is the largest single speed result here.** The
multi-query score kernel exists and is wired, so a chunked-prefill or prefix-cache continuation is
served rather than refused and `enable_chunked_prefill` / `enable_prefix_caching` are available at all.
The readout that goes with it is **114x cheaper per query** at Q=256 — one launch a chunk instead of
one per token, and the GQA-group padding gone, which alone was 4x the arithmetic. Together they took
the continuation's collapse at 4 and 8 concurrent 7.4k prompts from 225 s and 547 s to **58.6 s and
188.7 s: 3.84x and 2.90x**. Decode is unchanged, and that is the point rather than a caveat — a
continuation has `q_len` query tokens per stored row and an axis to amortise over; single-stream decode
has exactly one, so every decode number above stands.

**But the capacity does not convert into throughput, and that is the hardest number here.** At batch 8
we serve **47 tokens a second where stock serves 324** — and the gap WIDENS with batch, 2.0x at batch 1
and 6.9x at batch 8. Our throughput plateaus: batching buys 1.8x and then stops, against stock's 6.4x
over the same range. The reason is structural — the per-step work is issued per REQUEST, 28 read
layer-calls and 28 write layer-calls for each request in the batch, each carrying host readbacks that
drain the pipeline, where dense attention merges the batch into one kernel per layer. So the argument
that a smaller cache buys more concurrency and more concurrency buys more speed does **not** hold on
this port as it stands, and saying so is the point of this README rather than a footnote to it.
Getting that number took two instrument fixes (forcing generation length; widening the step gap past
the run-to-run variance of our own much longer prefill), both recorded in the measurements section,
because the first two answers the instrument gave were wrong in our favour.

**Capacity is not yet the win it should be.** The format is 4.53x, but vLLM sizes the cache from its
peak-memory profiling pass and the fit's temporary buffers get charged against us permanently, so the
realised capacity is 1.74x against stock — behind in-tree fp8 (8.77x). Fixing that accounting is the
highest-value single piece of work here.

**Where to start reading.** `vllm_ttkv/layout.py` (the format), `vllm_ttkv/rope_store.py` (the pre-RoPE
mechanism, the crux of the design), then the measurements section of this file — which records three
hypotheses that the numbers killed, including two claims that an earlier version of this file made and
that turned out to be wrong.

## Repository layout

| path | what it is |
| --- | --- |
| `vllm_ttkv/` | the plugin: layout, encode, decode, the Triton kernels, the vLLM adapter |
| `tests/` | the test suite, the benchmarks, and the two-arm serving harness |
| `tools/` | `calibrate_r1.py`, which produces the per-layer rank table the backend needs |
| `calibration/` | a shipped calibration artifact for Llama-3.2-3B at 16k |
| `features/` | the acceptance criteria as Gherkin |
| `codec/` | the compression codec and CUDA kernel this plugin stores into — vendored so the repo runs standalone |

Run the tests with `python -m pytest`; the benchmarks are `tests/bench_*.py` and are not collected by
pytest. Every environment variable the port uses is documented where it is read.

---

# vllm-ttkv — out-of-tree vLLM backend for a pre-rotary TT-KV cache

### The design, in depth

**Status of this section.** Finished and tested: `layout.py`, `reference.py`, and the `krt` score kernel
in both forms (`krt_triton.py`, and the shipped CUDA kernel through `tests/test_kernel_krt.py`). The
`NotImplementedError`s that remain are deliberate refusals — places where the alternative was a
plausible wrong answer — not unfinished work: the port serves end to end (see the status above).
Everything that touches vLLM's API is written to the real symbols. It is meant to be read, and to make
the decisions visible.

Read this file first, then `vllm_ttkv/layout.py` (the design), then `vllm_ttkv/rope_store.py`
(the crux).

---

## 1. Why this is a plugin, and not a `kv_cache_dtype`

Three facts about vLLM (verified at `vllm-project/vllm` @ `276fbcff`) shape everything here.

**Fact 1 — a backend cannot rotate, and never sees raw K.** `Attention.forward` is

```python
def forward(self, query, key, value, output_shape=None, output_dtype=None)
    # vllm/model_executor/layers/attention/attention.py:488-495
```

No `positions`, no `cos_sin_cache`. And every standard model rotates first
(`llama.py:228-233`): `q, k = self.rotary_emb(positions, q, k)` then `self.attn(q, k, v)`, with
the cache write inside the attention layer. So the RoPE-versus-cache-write **ordering is owned by
the model forward**, and cannot be changed from a backend class alone.

`rope_store.py` is the module that changes it; `patch.py` injects it. This is the part that
cannot be avoided, and it is why the honest unit of this change is "a plugin", not
"a new cache format".

**Fact 2 — no new `kv_cache_dtype` string is needed.** `CacheDType` is a closed `Literal`
(`vllm/config/cache.py:39-61`) with no registration hook, and `selector.py:126-131` rejects an
unknown string. But a *backend* can advertise an existing one:
`HpcAttentionBackend.supported_kv_cache_dtypes = ["auto", "bfloat16", "fp8_e4m3"]`. Reusing
`"auto"` and steering selection through the platform selector keeps this plugin out of
`vllm/config/` entirely — the difference between a plugin and a fork.

**Fact 3 — the supported pattern for owning the cache write already exists.** The HPC backend
established it: a model-forward module performs the write (`HpcRopeNorm`), the backend sets
`forward_includes_kv_cache_update = True` (`hpc_attn.py:288`) so `Attention.forward` skips its
generic `unified_kv_cache_update` (`attention.py:541-568`), and a metadata flag coordinates
(`hpc_kv_written`). HPC stores *after* rotation; we store *before* it. That single move is the
entire ratio argument.

---

## 2. The design decision that makes the layout simple

vLLM's page is `[num_blocks, block_size, num_head_slots, state_content_bytes]`. TT-KV is not
homogeneous per head: the tensor-train coefficients are per **token** and shared by all 8 KV
heads, while the residual and V rows are per (head, token).

The obvious encoding — one slot per head — replicates the coefficient block 8×, and at
`r1 ≈ 128…600` the coefficients are most of the K bytes. So instead the slot is **per token**
and the head axis is folded into its content:

```
num_head_slots      = 1
state_content_bytes = K_SLOT_BYTES + V_SLOT_BYTES   # ONE region: K slot then V slot

one token's blob    = [ K slot ][ V slot ]
one token's K slot  = [r1 coefficients][coefficient ruler][H residual rows]
one token's V slot  = [H lane rows: bitmap, packed codes, fp16 ruler]
```

That the two slots share one region is not a choice — it is what `state_content_bytes` means.
`AttentionSpec.state_content_size_bytes` returns it verbatim when a backend publishes one, and
otherwise falls back to `(head_size + head_size_v) * dtype_size`
(`v1/kv_cache_interface.py`), i.e. **dense K followed by dense V**. Publishing `K_SLOT_BYTES` alone
was a real bug and a silent one: every V write would land in the next token's coefficients while
staying inside the allocation, so nothing raises and the K store reads back as noise.
`layout.TTKLayout.v_offset` is the derived offset, and `README` §2's "key cache / value cache" split
was the wrong reading of this field.

**The decoder is per `(request, layer)`, not per layer.** Each request's prefill is its own TT-SVD
fit, so a batch holds as many decoders as it holds requests — same shape, different values. Sharing
one basis across a batch is not a rounding difference: a rank-`r1` basis spanning B sequences must
cover B times the key diversity, which is why the codec's pooled store gives each sequence its own
decoder instead of relabelling tokens. It is addressed by `block_table[row, 0]` — a request's first
physical block, fixed for its lifetime — because vLLM's `CommonAttentionMetadata` carries **no
request identifiers**, so `req_id` is not reachable from a kernel while a block id is.
Measured while building `tests/test_impl_decode.py`: borrowing another request's decoder moved the
answer by **2.19e-02**, with correct shapes and no error anywhere.

**Two open questions from the port plan dissolve under this:**

- *"does the paged block size divide the TT row size?"* — there is no row to divide. A slot is
  per token, so `block_size` counts tokens and the K page is exactly `block_size × K_SLOT_BYTES`.
- *"does the coefficient table need a second cache region?"* — no. A true sibling region needs
  `KVCacheGroupSpec` / `is_index_group_leader` (the DSA indexer pattern,
  `vllm/v1/attention/backends/mla/index_group.py`) or a multi-shape spec (`MambaSpec`). Folding
  the head axis makes the coefficient table part of the slot, so one `AttentionSpec` covers the
  layer.

`layout.py` implements this and is tested against the container's own billing — see
`tests/test_layout.py`.

---

## 3. File map

| file | status | what it carries |
|---|---|---|
| `vllm_ttkv/layout.py` | **done, tested** | slot geometry: what one cache slot holds, per-token packing, alignment. The block-layout answer. |
| `tests/test_layout.py` | **done, passing** | round-trips against the codec's own `PackedRows`; checks the layout's byte arithmetic against the container's `nbytes()` |
| `vllm_ttkv/reference.py` | **done, tested** | the reference pre-rotary decode — reconstruct, patch the residual, rotate each row by its own position, contract. The oracle `csrc/ttkv_krt` gets ported against. |
| `tests/test_reference.py` | **done, passing** | decode vs a second index order; the `_refuse_pre_storage` negative control; K-path fidelity |
| `tests/test_kernel_krt.py` | **passing on the GPU box** | the SHIPPED CUDA `krt` kernel (NVRTC, sm_86) against the reference decode: **rel err 4.4e-07**, plus a per-row-rotation control (0.398). Skips cleanly with no CUDA |
| `vllm_ttkv/krt_triton.py` | **done, tested** | the portable `krt`: reconstruct via `tl.dot`, patch the residual, rotate each row by its own position, contract. Reads the slot directly. Carries two hard-won notes -- see below |
| `tests/test_krt_triton.py` | **passing on the GPU box** | Triton `krt` vs `reference.raw_scores` over all three configs x three lengths: **1.2e-07..2.7e-07**; two negative controls; a rank sweep that guards a Triton compiler abort |
| `tests/bench_krt.py` | **run on the GPU box** | CUDA `krt` vs Triton `krt` on identical stores, rank swept 17..278, with agreement reported alongside the timings |
| `tests/test_e2e_forward.py` | **passing** | a REAL model (Qwen2-0.5B, d=64) driven through `make_cache`: a near-lossless store under both storage modes, the shippable-config byte comparison, greedy decode through a folded store, and criterion 9. The only layer that exercises the `Cache`/`DynamicLayer` contract and the un-rotate inside a live attention |
| `features/*.feature` + `tests/test_bdd.py` | **passing (10 scenarios)** | the section-5 acceptance criteria as executable Gherkin. Step bodies CALL the pytest layer rather than re-implementing it, so a scenario cannot pass by testing a second implementation of the thing it claims to test |
| `vllm_ttkv/__init__.py` | done | `register()` — the `vllm.general_plugins` entry point |
| `vllm_ttkv/backend.py` | real | `AttentionBackend`: real flags and dtype strings; `customize_spec` publishes the per-layer slot width from the calibration table, with a `head_dim` guard |
| `vllm_ttkv/rope_store.py` | **done** | two layout registries — live stores (forward-time truth) and the calibrated table (the only thing that can answer at cache-allocation time); `load_calibration()`; `decoder_for_layer()`; the per-request **decoder registry** keyed by `request_key(block_row)` = `block_table[row, 0]`; `ensure_store()`. **No write path** — see below |
| `vllm_ttkv/encode.py` | **done, tested** | the WRITE: the fit into a rank budget (capped, then zero-padded exactly), the frozen-decoder projection, the V lane encode, and the slot scatter. Imports the codec lazily, so the read path is usable where the codec is not packaged |
| `tests/test_encode.py` | **passing (12 tests)** | pad-is-exact at the byte level; the budget table below; pinned against `tools/measure_fold_gap.py` so the measured fold penalty keeps describing the shipped arithmetic; the pad-slot and wrong-layout refusals |
| `vllm_ttkv/select_triton.py` | **done, tested** | the FUSED per-token encode: lane selection, ruler quantisation, bitmap, MSB-first bit packing and the slot addressing, one kernel per store, written straight into the cache's own bytes. The torch path stays live as the fallback and as the definition of the format |
| `tests/test_encode_triton.py` | **passing on the GPU box** | the fused write vs the torch write with `torch.equal` on a whole slot tensor: all three configs, K and V, fp32 and bf16, plus the fit-path and non-power-of-two fallbacks, the two slot refusals, and the tie and all-zero cases -- where the two paths may legitimately differ in the BITMAP and are checked on the reconstruction instead |
| `tests/bench_write_path.py` | **run on the GPU box** | the write path alone, both arms, at engine shapes with the queue held deep. The tool that says what the write costs -- the engine's phase table cannot, because what it times is the queue draining |
| `tests/bench_step_plan.py` | **run on the GPU box** | the plan path at two queue depths, at both harness contexts: every readback's empty cost is 24-63 us and its deep cost is the queue (11-13 ms), while the O(seq_len) slot list is host work at any depth. The tool that separates "optimise this" from "do this less often" |
| `tools/calibrate_r1.py` | **done, run** | produces the per-layer `r1` artifact by calling `ttkv_codec.tt_svd` directly |
| `calibration/*.json` | shipped | e.g. `llama-3.2-3b-16384.json` — LongBench prose, per-layer max over 4 samples at 16k |
| `vllm_ttkv/decode.py` | **done, tested** | the attention MATH, with no vLLM import so it is testable where vLLM cannot be installed: the vLLM-cache → kernel angle-table translation, the block-table slot gather, the K/V blob split, the causal GQA prefill, and `check_row_counts` |
| `tests/test_decode_conventions.py` | **passing (8 tests)** | the vector-layout conventions, longhand. The vLLM half/straddle rotation vs `rope_apply`, absolute-position indexing, the partial-rotary and past-the-table refusals, `align_up` slot arithmetic, and vLLM's flat slot indexing |
| `vllm_ttkv/v_out_triton.py` | **done, tested** | the V readout: lane decode with the half-carry rank trick, GQA weighted reduction via `tl.dot` with the group axis padded to ≥16. No decoder and no rotation — V has no basis |
| `tests/test_v_out_triton.py` | **passing on the GPU box** | vs `reference.dense_v_from_slots` over all three configs: **1.45e-07..1.98e-07**; G=7 (padded to 16) at 1.8e-07; a per-row-ruler control; a bitmap-permutation control |
| `vllm_ttkv/impl.py` | **done** | WRITE + READ. The write because this is the only place in the step holding raw K, the slot mapping and the per-request split; the read as an adapter (decode/prefill split from `query_start_loc`, per-request decoder resolution). Writes into `output` — the op mutates it, so a `return` is a no-op. Twelve refusals where a plausible wrong answer was the alternative |
| `tests/test_impl_decode.py` | **passing on the GPU box (32 tests)** | the real `forward` END TO END: raw K in → fit → slots → frozen projection → attention out (**2.7e-07** read-vs-write, store vs the model's own keys 0.0297 K / 0.2822 V); plus single and ragged batched decode (**3.0e-07..3.3e-07**), longhand causal GQA prefill (1.7e-07), the guard set, and the plan path's two ends: the device slot build agreeing with the host build on a row with non-contiguous blocks and padding, and the step refusing a padded slot for the whole step at once |
| `vllm_ttkv/metadata.py` | done | the six `CommonAttentionMetadata` fields the impl relies on, and NO `kv_written` handshake: vLLM hands one metadata object to every layer of a group, so a per-step flag cannot record that a particular LAYER wrote -- that is what the popped stash is for. The tail field is deleted |
| `vllm_ttkv/patch.py` | done | injects the raw-K capture by rebinding `forward` on the model's own rotary module instance, and `stash_raw_k` is the seam the writer reads (it carries the model's RoPE table too, because the attention layer has no `rotary_emb` of its own). Distinguishes "arch absent" from "import broken" instead of reporting the second as the first. It also settles WHERE the write lives: not here, because this file has no slot mapping -- the capture is here, the write is in `impl` |
| `pyproject.toml` | done | the `vllm.general_plugins` entry point; vLLM deliberately not a dependency |

### What is left, exactly

**The run works.** `run_vllm_ttkv.py --arm ttkv` serves Llama-3.2-3B through this backend end to end,
and `--compare` measures it against stock vLLM (see §4 for the numbers and for the capacity caveat).
What remains is a short, ordered list, and none of it is plumbing:

1. **The READ page** (see §4). The write is fused and measured (5.84x on the write path) and the plan
   path is settled -- one queue drain per metadata object, slot lists built on the device, and an A/B
   that puts it at -4.7% (1082 tokens) and -1.2% (4322) with the O(1)-in-context property as the real
   reason. What is left is the read. At decode shapes the TT-KV work per layer is now ~0.25 ms for the
   write, ~0.2-0.29 for the projection and ~0.6-0.8 for the read (`krt`, gather, split, cos/sin,
   softmax, `v_out`), so the read is the largest single item and `krt` dominates it at ~0.22-0.36 ms
   a layer-call that does not move with context. Decode is 79 ms/step against the stock arm's 21 (both
   measured the same way, below), of which ~21 ms is the model's own memory-bound weight streaming and
   cannot be moved: the read path is what stands between the port and parity.
2. **Transient memory.** The format is 4.53x and the capacity is 1.74x, because vLLM budgets the
   cache from the peak memory of its profiling pass and this impl's peak is high -- the fit's SVD
   workspace, per-request gathers, Triton workspace, `.contiguous()` copies. This is the difference
   between the ratio being true and the ratio being *useful*, and it is a footprint job.
3. **Multi-query `krt` — done, and the continuation is 3.84x faster than the version it replaced.**
   A continuation chunk (`seq_len > q_len`) used to be REFUSED rather than silently attended over the
   chunk alone. It is now served by `_prefill_continuation` over `krt_multi_from_slots` +
   `v_out_multi_from_slots`: one reconstruction serves every query in a block, and the readout is one
   launch a chunk instead of one per query, with the group padding gone. So
   `enable_chunked_prefill` and `enable_prefix_caching` are both available -- `--chunked` and
   `--prefix-cache` in the harness -- and the prefill collapse at N=4 and N=8 went from 225.3 s and
   546.7 s to 58.6 s and 188.7 s (3.84x and 2.90x). What is left on this path is the *score*
   reconstruction, still rebuilt once per query block, and the materialised
   `[HQ, q_len, seq_len]` probabilities.
4. **TTFT, which for a short generation is most of the wall clock -- and is the WORSE of the two
   ratios.** Measured by isolating it (`--new-tokens 1` makes the run's wall time the prompt's forward
   pass plus one step):

   | | prefill, 1082 tokens | steady-state decode |
   |---|---|---|
   | stock vLLM (FLASH_ATTN, bf16) | ~0.15 s (~7.2k tok/s) | ~21 ms/step |
   | TT-KV (`safe`) | ~4.0 s (~270 tok/s) | ~79 ms/step |
   | | **26x slower** | **3.8x slower** |

   The prefill is 28 TT-SVD fits at ~138 ms each, one per layer per request, and it is 3.86 s of the
   5.26 s run. Options are the ones already measured -- chunking makes it worse (4 chunks of 4096 cost
   3.3x a single 16k fit), so the real levers are the fit itself and prefix caching (on by default,
   and it makes the cost per unique prompt rather than per request).
5. **Varlen confirmation.** `impl._plan` splits a mixed batch and verifies the decode ordering, but
   the tiled kernel reading a varlen store has not been exercised on a batch that mixes a prefill
   with several decodes. Continuous batching is the reason the store is per request, so this is
   worth an explicit test rather than an assumption.

The write path landed, and it settled the contradiction between `patch.py` and `rope_store.py`:
`patch.py` said the IMPL writes (it is the first point in the step holding both raw K and the slot
mapping), while `rope_store.TTKVRopeStore.forward` claimed to own `rotary_emb` and also took a slot
mapping. `patch.py` was right — the wrapper intercepts `rotary_emb`, which has no slot mapping, so
nothing using one could be called from there. So:

- `rope_store.TTKVRopeStore.forward` (and its `_split` / `_encode_raw_k` / `_rotate` stubs) are
  **deleted**, along with the second writer they implied;
- `vllm_ttkv/encode.py` holds the slot arithmetic;
- `impl._write_store` orchestrates and is the only writer.

**The fold schedule is now a pure function of the batch.** A request with no registered decoder is
on its first chunk (fit, and keep the decoder); one with a decoder is past it (project). That is the
same registry the read path uses to find the basis, so the two halves cannot disagree about which
phase a request is in — and there is no per-request token counter to survive preemption, recompute
or reordering.

**The kernel decision, now measured.** `krt` exists twice: the shipped CUDA C++ kernel, and a Triton
port. They were benchmarked on identical stores at n=16384 (`tests/bench_krt.py`), because the only
real question was whether portability costs speed:

| r1 | CUDA ms | Triton ms | tri/cuda |
|---|---|---|---|
| 17 | 0.479 | 0.714 | 1.49 |
| 62 | 0.816 | 0.859 | 1.05 |
| 119 | 1.241 | 1.069 | 0.86 |
| 278 | 2.518 | 1.609 | **0.64** |

Triton wins from roughly r1 ~ 90 upward, and that is where real layers live (measured r1 93..616,
mean 326 on prose at 16k). The reason is in the CUDA kernel's own comment: its reconstruct loop is
*instruction*-bound scalar FMA, whereas `tl.dot` maps to fused MMAs, so Triton's advantage grows with
rank. Below r1 ~ 90 the CUDA kernel is ahead, which is the regime a short prefix sits in.

So Triton is the default for both reasons at once -- it installs without a toolchain, and it is
faster where it matters. The CUDA kernel stays as the opt-in path for short prefixes. All three
implementations agree on one store: reference, CUDA (5.0e-07) and Triton (1.7e-07), CUDA vs Triton
within 5.6e-07.

A benchmark at ONE rank would have said the opposite. My first Triton number, 0.831 ms at r1=17, sits
in the regime where Triton is worst and is an order of magnitude below any real layer's rank.

There is still **no `csrc/`**, and the benchmark removes most of the reason to build one. If an AOT
extension is wanted later, the bodies in `ttkv_kernels.cu` are plain CUDA C++ (shared memory,
`__syncthreads`, fp32 FMA, no `wmma`), so what changes is the *delivery*, not the maths.

**How the layout generalises, and what does not.** Measured with the plugin's own layout code
across d in {64, 128, 256, 512} at a fixed KV WIDTH (H*d) and a fixed lane *fraction*, the joint
ratio moves only 4.27x..4.57x. It generalises because the bitmap is `H*(d/8)` and the residual
lanes are `H*d*res_frac` -- both functions of `H*d`, not of the split. The invariant is the KV
width, which is NOT the model's `hidden_size` (Gemma-2-9B: 16 heads x 256 = 4096 q-dim against a
3584 hidden; Qwen3-32B: 64 heads x 128 against 5120).

What does NOT generalise is the CONFIG. `res_k` comes from `res_frac`, a fraction of `d`, so it
follows the geometry. `v_lanes` (`k_lanes` in the codec) is an ABSOLUTE count tuned at d=128, so
it does not: at d=64 the shipped 81 lanes do not exist (gpt-oss-120b/20b, Llama-3.2-1B), and at
d=256 the same 81 lanes are a third of the row rather than two thirds, which reads as a HIGHER
ratio while being a coarser approximation (Gemma-2-2B/9B). `TTKLayout` rejects the first case and
`tests/test_layout.py` documents the second.

**Where the 3.5x in the e2e fixture came from, and why it is not the campaign's 5-7x.**
`tests/bench_ratio_vs_context.py --real` captures real K and V off `k_proj`/`v_proj` from cached
weights and sweeps length x allocation x storage at a fixed geometry. It reproduces the fixture
exactly, which is what makes the rest of the table usable:

| real weights | alloc | n=48 | 512 | 4096 | 16384 |
|---|---|---|---|---|---|
| Qwen2-0.5B (KV width 128) | campaign4x | **3.53x** | 4.13 | 4.27 | 4.31 |
| Qwen2-0.5B | safe | 3.62 | 4.27 | 4.41 | 4.46 |
| Qwen2-0.5B | tight | 4.60 | 5.69 | 5.96 | 6.03 |
| Llama-3.2-3B (KV width 1024) | campaign4x | 3.19 | 4.29 | 4.75 | 4.84 |
| Llama-3.2-3B | safe | 3.23 | 4.37 | 4.84 | 4.93 |
| Llama-3.2-3B | tight | 4.09 | 6.07 | 7.04 | 7.23 |

`storage="pre"` throughout. The fixture reads **3.514x**; the same model, the same n=48, the same
allocation and the same storage mode read **3.53x** here. Nothing was wrong with the fixture except
that it was never a release point. Three independent things were true at once:

1. **Its allocation is `campaign4x`.** The identity is asserted in the run (not assumed): the
   fixture's `res_frac=0.5, res_bits=4, lanes=0.5*d, vbits=4` re-expresses to exactly `campaign4x`'s
   `64 lanes @4b`. That is the WEAKEST of the three, archived, and its own label is only 4.252x.
2. **n=48, not 16384.** The decoder is `r1*H*d` bytes per LAYER, shared by every token in it, so its
   share of the bill is `r1*H*d/n`. Measured `decoder%` for Llama/safe: **36.2% at n=48, 0.3% at
   16384**. That single term is most of the gap.
3. **KV width 128 against 1024.** The coefficient row is per-TOKEN and shared across heads, so its
   share is `r1/(4*H*d)`. Measured `coeff%` at 16384: **12.9% for Qwen2, 5.2% for Llama**. The
   load-bearing quantity is the KV width, not `d` -- the same finding `tests/test_layout.py` records.

**The `tight` column is the control on the 5-7x claim.** At 16384 on real Llama weights it reads
**7.23x**, against the published pre-rotary label of 7.42x (`tight`); `safe` reads 4.93x against
5.08x. Random token ids, so these are a slightly pessimistic floor -- prose and RULER compress
better. The campaign range is reproduced from the shipped code, on real weights, from outside the
ledger that produced the labels.

**And it is why the `POST_ROPE_RATIO_PACKET.md` rows peak at ~6x.** Same weights, same n=16384, same
allocation, only `storage` varying: `safe` 4.33x post vs **4.93x** pre, `tight` 6.01x post vs
**7.23x** pre. The 6.x figures are the *post-rotary* arm. They are not a shortfall against 7x; they
are the cost this port exists to avoid, and the packet's own prose/RULER rows (4.49/6.23 RULER,
3.64/4.71 prose) sit below this because a real corpus is harder than random tokens.

**One number worth more than the ratios: the r1 spread.** At n=16384 pre-rotary on Llama the
per-layer `r1` runs **6 .. 42 .. 166** -- a 28x spread across 28 layers of a single prompt. That is
the F4 constraint stated as a measurement: `r1` is data-dependent, varies by layer, and is uncapped,
so a uniform page cannot hold the store. The calibration table in `backend.py` exists for this.

**The baseline: what vLLM's own KV compression gets on this GPU.** `tests/bench_vllm_baseline.py`
runs **unmodified vLLM** -- TT-KV is not involved -- so nothing here can flatter the plugin. This is
the number the port has to beat, on the same GPU, model and context length:

`meta-llama/Llama-3.2-3B` (H=8, d=128, KV width 1024), `max_model_len=16384`, prompt 16256 tokens,
`gpu_memory_utilization=0.90`, one arm per process, A4000 16 GB:

| `kv_cache_dtype` | attention backend | KV tokens | max concurrency | TTFT | decode |
| --- | --- | --- | --- | --- | --- |
| `auto` (bf16) | FLASH_ATTN | 69,376 | 4.23x | 2720 ms | 43.7 tok/s |
| `fp8` | FLASHINFER | 143,712 | **8.77x** | 2548 ms | **48.0 tok/s** |
| `turboquant_k3v4_nc` | TURBOQUANT | 203,680 | **12.43x** | 2771 ms | **34.0 tok/s** |

`concurrency` is vLLM's own `Maximum concurrency for 16,384 tokens per request`, read out of the
engine log. It is the bytes axis and needs no accounting of ours: vLLM sizes the cache at init, so
the figure is a direct inverse of bytes/token.

Three things fall out of this that change how the port should be pitched:

1. **The number to beat is 2.94x, not 2x.** `turboquant_k3v4_nc` gives 12.43/4.23 = **2.94x** the
   capacity of bf16 -- comfortably past fp8 -- and it is the in-tree analogue of what TT-KV does.
   (Theoretical for K3V4 is ~4.6x; the rest goes to per-token/per-head scales and block padding.)
2. **fp8 is free, and that is unusual.** It is simultaneously 2.07x smaller AND 4% *faster*
   (48.0 vs 43.7 tok/s), because FlashInfer's fp8 attention reads half the bytes and this workload
   is memory-bound at 16k. So fp8 is not merely a compression baseline, it is the speed baseline,
   and an implementation that is slower than bf16 cannot justify itself by ratio alone.
3. **TTFT is flat across all three** (2548..2771 ms), so prefill at 16k is compute-bound rather
   than cache-size-bound. That is the cost side of the pre-rotary trade: our decode pays a rotation
   per stored row, and this table says nothing about prefill either way.

**So the target is stated precisely:** match or beat 34.0 tok/s at 12.43x concurrency, or show that
5-7x at matched quality is worth a decode regression. `turboquant_k3v4_nc` is the arm to compare
against -- it is 3-bit K / 4-bit V, which is the same description as the `tight` allocation.

Caveats, all of which are the reason the numbers are quoted with their settings: one warmup plus one
measured run per arm; random token ids; `decode_tps` derived as `(out-1)/(T_long - T_first)`, which
is slightly conservative; and each dtype selects its own attention backend (bf16 -> FLASH_ATTN,
fp8 -> FLASHINFER, turboquant -> TURBOQUANT) because that is what vLLM naturally does. Forcing one
backend across all arms was tried and rejected: a benchmark whose arms differ in the *cache format*
has to be honest about anything else that also differs, and pinning a backend that vLLM would not
have chosen is its own distortion. The first fp8 run also pays FlashInfer's JIT compile
(`init_s` 37-53 s vs 46 s here, and minutes when the cache is cold), which is why `init_s` is
reported separately and never used as a throughput figure.

**The prefill fit is the dominant cost, and it is NOT cheap.** Measured `tt_svd` on the real
`[n, d*H]` unfolding, median of 3, A4000:

| n | GPU ms/layer | CPU ms/layer |
|---|---|---|
| 1024 | 101.2 | — |
| 4096 | 108.3 | — |
| 16384 | 131.8 | 301.6 |
| 32768 | 185.1 | 544.5 |

**The cost is nearly FLAT in `n`.** It is set by `d*H`, not by context length — confirmed by holding
n=16384 and sweeping the other axis: `d*H`=256 → 16.8 ms, 512 → 39.8 ms, 1024 → 137.6 ms, i.e.
roughly quadratic in the KV width. That is the shape of an SVD of an `n x (d*H)` matrix with
`n >> d*H`.

Totals for a 28-layer model at 16k, **measured, not extrapolated**:

| | fit cost | TTFT (baseline 2.72 s) |
|---|---|---|
| one-shot over the whole prompt | **3.7 s** | ~6.4 s |
| four chunks of 4096 (the chunked-ratio path) | **12.2 s** | ~14.9 s |

**This retracts an earlier claim in this file.** Chunked prefill was described as buying the
`POST_ROPE_RATIO_PACKET.md` chunking ratio (+2.75% / +3.40% on prose) for free. The ratio is real,
but the cost is not free: chunking multiplies the fit by **3.3x**, because each chunk pays a
`d*H`-bound SVD regardless of how few tokens it holds. Chunking the *fit* is the wrong move; fitting
once is cheaper than fitting four times even though the four fit less data.

Encoding with a FROZEN decoder, by contrast, is genuinely free: projecting one token onto a frozen
`[r1, d*H]` decoder costs **0.009 ms** (launch-dominated at r1=103, d*H=1024). So the whole cost
sits in the prefill fit, and the streaming/append path costs nothing measurable.

| | fit cost | decoders/seq | rank quality at chunk=2048 |
|---|---|---|---|
| fold (fit chunk 1, append the rest) | 1x | 1 | ~14% under-ranked |
| rechunk (fit every chunk) | 3.3x | N (sized at allocation) | full |
| one-shot at end of prefill | 1x | 1 | full, but needs a dense K buffer |

The rank gap is the reason to prefer a LARGE first chunk: `r1` per-layer mean on prose grows
59.7 (n=512) -> 87.3 (2048) -> 92.9 (8192) -> 100.7 (16384). A decoder fitted on 2048 tokens is 14%
short of what 16k wants; fitted on 512 it is 40% short. **Fold is the right default, but only if the
first chunk is large** -- `max_num_batched_tokens` becomes a correctness-relevant knob, not a
throughput one.

**But rank shortfall is only a proxy, so it was measured properly.** `tools/measure_fold_gap.py`
fits a decoder on the first `c` tokens, projects the WHOLE 16k block onto it, and compares against a
one-shot fit -- both arms through the shipped codec, using `PackedKLayer.decoder_f()` for the frozen
decoder rather than a reimplementation of the residual scheme:

| first chunk | r1 of the fit | rel-F error | vs one-shot | mean row cosine |
|---|---|---|---|---|
| one-shot | 100.7 | 0.06003 | 1.00x | 0.9981 |
| 512 | 59.7 | 0.09670 | **1.61x** | 0.9952 |
| 2048 | 87.3 | 0.07072 | **1.18x** | 0.9974 |
| 4096 | 90.7 | 0.06766 | **1.13x** | 0.9976 |
| 8192 | 92.9 | 0.06476 | 1.08x | 0.9978 |

**The fold penalty is far smaller than the rank gap predicts, because the residual patch absorbs
it.** A 13% rank shortfall at chunk=2048 costs 18% of error, not 13% of everything -- the fixed-lane
residual is doing exactly the job it was designed for, and this is the first evidence that it holds
up under a decoder that is deliberately under-ranked.

And **a larger first chunk is nearly free**, because the fit is flat in `n` (101 ms/layer at n=1024
vs 132 ms at 16384). So the accuracy can be bought back at almost no cost: chunk 4096 costs 7 ms
more per layer than chunk 1024 and cuts the fold penalty from 1.61x to 1.13x.

### The knobs, settled on this hardware

| knob | setting | why, measured |
|---|---|---|
| fit frequency | **once**, on the first chunk | 3.7 s vs 12.2 s for four chunks (3.3x), for ~3% of ratio |
| first chunk (`max_num_batched_tokens`) | **>= 4096** (2048 floor) | 1.13x vs 1.18x vs 1.61x error; the fit is flat in `n`, so this is nearly free |
| dense tail after the fit | **none** -- encode each decode token immediately | the ENCODE is 0.009 ms/token, but the two calls around it are ~0.61 ms a layer-token of FIXED host time: ~17 ms of a ~64 ms decode step. Deferred and batched across layers it is ~0.6 ms/step -- see "where the write's fixed cost goes" in §4 |
| allocation | **tight** 6.42x, **safe** 4.53x | calibrated, per-layer, prose 16k |
| per-layer `r1` | keep | worth 4% of bytes, and exactly-safe instead of safe-and-wasteful |

**`fold_T` disappears as a knob.** In the torch container, folding existed because appending one row
re-quantised and re-packed the whole store (~20 ms/layer/token, 86% of the decode step). In vLLM
there is no re-pack -- a row is written to its slot through `slot_mapping` -- so the store can be
appended to one token at a time. The dense tail then exists only for the first chunk, before the fit,
and **the ragged-tail varlen problem goes away with it**. That was the hardest integration issue in
the previous design and it is dissolved by a measurement, not by a workaround.

**BUT THE MEASUREMENT THAT DISSOLVED IT COUNTED THE WRONG CALL, and §4 now has the corrected one.**
0.009 ms is `encode_frozen`'s arithmetic -- one token projected against the frozen decoder. The two
calls the step actually makes, `encode_frozen` and `pack_and_write`, measure **0.26 and 0.35 ms a
layer-call at one token**, both flat in tokens and both dominated by host issue time rather than by
the GPU. That is ~17 ms of a ~64 ms decode step, 28 calls' worth of fixed overhead for work whose
arithmetic is microseconds. So "there is nothing to batch up" is true of the arithmetic and false of
the call: at decode each layer has exactly one token, and one token per call is the worst possible
amortisation. Whether that becomes a deferred/one-token-lagging design is now a decision with a
number behind it rather than an assumption.

### The bar, on the calibrated numbers

| | mean ratio (prose, 16k) |
|---|---|
| bf16 | 1.00x |
| fp8 | 2.07x |
| `turboquant_k3v4_nc` | 2.94x |
| **TT-KV `safe`** | **4.53x** |
| **TT-KV `tight`** | **6.42x** |

So `tight` carries **2.18x** the KV capacity of the strongest in-tree baseline, and `safe` **1.54x**
-- with `safe` being the read-rotated, no-measurable-loss default. What is still unproven is the
accuracy side: `tight` is documented as costing 8-12 points, and the fold adds 1.13x of
reconstruction error on top. **A RULER run at matched allocation is the next measurement**, because
none of the numbers above say anything about task accuracy.

---

## 4. Running the tested part

```bash
# everything, all three layers
python -m pytest tests/ -v

# or one layer at a time
python -m pytest tests/test_layout.py tests/test_reference.py   # unit, CPU only, no model
python -m pytest tests/test_e2e_forward.py                      # e2e, real model, CPU or GPU
python -m pytest tests/test_bdd.py                              # the Gherkin scenarios
python -m pytest tests/test_krt_triton.py tests/test_kernel_krt.py   # kernels, needs CUDA
python -m pytest tests/test_v_out_triton.py                     # the V readout, needs CUDA
python -m pytest tests/test_encode.py -s                        # the WRITE path, needs CUDA
python -m pytest tests/test_impl_decode.py -s                   # the impl, read AND write, needs vLLM + CUDA
python -m pytest tests/test_decode_conventions.py               # the vector-layout conventions, CPU
python -m pytest tests/test_spec_sizing.py                      # per-layer cache sizing, needs vLLM (CPU ok)
python -m pytest tests/test_patch_capture.py                    # the raw-K capture and the stash contract
python -m pytest tests/test_encode_triton.py                    # the fused write vs the torch write, bytes

# benchmarks are not tests and are not collected
python tests/bench_krt.py                                       # CUDA vs Triton, rank swept
python tests/bench_write_path.py                                # the write path, both arms, engine shapes
python tests/bench_step_plan.py                                 # the plan path: work vs wait, two queue depths
python tests/bench_ratio_vs_context.py                          # ratio vs length, synthetic
python tests/bench_ratio_vs_context.py --real                   # the same, on cached weights
python tests/smoke_register_vllm.py                             # plugin resolves against vLLM
python tests/bench_vllm_baseline.py --dtype fp8                 # unmodified vLLM, one arm

# the real thing: serve, then compare two arms
export VLLM_TTKV_CALIBRATION=$PWD/calibration/llama-3.2-3b-16384.json
export VLLM_TTKV_CONFIG=safe
export PYTHONPATH=/opt/ttkv/integration          # the codec, for the encode
python tests/run_vllm_ttkv.py --arm ttkv     --out /tmp/ttkv.json
python tests/run_vllm_ttkv.py --arm baseline --out /tmp/base.json
python tests/run_vllm_ttkv.py --compare /tmp/ttkv.json /tmp/base.json
```

Two of those need vLLM and run in **its** venv, not the kernel one -- `test_impl_decode.py` and
`smoke_register_vllm.py`. See `REMOTE_ACCESS.md` §1b for why the venvs are separate and for the
three environment workarounds (`torchcodec` removal, `VLLM_USE_FLASHINFER_SAMPLER=0`,
`HF_HUB_OFFLINE=1`).

Current state, measured: **74 passed / 2 skipped** in the kernel venv (`test_impl_decode` and
`test_spec_sizing` skip there -- they need the real `AttentionImpl` base class / vLLM) and **93
passed** in the vLLM venv (`--ignore=tests/test_bdd.py --ignore=tests/test_kernel_krt.py`). The two
ignored files fail for environment reasons that have nothing to do with this port: `pytest_bdd` is
not installed in vLLM's venv, and `test_kernel_krt.py` drives the **hand-written CUDA** kernel, whose
NVRTC (12.9, from the cu129 wheel) emits PTX the box's 552 driver refuses to load --
`cuModuleLoadData failed: 222 the provided toolchain is unsupported`. That test passes in the kernel
venv (NVRTC 12.4) and the shipped read path is Triton, so it is a packaging note rather than a
defect.

### The first real run, measured

`run_vllm_ttkv.py` on Llama-3.2-3B, 1082-token prompt, 16 greedy tokens, 16k `max_model_len`,
`enforce_eager=True`, on an Ampere consumer card (sm_86):

| arm | KV cache blocks | KV tokens cached | bytes/token (all layers) | gen s | first divergence |
|---|---|---|---|---|---|
| stock vLLM (FLASH_ATTN, bf16) | 4159 | 66,544 | 114,688 | 0.46 | -- |
| TT-KV (`safe`) | 7244 | 115,904 | 25,328 | 5.32 | token 12 |

`12/16` greedy tokens identical, both arms fluent, and the generated text is unchanged by the
performance work below -- which is the check that it was performance work. `5.32` is the run with the
fused encode in place; before it the same run was 5.50 s, and the difference is the decode (the
prefill is the TT-SVD fit either way).

**Those two numbers are the sum of two very different halves, so they are measured separately.**
`--new-tokens 1` makes the run's wall time the prompt's forward pass plus one step, and subtracting a
16-token run gives each half:

| | prefill (1082 tokens) | decode (steady state) |
|---|---|---|
| stock vLLM (FLASH_ATTN, bf16) | 0.18 s total, i.e. ~0.15 s (~7.2k tok/s) | (0.49 - 0.18) / 15 = ~21 ms/step |
| TT-KV (`safe`) | 4.08 s total, i.e. ~4.00 s (~270 tok/s) | (5.26 - 4.08) / 15 = ~79 ms/step |
| | **26x slower** | **3.8x slower** |

So the prefill is the *worse* ratio of the two, it is the single largest item in the integration, and
it is what a user waits for: 4 s of time-to-first-token against the stock arm's 0.15 s. The decode is
what the fused write below improved, and it is still 3.8x off -- but the honest read of this table is
that the fit, not the decode, is where the largest absolute cost sits (§3, item 4).

**The format ratio is exactly what the calibration predicted: 114,688 / 25,328 = 4.53x**, and that
is now measured through vLLM's own allocator (`pack_spec_for_layer` -> `page_size_bytes` per layer:
904 B/token/layer against the stock 4096), not through a synthetic harness.

**The capacity ratio is not, and the reason is worth reading.** The two arms were handed different
cache budgets:

```
ttkv      Available KV cache memory: 2.73 GiB   GPU KV cache size: 115,904 tokens (7.07x concurrency)
baseline  Available KV cache memory: 7.11 GiB   GPU KV cache size:  66,544 tokens (4.06x concurrency)
```

vLLM sizes the cache from the peak memory it observes during its profiling forward, and the TT-KV
arm's peak is higher because of transient host/device allocations the impl makes on that pass -- the
fit's SVD workspace, the per-request gathers, Triton's own workspace, and the `.contiguous()` copies
`split_blob` documents. So the arm gets 2.6x less memory and still caches 1.74x more tokens. The
4.53x is therefore *earned and observable*, and turning it into capacity is a memory-footprint job
rather than a format job -- which is a concrete, measured next step instead of a hope.

### The decode, profiled, and what it cost

`VLLM_TTKV_PROFILE=1` accumulates host time per phase and prints a table at exit (`vllm_ttkv/prof.py`,
shared so that `encode.py` can add its own phases). The first profile was unambiguous, and not about
kernels at all:

```
ms per layer-call            phase                 what it was paying for
2.00                         write.pack_and_write  the slot ids read back TWICE
1.34                         write.project
1.08                         decode_total          gather + the angle table
0.66                         plan                  cu_seqlens + seq_lens
0.24                         block_slots           its own idx.max()
```

**Every one of those values is layer-independent, and all of them were being recomputed for all 28
layers.** vLLM hands one metadata object to a group of layers (measured: 448 layer-calls over 112
metadata objects -- 4 layers per object, 7 objects per step), so that is ~6 device readbacks per
layer-call, ~150 per step, each draining the pipeline. `impl._Step` now computes and validates the
per-step work ONCE, keyed on a serial that `metadata.build` stamps so a cached bundle cannot outlive
its step:

```
phase                ms/call   before   after      readbacks per layer-call:  4.8 -> 0.5
plan                   0.165    0.66
decode_total           0.603    1.08
block_slots            0.024    0.24
cos_sin                0.025    0.16
gather                 0.030    0.099
```

The validation MOVED rather than disappearing: `_Step` reads the block table and the slot mapping
back once, which is what makes their bounds checkable with no further readback at all -- the check
becomes arithmetic on host values. `gather_slots` and `write_slots` still refuse by default and take
`check=False` only from `_Step`.

That took the decode from **167 ms/step to 113 ms/step** (steady state ~8.8 tok/s, against the stock
arm's ~35 tok/s), and the 16-token run from 6.35 s to 5.50 s. The finer profile then says where the
remainder is, and it is one thing:

```
phase                ms/call   launches?   what it is
pack.v                 0.944   ~20         the codec's V lane selection + bit packing, via torch
proj.resid             0.728   ~20         topk + mask scatter + the residual's bit packing
krt                    0.370    1          the Triton K kernel (one launch, so it is the cheap one)
pack.k                 0.314   ~10         the K slot's bytes
proj.matmul            0.192    1
proj.quant             0.104   ~5
pack.scatter           0.051    4
everything else        0.170   ~10         split, gather, cos/sin, softmax, v_out
```

**~70-80 tiny kernel launches per layer per token, each ~30 us of host time.** The encode path is
vectorised already (`PackedRows.build` has no Python loop and explicitly documents avoiding `.item()`
syncs), so the cost is not a hot loop -- it is that each op is a separate launch of a kernel too
small to hide its own launch. That makes the next step a specific, bounded job rather than a hunt:
**fuse the two per-token encodes into one Triton kernel each** (`encode_v`, and projection + ruler +
residual + pack for `encode_k`), which should take 2.4 ms/layer down to ~0.2 and the decode step to
~35-40 ms -- at or below the stock arm's 29 ms, since we would then be launching 4 kernels per layer
instead of ~80. Both kernels need a byte-exactness test against `pc.PackedRows.build` (the existing
`test_encode.py` pins the shipped encode against `tools/measure_fold_gap.py` in exactly this spirit),
because a bit-packing or `topk` tie mismatch would be a silent accuracy difference rather than a crash.

### The fused write, done and measured

`select_triton.py` is that fusion, and it landed the write at **5.94x**. A store is now one kernel
plus one scatter: per row the kernel loads the values, selects exactly `k` lanes, quantises, builds
the bitmap, packs the MSB-first code stream and writes all three fields at the slot's own address --
where the torch path assembles those fields in a slot buffer and then a Python loop over heads copies
them into the cache.

**The contract is byte equality, not accuracy**, because a bit-packing or a `topk`-tie mismatch is a
silent difference rather than a crash. `tests/test_encode_triton.py` therefore compares against the
OTHER IMPLEMENTATION with `torch.equal` on a whole slot tensor, all three configs, K and V, fp32 and
bf16 -- and asserts that the fused branch actually ran, since otherwise the comparison would be true
by construction.

Three things that only showed up because it was checked that way:

  * **The ruler is the row's MAX magnitude, not the selection threshold.** The threshold `t` that
    keeps exactly `k` lanes is the SMALLEST kept magnitude; `amax` over the kept lanes is the largest,
    and since every lane above the threshold is kept those differ. One line, and it moved every code.
  * **`sc` is a reciprocal multiply, not a division.** The codec divides by a Python scalar, and
    torch's scalar division is `x * float32(1/3)`: measured over 2**20 random fp32 pairs, that differs
    from the correctly-rounded quotient in 349427 of them and matches the reciprocal multiply in all
    of them. So `sc` is one ulp off about a third of the time, and the kernel has to reproduce THAT,
    or the two writers disagree on any value near a half-integer. The fp32 fixtures passed and the
    bf16 ones failed -- bf16 magnitudes are coarse enough to sit on halves -- which is how it was
    found. `v/sc` IS a tensor-over-tensor division and is correctly rounded, so the kernel uses
    `div_rn` there and `rint` (half to even) for the rounding, both matching torch exactly.
  * **The selection can be exact without matching `torch.topk`'s tie order**, because `topk` documents
    no tie order at all. The kernel breaks a tie by the lowest lane index; with distinct magnitudes
    the k largest is a unique set and the bytes match, and on tie-heavy rows the two may differ in the
    BITMAP while decoding to the same values. Both cases are tested, the second on the reconstruction
    and the lane count rather than on bytes.

Measuring it then needed a second instrument, and this is the methodological half: **the engine's
phase table cannot show a win on this path.** Once a step is deep enough that the GPU queue never
drains, a phase's host wall-clock is the queue draining, not the work submitted -- the fused write
measures 0.34 ms a layer here while `VLLM_TTKV_PROFILE` attributes 1.43 ms to the same call. Both are
true about different things; only the first says what to optimise. `tests/bench_write_path.py` does
the comparison with the queue held deep, on the shapes vLLM allocates:

| | ms per layer-call | ms/step over 28 layers |
|---|---|---|
| torch (`PackedRows` + `pack_*_slots`) | 1.794 | 50.2 |
| fused (`select_triton`) | 0.307 | 8.6 |
| **ratio** | **5.84x** | |

And the same arms on a 1082-token chunk: **0.710 against 2.196 ms/layer-call, 3.09x** (that is the
frozen encode; a request's FIRST chunk is the fit, which keeps the torch packing because the codec
derives its residual from the unpadded product internally and re-deriving it would be a second
spelling of the same residual).

End to end on the same run: the decode step went **94 ms to 79 ms** and the 16-token run **5.50 s to
5.32 s**, with the generated ids unchanged (`12/16` greedy agreement with the stock arm, first
divergence at token 12, exactly as before -- the check that this was performance work). The run total
moves less than the decode does because the prefill is 3.86 s of the 5.32 s and it is the TT-SVD fit,
which is untouched: 137.8 ms a layer, the same before and after.

Where the decode step's 79 ms now goes, per layer-call: ~0.31 write, ~0.20-0.29 projection,
~0.6-0.8 read (`krt` plus gather, split, cos/sin, softmax, `v_out`), ~0.17 plan -- against the stock
arm's 29 ms total, of which ~25 ms is the model's own memory-bound weight streaming. So the next
target is the READ path, and the write is no longer the thing to look at.

Two switches on this path are worth knowing. `VLLM_TTKV_TRITON_PACK=0` forces the torch write, which
is both the fallback arm for that comparison and the way to check a suspected kernel bug against the
definition of the format. And the fused path only applies where it is byte-exact by construction: a
store whose `resid_src` is absent (a request's first chunk, i.e. the fit) keeps the torch packing for
both K and V, as does a head dim that is not a power of two or a bit width outside 2..8.

### The plan path: what `plan` was paying for

`plan` is `impl._Step`, and it exists so the 28 layers of a step share one round of addressing instead
of paying 28 times. It was the third-largest phase and the one that grew with context (`0.165` ->
`0.602` ms/call from 1082 to 4322 tokens), and the reason is worth stating plainly because it decides
what to do about it: **every number in that phase is a device readback, and a readback's cost is not
the op, it is the queue in front of it.** `qsl`, `seq_lens` and `block_table` are `.tolist()` calls,
and a device-to-host copy cannot start until everything already queued has finished.

`tests/bench_step_plan.py` measures exactly that, by running each of those calls at two queue depths
-- queue empty (sync first, so the number is the op) and queue deep (`LOAD` matmuls queued first, so
the number is the wait) -- on an A4000 (sm_86, WSL2), at the harness's two contexts:

| op (1082 / 4322 tokens) | empty us | deep us | what it says |
| --- | --- | --- | --- |
| `qsl.tolist()` (2 int32) | 30.5 / 30.8 | 12773 / 12844 | the wait IS the queue (11-13 ms deep here) |
| `seq_lens.tolist()` (1 int32) | 51.2 / 24.3 | 12769 / 12811 | reading ONE int costs the same as reading 271 |
| `block_table.tolist()` ([1, 68] / [1, 271]) | 52.0 / 45.9 | 12855 / 12928 | transfer size is irrelevant; the drain is the cost |
| slot list, python loop (1082 / 4322 int) | 16.6 / 63.3 | 16.6 / 64.0 | O(seq_len) HOST work -- never pays a drain |
| slot list, list -> cuda | 92.5 / 223.6 | 12956 / 12929 | a pageable H2D copy SYNCHRONISES: another drain |
| `block_slots_host` (both halves) | 87.4 / 241.9 | 12827 / 12918 | the old per-request build |
| `block_slots` (device, `check=False`) | 159.7 / 155.4 | 79.6 / 75.7 | flat in seq_len, and pure enqueue when deep |
| one write's `slot_mapping` slice `.tolist()` | 53.5 / 52.2 | 12829 / 12954 | one of these ran per request PER LAYER |
| WHOLE `_Step` build, host slots | 206.9 / 327.1 | -- | the old shape |
| WHOLE `_Step` build, device slots | 205.6 / 270.9 | -- | the new one: a wash at 1082, better at 4322 |

Two consequences, and they are different in kind:

* **The readbacks are a wait, not work.** Their empty cost is microseconds and their deep cost is
  whatever the GPU still owes. So they cannot be made faster; they can only be made rarer. The engine
  now pays **one drain per metadata object** -- the first readback of the build, which is `seq_lens`
  (`query_start_loc` usually has a `_cpu` twin, so it is free) -- and every later call in that build
  is microseconds. That is why the phase table's `plan` time is a statement about the queue, not
  about this file.
* **One readback was in the wrong place.** `_Step.write_slot_ids` validated its own slice, and the
  write calls it once per request per layer: one drain per layer-call, 28 per step. The step's writes
  are exactly `slot_mapping[:n_tokens]`, so the check now runs once, on that whole slice, in the
  build. Refusing a *later* request's padded slot while the caller is asking about the first is not a
  behaviour change to be shy of -- a padded token is a key that cannot be stored, and the read still
  covers `seq_len` rows.
* **The slot list no longer goes via the host.** `_Step.slots` validated the row from the host copy it
  already had (the row's first block is the request's decoder key) and then rebuilt the ids with a
  `range`/`extend` loop plus a second upload. It now validates there (`decode.check_row_host`, the same
  three refusals, moved out of `block_slots_host` rather than duplicated) and builds the ids on the
  DEVICE from the row tensor (`decode.block_slots(..., check=False)`, the bound having been checked
  already). The device build is not free -- ~150 us against ~87 at 1082 -- but it is FLAT in context
  where the host one is O(seq_len) (~47 ns a token, measured above): at the ~110k tokens where this
  cache's decode speed reaches stock's, the host build would be ~5.3 ms per object, ~37 ms a step.

**The A/B, in the engine, both arms interleaved in one session** (`generate_s` for a 16-token run,
two runs per arm; the arms differ only in these three behaviours, driven by a temporary driver that is
not kept, since `bench_step_plan.py` already measures the pieces):

| context | legacy | after | |
| --- | --- | --- | --- |
| 1082 tokens | 4.93, 4.92 s | 4.69, 4.70 s | **-4.7%** |
| 4322 tokens | 6.03, 6.00 s | 5.97, 5.91 s | **-1.2%** |

Run-to-run spread inside an arm is <=0.01 s, so the sign is real, and the phase table says where it
came from: `write.pack_and_write` fell from 231 to 111 ms (1082) and 519 to 200 ms (4322) -- the
per-layer validation readback -- while `block_slots` *rose* from 10 to 88 ms. The rise does not show
up in the wall clock because enqueued kernels overlap with the GPU; the saving does, because a drain
is the host standing still. The honest summary is that this change is worth ~5% at 1k tokens, ~1% at
4k, and it is the O(1) property, not the 1k number, that justifies it. The absolute seconds are lower
than the 5.26-5.32 s recorded earlier for the same legacy code, so **compare the arms, not the runs**:
that earlier figure was taken in a different session.

### Where the write's fixed cost goes, and what a batch would amortise

`bench_write_path.py` gained a shape section for the question that decides the next design move: at
decode the write runs **once per token per layer**, so what matters is the part that does not scale
with tokens. A4000 (sm_86, WSL2), 1 token, `safe` layout (r1=112, slot 912 B):

| measurement | ms/call | reading |
| --- | --- | --- |
| host, queue deep | 0.348 | the number as usually quoted |
| host, queue **empty** | 0.336 | equal to the above -- so this is **work, not a wait** |
| gpu, from stream events | 0.337 | the same span; the kernels themselves are ~0.05 ms and hidden |
| `pack.coeff` | 0.071 | two advanced-index assignments -- the largest single host item |
| `pack.k` / `pack.v` | 0.053 / 0.044 | each prepped + launched; the launch is 0.033 of it |
| `pack.fused.prep` | 0.014 | the checks and the cache's byte view |
| unaccounted in the wrapper | ~0.15 | outside every phase in `pack_and_write`/`_fused_write` |

and the sweep that says what batching buys (same store, same sizes):

| tokens per call | ms/call | us/token |
| --- | --- | --- |
| 1 | 0.320 | 319.7 |
| 8 | 0.339 | 42.4 |
| 28 | 0.351 | 12.6 |
| 64 | 0.354 | 5.5 |
| 256 | 0.395 | 1.5 |
| 1082 | 0.785 | 0.7 |

**The sweep is per CALL, and that is the number that matters — with one correction to how it was
first read here.** A batch of 28 tokens costs 0.351 ms against 0.321 for 1, so a call's cost is
almost entirely fixed. But that fixed cost is only amortisable when the tokens in the call belong to
**the same layer**: same layout, same slot width, same cache, same decoder. At decode, batch 1, each
layer has exactly one token, and the 28 layers of a step do NOT share a layout — the calibrated `r1`
runs 14 .. 143 across them, so their slot widths differ and they write to 28 different caches. So
"batch the 28 layers into one call" is not available, and the earlier version of this paragraph
claimed `~9 ms/step` from it, which was wrong.

What IS available, and is larger, is batching across REQUESTS. `impl._write_store` loops
`for r in range(plan.n_reqs)` and calls `pack_and_write` once per request, so a step makes
`28 x B` calls:

| batch | write calls per step | ms/step for the write at 0.25 ms/call | if batched across requests |
| --- | --- | --- | --- |
| 1 | 28 | 6.9 | 6.9 (nothing to batch) |
| 8 | 224 | **55** | ~7.2 (28 calls of 8 tokens) |
| 16 | 448 | **110** | ~7.4 |

That is the number to act on, and it is a *concurrency* result rather than a single-stream one: at
batch 1 there is nothing to amortise, and at batch 8 the write alone is 55 ms of a step. It needs the
requests' residuals and V concatenated into one call per layer — the layout is per layer, so no kernel
change is needed — and it is currently blocked behind batch sizes above 1, which the prefill branch
refuses until a multi-query `krt` exists.

Two loose ends worth their own measurement before any of that: the ~0.15 ms a call that is outside
every phase (four times the launch it pays for), and the fact that `encode_frozen` at one token is
0.264 ms where the table in §3 quotes 0.009 ms for it.

### The read's marginal cost, measured, and the claim it falsifies

`tests/bench_decode_read.py` measures `impl._decode_one`'s calls per layer-call -- gather, split,
cos/sin, `krt`, softmax, `v_out` -- with the queue deep (so the number is throughput, not drain) and
alongside `scaled_dot_product_attention` over an fp16 cache of the same length, which is what stock's
read costs in bytes. Llama-3.2-3B geometry (H=8, d=128, HQ=24, G=3), `safe` layout.

| tokens | ours ms/layer-call | ours us/token/layer | dense ms/layer-call | dense us/token/layer |
| --- | --- | --- | --- | --- |
| 1082 | 0.453 | 0.419 | 0.043 | 0.040 |
| 4322 | 1.636 | 0.379 | 0.060 | 0.014 |
| 8192 | 3.028 | 0.370 | 0.100 | 0.012 |
| 16384 | 5.937 | 0.362 | 0.190 | 0.012 |
| 32768 | 11.828 | 0.361 | 0.351 | 0.011 |

**Our marginal is 0.36 us a cached token a layer; dense is 0.010 -- 36x, and the gap widens with
context because both are linear.** At 32k that read is 331 ms a step against stock's 10, before
either pays the model's own ~20 ms of weight streaming. So the claim this replaces -- "crossover at
~110k tokens, we win after that" -- was wrong, and the way it was wrong is worth keeping: it came from
the engine's phase table, where `decode_total` looked flat in context. A phase measures the host
issuing an async kernel group, not the group running, so a compute-bound read looks free there while
its cost lands in a later phase or a later sync. Only the throughput measurement above can see it.

WHY, and it is not bytes. Our slot is 912 B a token a layer against dense's 4096, so at 285 GB/s our
READ floor is ~3 ns a token a layer; we take 360. Reconstructing a key is `r1*d` = 14k FLOP a row a
head, ~116 kFLOP a token a layer, which is ~6 ns at this card's fp32 peak; we take 360.

**The first explanation for that was wrong, and the sweep that killed it is the useful part.**
`tests/bench_decode_read.py` sweeps `kt`, the tokens per tile: the grid is `(ceil(n/kt), H)` programs,
so a smaller `kt` means MORE re-reads of the head's `[r1, d]` decoder slice and the same arithmetic.
If decoder traffic were the cost, `kt=32` would be worse than `kt=64`. It is not (1.39 against 1.33 ms
at 16384), and taller tiles are catastrophic (`kt=128` 14.6 ms, `kt=256` 59.0), because `KT` sizes a
token axis the block holds. So the tiling is already at its optimum, `br` is not a lever either, and
what remains in `krt` at `kt=64` is the reconstruction: 1.9 GFLOP in 1.33 ms is **~10% of this card's
fp32 peak**.

That reframes the project. The algebra is fixed -- pre-rotary storage means the rotation lands on the
reconstructed key, so the score cannot be factored as `(q @ D^T) . coeff`, which is the one trick that
would cut the arithmetic by ~100x. What is NOT fixed is the kernel's 10% efficiency, and that is where
the whole long-context story now lives: at peak the reconstruction is ~6 ns a token a layer against
dense's bandwidth-bound ~11, so parity-and-better is reachable and 10% efficiency is what makes us
36x worse today.

### What the vLLM integration itself costs, itemised

The port is not just "our format on their runtime" -- several of the costs are imposed by the shape of
vLLM's interfaces, and they are worth naming because some are fixable from inside the plugin and some
are not. Measured where possible, marked as such.

1. **We forgo vLLM's attention kernels entirely.** This is the largest one and it is inherent to a
   reconstruct-then-dot format: FLASH_ATTN, FLASHINFER and TURBOQUANT all assume the cache holds keys
   that can be dotted with the query, and ours holds coefficients, a bitmap and codes. So `krt`,
   `v_out` and the softmax are ours, and they are compared against kernels that have had years of
   tuning. Measured: our read marginal is 0.36 us a cached token a layer against 0.011 for a dense
   fp16 SDPA call -- 34x, of which the kernel quality is one part and the format's arithmetic
   (`r1*d` against `2d` per row) is the other.
2. **CUDA graphs are unavailable, and that is a small cost -- measured, because it was expected to be
   large.** `TTKVAttentionMetadataBuilder` inherits `AttentionCGSupport.NEVER`: the impl splits the
   batch with `.tolist()` and resolves decoders in a Python registry, so a captured graph would replay
   one batch's shapes against another's data. `--graphs` on the BASELINE arm prices what that gives
   up: **49.7/47.9 -> 56.0/55.4 tok/s (+14% decode**, two reps each, 7.4k context), for 4159 -> 3934
   blocks (-5% capacity, because torch.compile needs workspace). So being host-interactive costs ~14%
   and not the 2x a launch-bound path would suggest -- which is consistent with the write-path
   finding that the kernels are hidden behind host issue time.
3. **The cache is sized from PEAK memory, and our peak is transient.** vLLM budgets the pool from its
   profiling pass, so the fit's fp32 intermediates are charged against us forever even though they
   exist for milliseconds. This is the whole of the gap between the format's 4.53x and the realized
   **1.74x** concurrency, and it is a pure integration artifact rather than a property of the format
   -- the single highest-value thing to fix for anyone whose use case is capacity (see §4, and note
   that fp8 reaches 8.77x and TURBOQUANT 12.43x in-tree, so 1.74x is not a claim worth making yet).
4. **One page width for every layer and every request, against a per-layer `r1` that runs 6 .. 42 ..
   166.** The spread is 28x inside a single prompt, so a uniform page must be sized for the layer that
   needs the most and every other layer wastes the difference. `tools/calibrate_r1.py` and the
   `backend.py` table exist to live with this. A per-layer width would recover real bytes, and it is
   the clearest answer to "would a different engine design let this format save more".
5. **No end-of-step hook, so the write cannot be batched across layers.** A transformer computes each
   layer's K during that layer's forward, and attention at that same layer needs the token -- so the
   write has to happen per layer, per token, which is the worst possible amortisation for a
   fixed-cost call (§4: 0.35 ms a call for 28 tokens against 0.32 for 1). Deferring needs the store to
   lag one token, which is a change to the read path, not something the engine can offer.
6. **No first-class hook for pre-rotary K.** `patch.py` rebinds `forward` on the model's own rotary
   module to capture the raw K. It works, and it is a monkeypatch rather than an API.

### Why more kernels will not make this faster

The instinct is that the read is slow because the kernels are not hand-written CUDA. `tests/bench_krt.py`
compares the shipped CUDA `krt` (`ttkv_cuda`, NVRTC, sm_86) against the Triton port on identical
stores, and both against the reference, at n=16384:

| r1 | CUDA ms | Triton ms | tri/cuda |
| --- | --- | --- | --- |
| 17 | 0.467 | 0.713 | **1.53** |
| 62 | 0.815 | 0.831 | 1.02 |
| 119 | 1.257 | 1.080 | **0.86** |
| 278 | 2.555 | 1.612 | **0.63** |

They cross near r1=90 and Triton wins above it -- and **21 of the 28 layers of this model sit above
that crossover** (`calibration/llama-3.2-3b-16384.json`: r1 per layer runs 14, 28, 67 .. 110 .. 143;
only 7 are below 90). So swapping in the CUDA kernel would make the read *slightly worse*, not
better, and that is before the far larger point:

| at 1082 tokens, per decode step | ms | of a ~79 ms step |
| --- | --- | --- |
| the model's own weight streaming | ~21 | stock's whole step, irreducible |
| our read (`decode_total`, all six pieces) | ~14.9 | 19% |
| `krt` alone, inside that | ~6.2 | **7.8%** |
| our write (`pack_and_write`) | ~6.9 | 9% |
| our projection (`encode_frozen`) | ~11.7 | 15% |
| plan and the per-layer adapter work | ~15 | 19% |

**Zeroing `krt` entirely would move 79 ms to 72.8 -- 12.7 to 13.7 tok/s. Zeroing the whole read would
give 15.6.** Reaching 40 tok/s needs a 25 ms step, i.e. ~4 ms of our own work against ~58 today, and
no kernel quality delivers a 14x. What delivers it is doing 28 rounds of small per-layer work in
fewer, larger rounds -- the write batched across layers is measured at 0.35 ms for 28 tokens against
9.7 for 1-at-a-time, and the read has never had the fusion the write got. This is a count-of-operations
problem, not a kernel-quality one, and it is the same finding from three directions now (the write
path, the plan path, and the A/B that priced CUDA graphs at only +14%).

**One correction since this was written, and it sharpens the claim rather than weakening it.** The
read has now had the query-blocked change the write got in spirit — `v_out_multi_from_slots`, in "The
continuation chunk" above — and it delivered 3.84x. That does not contradict this section, and the
reason is the whole point: it helped on a CONTINUATION chunk, where there are `q_len` query tokens per
stored row and an axis to amortise the decode over. Decode has exactly one query token per cached row,
so there is no such axis and every number in the table above stands unchanged. What does not help is
kernel QUALITY; what helps is finding an axis over which the per-layer work can be shared. Chunked
prefill has one; single-stream decode does not.

### Decode against context, in the engine: the phase table was wrong

`tests/bench_decode_vs_context.py` measures the one number the whole long-context question rests on,
and it settles an argument that the two instruments had been having. For each context it runs two
generations of the same prompt (`max_tokens=1` and `1+GAP`) and takes the difference over `GAP`, so
prefill cancels and only decode steps are timed. A4000, Llama-3.2-3B, batch 1, `safe`:

| prompt tokens | ours ms/step | ours tok/s | stock ms/step | stock tok/s | ratio |
| --- | --- | --- | --- | --- | --- |
| 1082 | 34.6 | 28.9 | 19.4 | 51.6 | 1.8x |
| 4322 | 56.5 | 17.7 | 21.2 | 47.3 | 2.7x |
| 8642 | 87.5 | 11.4 | 21.9 | 45.7 | 4.0x |
| 14402 | 124.4 | 8.0 | 23.5 | 42.5 | **5.3x** |
| | **+6.7 us/cached token** | | **+0.31 us/cached token** | | |

**The disagreement is resolved against the phase table, and it was never close.** `decode_total` looked
flat in context; the wall clock says our step grows 6.7 us per cached token, i.e. 0.24 us a token a
layer, which is the same order as the 0.36 the isolated kernel bench measured. A phase times the host
ISSUING an async kernel group, so a compute-bound read looks free in the table while its cost lands in
the step's wall clock — the same trap the write path demonstrated. Every "our read is flat" statement
in this file before this section was wrong, and the prefill statement was too: prefill is not flat in
prompt length either (ours ~3.5 s fixed plus ~0.48 us a token, stock's ~0.16), so the prefill ratio is
26x at 1082 tokens and 4.5x at 14402, not the ~1.6x an earlier version of this file claimed.

**And the widening slope is inherent to the format, which is the part worth sitting with.** Reading one
cached token costs us `r1*d` multiply-adds per KV head to reconstruct the key, and costs dense `d` to
read plus `G*d` to dot. The arithmetic ratio is therefore about `r1/(1+G)` — 110/4, ~27x for this
model — and it does **not** improve with batch size, because at decode each request's cached rows are
reached by exactly one query token per step.

Amortisation needs many query tokens per cached row — a `Q` large enough that the reconstruction is
shared by enough queries to matter. A row is reconstructed ONCE and reused by every query that attends
it, so the per-row cost becomes `r1*d/Q` against a dot that is `HQ*d` however many queries there are.
The dots are identical to dense's, so **we never become cheaper than dense on arithmetic — the
reconstruction simply fades from a 27x penalty to a rounding error.** What is left at that point is the
byte advantage (912 B a token a layer against 4096), which wins only if the kernel is read-bound rather
than compute-bound. So the honest claim for the query-dense regime is *parity is reachable*, where
single-stream decode is 5.3x. Whether parity becomes a win depends on the kernel, and that is the open
question — not a settled advantage.

* **single-stream decode is the regime this format is worst at**, and no kernel quality fixes it: even
  at 80% of fp32 peak the reconstruction is ~0.43 us a token a step, which at 16k is ~41 ms against
  stock's ~24, i.e. still ~1.7x slower. Today we are at ~5-10% of peak, so there is real ground to
  gain, but not to a win;
* **query-dense work is where the penalty fades** — and it is where the port is currently weakest, since
  `_prefill` is the torch `attend_prefill` and costs 4.5x stock. Parity is reachable there; a win is not
  established, because the dots are the same as dense's and our advantage is bytes, not FLOPs. That, not
  the decode kernel, is where the next engineering effort pays.

This is worth stating plainly because it changes the pitch: the honest claims are a 4.53x smaller cache
and better accuracy per byte, with speed arriving in prefill and batched regimes rather than in
single-stream decode.

### The continuation chunk: the first query-dense work, and a 3.84x

The section above ends by pointing at query-dense work as where the effort should go. This is the
first piece of it, and it started from a collapse rather than a plan.

**The cliff, and why it is not avoidable by configuration.** Running the concurrency sweep with
long prompts (`--context 8192`, i.e. 7379-token prompts) at the shipped `max_num_batched_tokens`
of 16384 produced this, on a configuration that is *not* the published decode table (chunked
prefill and prefix caching are both on here, which is what the sweep wants):

| N | prefill s | tokens | tok/s prefill | blocks / pool |
| --- | --- | --- | --- | --- |
| 1 | 7.53 | 7,379 | 980 | 462 / 7244 |
| 2 | 13.79 | 14,758 | **1,070** | 924 / 7244 |
| 4 | 225.26 | 29,516 | **131** | 1848 / 7244 |
| 8 | 546.67 | 59,032 | **108** | 3696 / 7244 |

N=1 and N=2 run at ~1,000 tokens/s and N>=4 at ~110, an 8-10x drop with no error anywhere. The
cause is one number: `2 x 7379 = 14,758` fits the 16,384 budget and `4 x 7379 = 29,516` does not, so
from N=4 the scheduler splits the prefill and every split schedules a CONTINUATION chunk — a request
with rows already in the store and only some of its queries in the current batch.

**Raising the budget is not available as an escape.** It costs activation memory that comes out of
the same 16 GB: at 65536 and at 32768 the engine refuses to start with `No available memory for the
cache blocks`, leaving nothing for the KV pool against `gpu_memory_utilization=0.90`. So 16384 is
this card's ceiling, the continuation path is what long prompts on a small card actually run, and a
4x-slower path there is a serving defect rather than an edge case.

**The diagnosis.** The continuation itself was already correct — it reads the whole store through
`krt_multi_from_slots` and agrees with attention over the whole store to 2.6e-07. Two things made it
slow, and the second is a factor of the group size rather than a constant:

1. **The readout was launched once per query token.** `v_out_from_slots` takes `[HQ, n]` weights and
   returns `[HQ, d]`, so a chunk of `q_len` queries cost `q_len` launches — 5,753 of them a layer,
   161k a step, at ~0.44 ms each.
2. **Its contraction padded the GROUP axis to 16.** It contracts `[GP, KT] x [KT, HALF]`, so `GP` is
   the only dimension that can clear `tl.dot`'s minimum of 16, and `G=4` therefore pads to 16 and
   does **4x the arithmetic it needs**. `krt` never had this problem because its query block clears
   the minimum on its own — the readout's shape simply did not have a query axis to put there.

`v_out_multi_from_slots` fixes both at once by moving the query axis into `M`: one launch a chunk,
and `M = BQ*GP` = 64 with `GP = next_power_of_2(G) = 4`, so the padding factor is 1.0 instead of 4.0.
The tokens still reduce inside the program, because an output row is a sum over every stored token.

| | before | after | |
| --- | --- | --- | --- |
| N=1 prefill s | 7.53 | 7.61 | unchanged — no continuation happens |
| N=2 prefill s | 13.79 | 13.70 | unchanged — no continuation happens |
| N=4 prefill s | 225.26 | **58.64** | **3.84x** |
| N=8 prefill s | 546.67 | **188.74** | **2.90x** |

Per prompt, the split run goes from 56.3 s (N=4) and 68.3 s (N=8) to 14.7 and 23.6, against 6.9 for
the unsplit N=2 — so a continuation is 2.1x a first chunk at N=4 and 3.4x at N=8, where it was 8.2x
and 9.9x. The whole four-point sweep also finished in 5m40s where the previous one was still running
after 13 minutes.

N=1 and N=2 being unchanged is the control: the change is confined to the continuation, so a
speedup that also moved the first-chunk path would have been a different, suspicious result. In
isolation, `tests/test_v_out_multi.py` prices the readout at **0.944 ms for Q=256 in one launch
against 256 x 0.440 ms = 112.6 ms per-query, 114x**, and at Q=16, where a chunk tail still lands,
17x.

**Two things this does not fix, stated because they are the reason it is 3.84x and not 10x:**

* the reconstructed K rows are still rebuilt **once per query block** inside `krt_multi_from_slots`,
  because the query axis is a grid dimension there. The reconstruction is ~60% of that kernel's
  arithmetic by count, so this is the next tiling change and it is a real one;
* the `[HQ, q_len, seq_len]` score matrix is still materialised and then softmaxed into a second
  tensor of the same size — ~0.9 GB each at these shapes, per layer, 50 GB of allocation traffic a
  step. Fixing that means a tiled online softmax, which is a fusion rather than a tiling change.

One measurement got worse in the process, and the fix took two more steps than expected: the
concurrency sweep's decode rate is a DIFFERENCE between two generations, and it came out negative
(-1.6, -1.5, -0.6 tok/s at N=1,2,4) with prefix caching on. Decoupling the flags (`--chunked` for
chunked prefill, `--prefix-cache` for caching) was necessary but not sufficient — the difference was
still unusable, for two further reasons, and both are written up in "Decode against concurrency"
below, which is where the honest answer to the throughput question now lives.

### Decode against concurrency: capacity does not convert to throughput

This is the measurement the whole project is aimed at — a 4.53x smaller cache is only worth anything
if it buys more concurrent requests, and more concurrent requests are only worth anything if they
produce more tokens a second. The answer is **no, not as implemented**, and it took three attempts to
get an instrument that could say so.

**Why the first two answers were wrong.** `run_vllm_concurrency.py` derives the decode rate as a
difference of two generations of the same prompts, so the prefill cancels. That is sound in principle
and it is exactly what the published decode table uses. It failed here for reasons that are specific
to this port:

1. **Generation length was not forced.** Neither call passed `ignore_eos`, so on repeated prose greedy
   decoding can stop on EOS in BOTH calls, the two cost the same, and the difference is ~0 — which
   divides into a rate of hundreds of tokens a second, or a negative one. This reads as a fast result
   rather than as a broken one, which is why it survived a whole sweep. `tests/bench_vllm_baseline.py`
   had this flag; this harness and `bench_decode_vs_context.py` did not.
2. **The marginal was smaller than the prefill's own variance.** This port's prefill is SECONDS (5-36 s
   here) where stock's is 0.6-2 s, and the decode marginal for a short gap is ~2 s. So the instrument
   was measuring the run-to-run wobble of a 36-second prefill. The tell was an N=2 result implying
   **4.5 ms per decode step**, against a step the table above prices at ~34-39 ms. Widening the gap to
   256 steps makes the marginal tens of seconds and the wobble irrelevant.

Both fixes are in, and the control that proves the instrument now works is that stock scales on it
monotonically while ours does not. Context 2048, so nothing splits and no continuation is involved;
`gap=256`, 16 + 256 tokens a request; A4000, batch 1 to 8:

| N | ours decode s | ours tok/s | ours ms/step | stock decode s | stock tok/s | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 9.87 | 25.9 | 38.6 | 5.03 | 50.9 | 2.0x |
| 2 | 12.66 | 40.4 | 49.5 | 5.33 | 96.1 | 2.4x |
| 4 | 22.10 | 46.3 | 86.3 | 5.66 | 181.0 | 3.9x |
| 8 | 43.41 | 47.2 | 169.6 | 6.32 | 324.2 | **6.9x** |

**Ours flattens at ~47 tokens a second; stock reaches 324.** Batching buys us 1.8x (25.9 to 47.2) and
then stops, while stock gets 6.4x from the same range. And the ratio degrades rather than holding,
2.0x at batch 1 to 6.9x at batch 8, so concurrency makes the comparison *worse* for us, not better.

**The reason is structural, and it is the same one the write path already showed.** Our step time grows
essentially linearly in batch size from B=2 on — 49.5 -> 86.3 -> 169.6 ms — which means the work is
being issued PER REQUEST and does not merge across requests. It is 28 layer-calls of read and 28 of
write for each request in the batch (`_decode_one` and `_write_store` both loop over requests), each
carrying host readbacks that drain the pipeline, so 8 requests cost 8 times the per-request work plus
a fixed per-step term. Dense attention merges the batch into one kernel per layer; this does not, so
batching cannot be the lever that pays for the format's arithmetic.

**What that means for the pitch, stated plainly.** The capacity numbers are real — 4.53x smaller, and
1.74x realised against stock today. But the argument that more concurrency converts that into
throughput does **not** hold on this port as it stands: at batch 8 we serve 47 tokens a second where
stock serves 324, and adding concurrency does not close it. Closing it needs the per-request work to
become per-step work — one gather, one reconstruction, one readout for the whole batch, which is the
same "fewer, larger rounds" finding as the write path and the plan path, for the fourth time.

### The rank budget, measured as an accuracy cost

The budget is fixed when the pages are allocated and a request's fit lands at its own natural rank,
so the fit is capped and then zero-padded up to the budget. The padding is EXACT — a padded fit and
an unpadded one reconstruct identical keys (max delta **1.19e-07**), and the residual is
bit-identical because it is computed before padding — so the only thing a budget costs is the cap.
Measured on a rank-48 block, `safe`, d=128:

| budget `r1` | natural rank | rel-F reconstruction error |
|---|---|---|
| 16 | 16 | 0.1937 |
| 32 | 32 | 0.1170 |
| 64 | 45 | **0.0439** |
| 128 | 45 | 0.0439 |
| 256 | 45 | 0.0439 |

It saturates the moment the cap stops binding, and a bigger budget is never worse — which is what
makes the calibration table's per-layer `r1` a meaningful thing to spend bytes on. `safe`'s
calibrated per-layer ranks (max 143 at 16k) are comfortably above this fixture's 45, so this is the
regime a real run is in.

And the shipped arithmetic is pinned to the tool that measured the fold penalty:
`tests/test_encode.py` asserts `encode_frozen` agrees with `tools/measure_fold_gap.py` to
**2.2e-04** — that residual gap is the tool keeping a row's ruler in fp32 where the format stores it
as fp16, i.e. the tool is marginally optimistic by ~500x less than the errors it was measuring.

Every file also runs as a plain script (`python tests/test_layout.py`), which is how they were run
before a runner existed. They skip cleanly rather than fail when a layer's requirement is missing:
no CUDA for the kernels, no `transformers` or no cached model for the e2e layer.

**The four layers are different questions, not a size ladder:**

| layer | question it answers | 31 tests |
|---|---|---|
| unit | does the layout bill what the container stores, and does the reference decode decode | 10 |
| kernel integration | do the CUDA and Triton kernels reproduce the reference to 1e-7 | 6 |
| e2e | does a real model forward through the cache reproduce dense logits | 5 |
| bdd | do the acceptance criteria still hold, in domain language | 10 |


All three resolve the codec from **either** layout — `<root>/integration` (this project) or beside
the plugin (a flat checkout) — so the same files run on the CPU workstation and on the GPU box.

`layout.py`, `reference.py` and the kernel test are the pieces that can be verified without a vLLM
install, so they are the pieces that are verified. Each asserts something that is easy to get
*plausibly* wrong:

- `layout.py` — that its per-token byte count equals what the container *actually bills*
  (`nbytes()`), so a drift between the layout and the codec fails a test instead of quietly changing
  a ratio claim; and that the packed slots round-trip to the same dense block
  `PackedRows.materialise()` produces, so packing is a faithful *rearrangement* of the codec's bytes
  rather than a second implementation that merely looks right. The second matters because a permuted
  layout still round-trips through pack/unpack — the assertion has to be on the materialised rows.
  That is the failure mode that has bitten this project before (a `(head, token)` order mistake reads
  as an *anti-correlated* metric, not as noise).
- `reference.py` — that the decode agrees with the same mathematics written in a **different index
  order** (`[token, head, dim]` against `[head, group, dim]`); that **omitting** the per-row rotation
  changes the scores by 0.93 relative, which is the `_refuse_pre_storage` mistake; and that the K
  path tracks attention over the raw keys. The negative control asserts on the *scores*, not on the
  softmax output, because the output turned out to be insensitive: a first attempt measured only
  1.7e-2 for a mistake as large as skipping the rotation entirely.
- `test_kernel_krt.py` — that the **shipped** `krt` kernel agrees with the reference (4.4e-07 on the
  A4000, NVRTC-compiled for sm_86), and that a rolled angle table moves its scores (0.398). Without
  the second, the first would pass even if the kernel rotated every row by one constant angle.

That last result is the one that de-risks the `csrc/` work: `krt`'s numerics are now pinned against
an independent oracle on real hardware, so moving its body into `csrc/` and swapping NVRTC for AOT
nvcc becomes a **delivery** change rather than a numerics change — and delivery is the part that
cannot be unit-tested without a host compiler.

One finding from the reference tests worth carrying into config selection: at `campaign4x` the V-side
relative error is **0.267** against the K-side's **0.030**, and V is what the output is made of — so
on these settings the **V lane count is the dominant fidelity lever, not the K rank**, and the
end-to-end gap against uncompressed attention is essentially all V. (Random V is a pessimistic
stand-in; real V compresses better. The direction holds, the magnitude does not.)

---

## 5. What is genuinely open

1. **The slot width must be known before the cache is allocated; `r1` depends on the data.**
   This is the sharpest constraint in the whole port. `r1` cannot be measured at prefill and
   used to size the cache in the same run, so it has to come from a calibration pass per layer
   (a shipped artifact), or from capping the rank — and capping is a format change, not a
   tuning knob (a cap of 64 under-ranks `tight` by 1.75× and `campaign4x` by ~4.4×). `backend.py`
   raises until the table exists rather than guessing.
   *Resolution, verified:* the per-layer width itself is accepted as-is. `UniformTypeKVCacheSpecs`
   sums page sizes across layers, so each layer keeps its own page — no padding to a global max,
   no second cache region. Constraint: all TT layers must share a base spec and `block_size`, so
   compress every layer rather than mixing TT with another attention type.
   **The table now exists and is wired.** `tools/calibrate_r1.py` captures pre-RoPE K per layer and
   calls `ttkv_codec.tt_svd` — *the same function `build_k_layer` calls, at the same `tol`* — rather
   than reimplementing the rank rule, because a second copy that agreed today would drift later and
   fail silently: a wrong `r1` is still a perfectly valid rank. The shipped artifact is
   `calibration/llama-3.2-3b-16384.json` (LongBench prose, 4 samples at 16k, per-layer max over
   samples), and vLLM is pointed at it with `VLLM_TTKV_CALIBRATION=<path>` (plus `VLLM_TTKV_CONFIG`
   for the allocation, default `safe`).

   Measured, `safe` allocation on prose at 16k: **r1 14..143, mean 103.1**, stable to within 8%
   across the four samples, giving **4.53× mean per layer (4.34..5.02)**. That independently
   corroborates the 4.59× measured on real K/V earlier — two different routes to the same number.

   Two decisions inside it are load-bearing and were settled deliberately. **Max over samples, never
   the mean:** `r1` varies by layer *and* by text, and the measured spread across corpora is 4× (21
   on RULER filler vs 85 on prose at 16k), so a mean under-ranks half the traffic. **Per-layer, not
   one global rank:** worth a measured **4%** of bytes here — modest, because most layers sit in a
   narrow band and a few outliers set the max — but it is exactly-safe rather than safe-and-wasteful.

   Two guards stop a wrong table from being used, both because the failure would otherwise be
   silent: `customize_spec` refuses when the table's `head_dim` differs from the spec's `head_size`
   (a rank fitted for a `[n, d·H]` unfolding is not a rank for a different matrix), and an unmatched
   layer name returns `None` rather than resolving to a neighbouring layer's rank.

   *One gap this exposed, worth noting:* `layout_for_layer` originally read a registry populated by
   live `TTKVRopeStore` instances, which is empty at cache-allocation time — before any forward.
   It now reads the calibration table, and the two registries are kept separate: the live one is the
   truth during a forward, the calibrated one is the only thing that can answer before one exists.

2. **The per-request decoder has no home in vLLM, and this is the hardest problem here.**
   Each sequence gets its own decoder (sharing one across a batch measured 5.8 points wrong), so
   it is per-request state of `32 layers × r1 × H·d` bytes — ~4.2 MB at r1=128. That is only
   ~1-3% of a realistic KV budget, so it is not a capacity ceiling. The problem is *mechanism*:
   it is produced mid-flight at prefill and must persist for the request's lifetime, and no
   attention-backend API allocates that. `RequestState`
   (`vllm/v1/worker/gpu/states.py`) is the only pre-sized-by-`max_num_reqs` container, and
   `EncoderCache` is the precedent for id-keyed request-scoped GPU tensors produced at prefill
   and freed on completion — but neither is reachable from a backend. The practical route is a
   plugin-owned pool with handles threaded through `attn_metadata`, the way HPC threads its
   `hpc_*` fields. Sizing it needs `max_num_seqs × layers × r1_max × H × d` before the run.

3. **Chunked prefill — the kernel exists now, and one half of the concern is still live.**
   A later chunk must attend over earlier stored rows, which are raw, so it needs the `krt` path
   over a prefix rather than the stock-SDPA prefill used for a first chunk. That path is written
   (`_prefill_continuation`, over `krt_multi_from_slots` + `v_out_multi_from_slots`), agrees with
   attention over the whole store to 2.6e-07, and is 3.84x faster than the per-query version it
   replaced — see "The continuation chunk" below. Both `enable_chunked_prefill` and
   `enable_prefix_caching` are reachable from the harness (`--chunked`, `--prefix-cache`).
   What remains is the second half: making the *build* depend on chunk boundaries makes the decoder
   depend on scheduler decisions, so a preempted-and-recomputed request could get a different
   basis. The fit still lands on a request's first chunk, so a request recomputed with a different
   first-chunk size refits. Prefer a fold schedule that is a function of the sequence, not of the
   scheduler's progress.
4. **The fold schedule is scheduler-driven.** Paged serving mixes sequences of different ages in
   one batch, so a global token counter does not work. `metadata.py` carries the tail length for
   this, but the policy is unresolved.
5. **The continuation's score matrix is materialised, and that is now the largest remaining cost.**
   `_prefill_continuation` builds `[HQ, q_len, seq_len]` scores, then softmaxes them into a second
   tensor of the same size. At the shapes a 16k budget actually produces that is ~0.9 GB a layer
   per tensor, so ~1.8 GB of allocation traffic a layer and 50 GB a step. A tiled online-softmax
   kernel that never materialises it is the next piece of work, and unlike the readout it is a
   fusion rather than a tiling change.
6. **Streaming drift is unmeasured.** The decoder is frozen at prefill and never refit, so
   quality should decay as generation moves away from the prefill span. Nothing in the
   measurements covers a long generation with many folds.
7. **The `patch.py` interception is the fragile part.** For production, prefer the upstream
   shape: a model variant that constructs the store explicitly, as `hy_v3.py` does for HPC.

---

## 6. Where this goes next

### Why pre-rotary is step one here, not step six

The natural engineering order for work like this is "get a serving post-rotary system, then add
pre-rotary on top". That is the order a **PR-shaped** effort takes, and it is a defensible one:
the pre-rotary form needs the model-forward ordering change (fact 1 above), which is the hardest
thing to get reviewed upstream, so you establish the format first and move the storage point
later.

**That ordering does not apply to this project.** Two reasons. First, the ratio loss is not a
quality dial to be optimised later — post-rotary costs ~16% of the joint ratio on RULER filler
and ~27-30% on prose (~36% on the K side alone), so it is the thing being avoided, not the
starting point. Second, and this is the part that changes the plan: **going out-of-tree removes
the constraint that produced the ordering.** There is no review to stage for and nothing to
bundle, so the model-forward intervention (`patch.py`) can land immediately rather than waiting
for the format to be accepted. Plugin-first and pre-rotary-first are the same decision, not two
independent ones.

The consequence is that the refactors that would otherwise be prerequisites are **not**
storage-point-agnostic, even where they look it:

- `krt` becomes the ONLY score kernel, because absorption is unavailable against raw keys. So
  the token-tile access pattern is the only access pattern — which strengthens the case for a
  token-major layout rather than merely permitting it.
- The dense tail must hold **raw** rows, not rotated ones. The fold and tail paths change
  semantics, not just addressing.
- The per-request decoder is fit on **raw** keys, so its correctness requirement is unchanged
  but its inputs and its drift behaviour are different.

So: do the refactors, but design them for the raw store from the start rather than adding the
storage point afterwards.

In increasing order of upstreaming:

1. **Finish the plugin** — `csrc/` with `krt` and `v_out`, the calibration table, the metadata
   builder. Prove decode matches the container to cosine 1.0 against
   `tiled_online_attention.py`, which is the deliberately-slow correctness oracle.
2. **Align with RSQR** ([#57672](https://github.com/vllm-project/vllm/issues/57672), open).
   It proposes raw-key storage with rotation deferred to attention time — the same substrate
   this needs. If it lands, the ordering half of this plugin is inherited rather than owned, and
   TT-KV becomes the compression layer on top. Comment there; do not propose a competing
   mechanism.
3. **RFC, then upstream.** Mirror the TurboQuant/UltraQuant path: a `KVQuantMode`, a `CacheDType`
   entry, a backend file, a registry entry — and the model-side module proper, which is the one
   change that genuinely needs to be in-tree.
