# DFlash2 drafter with `-sm tensor` on llama.cpp b11552

This branch (`mv/dflash-tp`) carries a hand port of upstream pull request
[ggml-org/llama.cpp #27858](https://github.com/ggml-org/llama.cpp/pull/27858),
"fix assertion DFlash2 with `--split-mode tensor` on CUDA", by **@art-den**, onto upstream build
b11552 (`23b0202a1`). The design and nearly all of the code are the PR author's. The port only
carries it forward to a newer tree and reports what it does on one machine.

It is a private working branch for one machine. It is not proposed upstream.

## What was ported, and from where

| | |
|---|---|
| Source | PR #27858, head `3286b033c`, 10 commits, based on b10665 (`ca3d5a3e1`), 7 files, +300 −32 |
| Target | b11552 (`23b0202a1`) plus three small local patches (see "Branches") |
| Result | one squashed commit on `mv/dflash-tp` |

What the PR does: with `-sm tensor` the output matrix (lm_head) is split along the vocabulary
across the GPUs, so the DFlash2 candidate selector, which upstream builds inside the compute graph,
cannot run. The PR skips the two selector tables at load, reads them from the draft GGUF on the
host, lets the decoder's hidden states ride the `nextn` output, and does the candidate ranking and
the block walk on the CPU from the gathered lm_head output. It also caps the draft context's
ubatch for block drafters, which keeps its compute buffer small.

What the port had to change:

- `common/speculative.h`: a textual conflict. Upstream added `common_speculative_are_compatible()`
  where the PR adds its two inline helpers; both are kept.
- `common/speculative.cpp`: the PR predates the `common_batch` wrapper. Three places moved to it:
  `batch.n_tokens` → `batch.size()`, `batch.token[i]` → `batch.tokens[i].id`,
  `common_batch_add(batch, …, { seq_id }, …)` → `batch.add(…, seq_id, …)`.
- `common/common.cpp`, `src/llama-ext.h`, `src/llama-model.{h,cpp}`, `src/models/dflash.cpp`:
  merged without change.

One thing to know when using it: **do not pin the drafter to one device** (`-devd`). A draft model
on a single device is loaded in layer mode, and its graph then cannot use the target's split
output matrix; the server aborts at load with
`pre-allocated tensor (output.weight) in a buffer (Meta()) that cannot run the operation`.
Without `-devd` the drafter inherits `-sm tensor` and loads.

## Branches

| Branch | Content |
|---|---|
| `mv/base` | b11552 + Docker build with `-j6` |
| `mv/fill` | + recurrent state cleared with a fill instead of a multiply by zero (issue #29830); CUDA fill is a no-op on empty tensors |
| `mv/reqdraft` | + a request may lower the draft depth (`speculative.n_max`) |
| **`mv/dflash-tp`** | + the port of #27858 (this branch) |
| `mv/dflash-tp-24549` | + upstream PR #24549 on top (built as a fallback; not needed) |
| `mv/tp`, `mv/thr` | earlier experiments: #24549 alone; an attention-kernel batch limit that turned out slower |

## Hardware and build

- 2 × NVIDIA RTX 5060 Ti 16 GB (180 W limit each), one on a CPU PCIe port, one behind the
  chipset; both links at 16 GT/s ×4 under load. Ryzen 5 8500G, 14 GiB RAM. Ubuntu, kernel 7.0,
  NVIDIA driver 595.91.07 (open module).
- Model: Qwen3.8-27B, `UD-Q4_K_XL` GGUF. Drafter: the matching DFlash2 GGUF, `Q4_K_M`.
- Build: `docker build --network host -f .devops/cuda.Dockerfile --target server -t <tag> .`
  (CUDA 12.8.1 image, GCC 14, upstream's options, `-j6`).

## Commands

Server (the measured configuration):

```
GGML_CUDA_ALLREDUCE=internal llama-server \
  -m Qwen3.8-27B-UD-Q4_K_XL.gguf -ngl 99 -sm tensor -c 262144 -fit off \
  -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 -np 1 -ctxcp 4 -cram 0 \
  --spec-type draft-dflash -md Qwen3.8-27B-DFlash2-Q4_K_M.gguf -ngld 99 --spec-draft-n-max 7 \
  --jinja --temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.0 --presence-penalty 1.5
```

`-fit off` is required in tensor mode. The second configuration measured replaces `-np 1` by
`-np 2 --kv-unified` (two slots sharing one KV store, so a small side request does not evict a
long conversation).

## Numbers

Method: one server load per row group. The context is padded with source code to each depth in
turn; at each depth a code task and a prose task generate 300 tokens at temperature 0 with a
fixed seed, three times each. Figures are decode tokens per second, code task / prose task.
The repeat spread is at most about 2 % from 32K on. All runs are from the same day on the same
machine.

| Configuration | 4K | 32K | 128K | 230K | Cold read 32K→128K / 128K→230K (tok/s) | VRAM per card after 230K (MiB) |
|---|---|---|---|---|---|---|
| A. b10975, layer split, DFlash2 depth 7 (what the machine serves) | 70.5 / 40.9 | 66.1 / 38.1 | 37.4 / 26.2 | 24.6 / 16.5 | 476 / 307 | 15,804 / 14,712 |
| B. b11552, tensor split, MTP depth 4, two slots | 62.7 / 45.0 | 56.2 / 44.7 | 40.9 / 29.3 | 30.8 / 23.6 | 653 / 458 | 15,674 |
| **C. this branch, tensor split, DFlash2 depth 7, two slots** | **84.2 / 53.9** | **83.8 / 50.6** | **61.4 / 35.2** | **41.9 / 31.1** | 672 / 488 | 15,800 |
| C with one slot | 85.6 / 54.6 | 83.7 / 50.2 | 61.6 / 35.1 | 41.3 / 30.8 | 671 / 486 | 15,186 |

C against A: +19 / +32 % at 4K, +64 / +34 % at 128K, +70 / +88 % at 230K.
C against B: +34 / +20 % at 4K, +50 / +20 % at 128K, +36 / +32 % at 230K.

Where it comes from, code task at 128K:

| | ms per target pass | tokens per pass | draft acceptance |
|---|---|---|---|
| A, layer + DFlash2 | 121 | 4.54 | — |
| B, tensor + MTP 4 | 93 | 3.80 | — |
| C, tensor + DFlash2 7 | 81 | 5.00 | 0.59 |

Draft depth, C with one slot, server loaded at depth 7 and the depth lowered per request:

| Depth | 4K | 32K | 128K | 230K |
|---|---|---|---|---|
| 7 | 85.6 / 54.6 | 83.7 / 50.2 | 61.6 / 35.1 | 41.3 / 30.8 |
| 5 | 58.2 / 39.0 | 59.3 / 39.1 | 45.9 / 29.4 | 30.4 / 23.3 |
| 3 | 54.3 / 42.4 | 49.8 / 39.6 | 38.7 / 30.1 | 29.0 / 23.3 |

The 16 GB cards have 16,311 MiB each, so C with two slots leaves about 0.5 GiB per card.

## Tests

- **Same model.** Ten fixed one-token probes at 4K and at 128K depth, temperature 0, top-10
  log-probabilities, compared with configuration A: mean KL divergence 0.0013 (limit 0.005, set
  beforehand as three times the run-to-run reference of A), largest single probe 0.012, no change
  of the top token. The probes of C are bit-identical to those of B: the port does not change
  what the target model computes.
- **Loop.** 54 minutes of mixed requests on C with two slots (conversations growing to 159K
  tokens, thinking on and off, restarts of the conversation): no error, no degenerate output.
- **Soak.** A multi-turn conversation grown to 252,294 tokens with regenerations: passed; no
  growth of host memory, no swap.
- **Unit tests.** The same source built with `-DLLAMA_BUILD_TESTS=ON`, `ctest` on the CPU: all
  pass except the tests that need a network download or the Python `jinja2` module, which the
  build container lacks. `test-backend-ops` was not run; the port touches no backend kernel.
- **Not tested:** DSpark drafters, `-sm row`, more than two GPUs, other models, Windows (the port
  keeps the PR's `_fseeki64` branch untouched).

## Limits of this evidence

- One machine, one model, one drafter file, one day.
- The stability evidence is one mixed-load loop and one soak to about 250K tokens. Upstream marks
  tensor mode as experimental.
- The CPU-side selector scans the full vocabulary once per block position on one thread; on this
  machine that cost is not visible in the pass time, but it will scale with vocabulary size and
  draft depth.
- A draft depth lowered per request (3 or 5 with the server loaded at 7) was slower per pass than
  depth 7 itself, and accepted fewer tokens per pass. Depth 7 was best at every context depth.
