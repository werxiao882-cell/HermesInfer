# AGENTS.md

Guidance for AI agents (and humans) working in this repository.

## Project

`myvllm` (brand `miniVLLM`) — a from-scratch, educational reimplementation of the
vLLM inference engine, forked from Nano-vLLM. Implements PagedAttention and
Flash Attention in Triton, continuous batching, prefix-cache machinery,
tensor parallelism, and CUDA graph capture. Targets **Qwen3** and **Llama-3.2**
model families. CUDA-only (no CPU/MPS path).

## Environment & Commands

Package manager is **uv**. Python must be `>=3.11,<3.12`. A CUDA-capable GPU is
required.

```bash
uv sync                                    # install deps from uv.lock
uv run python main.py                      # Qwen3-0.6B inference demo
uv run python main_llama32.py               # Llama-3.2-1B-Instruct demo
uv run python benchmark_prefilling.py       # attention impl comparison (prefill)
uv run python benchmark_decoding.py         # paged-attention comparison (decode)
uv run python benchmark_tps.py             # minivllm vs vLLM vs HF transformers
uv run pytest tests/ -v                    # run the scheduler regression tests
```

There is **no** configured lint, typecheck, or formatter. `black` and `isort` are
listed as dev deps but have **no tool config** in `pyproject.toml`; no `[tool.ruff]`
or `[tool.mypy]` sections exist. Treat `pytest` as the only verification command.

There is **no `[project.scripts]`** — no console entry point. Entry scripts insert
`src/` onto `sys.path` manually (`main.py:9`, `main_llama32.py:9`,
`test_scheduler.py:4`), so they expect to be run from the repo root with `src/`
uninstalled.

> Always run the scheduler tests after touching `engine/scheduler.py` or
> `engine/block_manager.py` — those are the only covered modules and the tests
> are regression-oriented (each class guards a previously-shipped bug).

## Repo Layout

```
src/myvllm/                 the engine package (src-layout, installed as `myvllm`)
  engine/
    llm_engine.py           LLMEngine — public API, spawns worker procs for TP
    model_runner.py         ModelRunner — GPU workhorse, KV cache, CUDA graphs
    scheduler.py            Scheduler — continuous batching, preemption
    block_manager.py        BlockManager + Block — paged KV address space
    sequence.py             Sequence + SequenceStatus — per-request state
  layers/
    attention.py            Triton kernels: store_kv, flash prefill, paged decode
    linear.py               tensor-parallel linears + weight_loader pattern
    embedding_head.py       VocabParallelEmbedding + ParallelLMHead
    rotary_embedding.py     RoPE incl. Llama-3 scaling
    activation.py           SiluAndMul (SwiGLU)
    layernorm.py            RMSNorm (class is named LayerNorm)
    sampler.py              Gumbel-max sampling
  models/
    qwen3.py                Qwen3ForCausalLM (has q/k norms)
    llama.py                LlamaForCausalLM (Llama-3 rope, no q/k norms)
  utils/
    context.py              Context singleton (attention metadata)
    loader.py               HF safetensors loader (QKV + gate_up merge)
  sampling_parameters.py   SamplingParams (greedy is forbidden — temp>1e-10)
  __init__.py               EMPTY — no flat public API; import submodules directly
tests/
  test_scheduler.py         regression tests for Scheduler (3 classes)
  scheduler_tests.md        how-to-run notes
benchmark_*.py              standalone benchmark scripts
main.py / main_llama32.py   usage demos
HowToApproachvLLM(_zh).md   the authoritative design doc — read this first
```

## Architecture in One Page

Request lifecycle: `LLMEngine.generate()` (`llm_engine.py:95`) →
`Scheduler.schedule()` (`scheduler.py:35`) → `ModelRunner.run()` (`model_runner.py:386`)
→ `Scheduler.postprocess()` (`scheduler.py:104`). One `step()` (`llm_engine.py:68`)
is one scheduler iteration.

- **Continuous batching** (`scheduler.py`): prefill from `waiting` first, then
  decode one token per `running` sequence. When KV blocks run out, `preempt()`
  (`scheduler.py:96`) deallocates and re-enqueues to `waiting`. A **no-progress
  guard** (`scheduler.py:80-91`) raises `RuntimeError` to avoid infinite spin.
- **PagedAttention** (`attention.py:283`): one `block_table` entry per token so
  decode chunks can straddle non-adjacent physical blocks; online softmax across
  chunks. Masked lanes point at block 0 to keep address math in-bounds
  (`attention.py:355-356`).
- **Flash Attention** (`attention.py:112`): variable-length prefill, online
  softmax with `m_i`/`l_i` recurrence, block sizes chosen by `head_dim`.
- **KV cache pool** (`model_runner.py:247`): one `torch.zeros` of shape
  `(2, num_layers, max_blocks, block_size, num_kv_heads, head_dim)`, sharded
  across `world_size`; size derived from `gpu_memory_utilization` after warmup.
- **Prefix caching** (`block_manager.py`): `xxhash.xxh64` chained by prefix; only
  full blocks are hashed. Cross-sequence reuse is **deliberately disabled** — see
  `block_manager.py:62-73` (the prefill kernel ignores `block_tables` and Qwen3
  derives RoPE positions from `cu_seqlens_q`, both wrong by `num_cached_tokens`).
  Do not "fix" this without reading the comment and `HowToApproachvLLM.md` §3.3.
- **Tensor parallelism** (`linear.py`, `embedding_head.py`): column/row parallel
  with `weight_loader` per-parameter callables; `RowParallelLinear` does
  `all_reduce(SUM)`; `QKVColumnParallelLinear` is GQA-aware. `world_size>1` spawns
  worker procs via `multiprocessing` "spawn" + shared-memory RPC with NCCL over
  `tcp://localhost:12345` (`llm_engine.py:13,29-37`, `model_runner.py:26,125-180`).
- **CUDA graphs** (`model_runner.py:406`): captured for batch sizes
  `[1,2,4,8] + range(16, max_bs+1, 16)`; smallest fitting graph replayed in
  `run_model` (`model_runner.py:354-378`). Skipped when `enforce_eager=True`.
- **Model dispatch** (`model_runner.py:32-69`): a `match` on the HF checkpoint
  *directory basename*. Only exact matches `'Qwen3-0.6B'` and
  `'Llama-3.2-1B-Instruct'` are accepted; anything else raises. Adding a model
  means adding a `case` branch (and a file under `models/`).

## Conventions

- **Imports:** models use `from myvllm.layers import *` (wildcard re-export from
  `layers/__init__.py`); keep `layers/__init__.py` re-exporting when you add a
  layer. `utils/__init__.py` only re-exports the context helpers, not the loader.
- **Type hints:** modern syntax preferred (`int | None`, `list[Sequence]`,
  `tuple[...]`). `llama.py` still uses `typing.Tuple` — match the surrounding file.
  Annotate signatures; locals/attrs are usually unannotated by convention.
- **Docstrings:** sparse. Module docstring only in `benchmark_decoding.py`.
  Triton kernels in `attention.py` use Google-style Args/Returns; most other
  functions use inline `#` comments describing steps — follow that locally.
- **`@torch.compile`** is on hot-path forwards (activation, layernorm, rotary,
  sampler). Preserve it unless you have a reason.
- **`__main__` blocks:** most layer/model files include a self-contained
  microbenchmark; do not delete these.
- **"Fixed:" comments** in `main.py` track corrections to upstream values —
  leave them.
- Known typos to be aware of (not necessarily worth fixing): `slided_weight`
  (`linear.py:106,219`), `activateion` (`qwen3.py:127`).

## Gotchas

- `vllm>=0.15.0` is a dependency only because `benchmark_tps.py` compares against
  it. The `myvllm` engine itself does **not** import `vllm` at runtime.
- `setup.py` is a stale legacy stub (pins `python_requires="==3.11.14"`, only
  `torch`). The source of truth is `pyproject.toml`.
- `SamplingParams.__post_init__` forbids greedy (`temperature > 1e-10`,
  `sampling_parameters.py:12`). Don't add a greedy path without updating it.
- The `weight_loader` callables on parameters (in `linear.py`) are **not** invoked
  by the default load path — `loader.py` does its own `torch.cat` merging for QKV
  and gate_up directly into `param.data`. If you wire up the per-parameter loaders,
  update both paths.
- Scheduler init order matters (see `HowToApproachvLLM.md` §6.2 and
  `llm_engine.py:42-46`); do not reorder `ModelRunner` vs `Scheduler` construction.
- The `Context` is a module-level singleton (`context.py:16`), set per iteration
  by `ModelRunner` and read by `Attention`/`ParallelLMHead`. Don't thread it
  through call sites — use `get_context()`/`set_context()`.

## Before You Commit

- Run `uv run pytest tests/ -v` if you touched `engine/` or `layers/attention.py`.
- Smoke-run `uv run python main.py` if you touched the engine or model code
  (requires a GPU and network access to download `Qwen/Qwen3-0.6B`).
- Do not commit changes to `uv.lock` unless you intentionally changed deps.
