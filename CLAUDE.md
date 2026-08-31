# CLAUDE.md — e30-rag maintainer notes

Context and gotchas that aren't obvious from the code. Read this before touching the
model loading, the dependency pins, or the answer path.

## the transformers version is pinned to 4.53.x — don't bump it casually

`pyproject.toml` pins `transformers>=4.46.1,<4.54`. Combined with colpali-engine
(`>=4.53.1,<4.58`), the only satisfiable window is **4.53.x** (uv resolves 4.53.3).

- **Floor** (`>=4.53.1`): colpali-engine needs it.
- **Ceiling** (`<4.54`): LEANN's vendored ColQwen retrieval path breaks on transformers
  `>=4.54` (a HybridCache API it uses was removed).

So both halves of the app only agree on 4.53.x. Raising the ceiling means patching LEANN
(or swapping the retriever) first — it's a project, not a version bump.

## two 4.53.x bugs are worked around in e30_rag.py — keep the workarounds

1. **chat-template shim** (top of `e30_rag.py`): `vidore/colqwen2-v1.0` ships an
   `additional_chat_templates/sentence_transformers.jinja`. transformers 4.53.x builds a
   wrong path for it, resolves it to `None`, then `open(None)`s it inside
   `ProcessorMixin.from_pretrained` → `TypeError: expected str, bytes or os.PathLike
   object, not NoneType`. The shim neutralizes `list_repo_templates` so the (unused)
   template is never discovered. Safe: ColQwen2/ColPali are retrieval encoders and never
   consume a chat template.
2. **`torch_dtype`, not `dtype`**: `_load_local_vlm`'s full-precision path must pass
   `torch_dtype=` to `from_pretrained`. The `dtype=` alias only exists in later
   transformers; on 4.53.x it gets forwarded to the model `__init__` and raises
   `unexpected keyword argument 'dtype'`.

## local answer model: 7B in 8-bit, fit onto a 16 GB GPU

- Default `--llm-model` is `Qwen/Qwen2.5-VL-7B-Instruct`, loaded in **8-bit** via
  bitsandbytes on CUDA (`--llm-4bit` switches to NF4 4-bit). 7B is the ceiling for a 16 GB
  card — 32B/72B don't fit even quantized, and the answer path is Qwen-specific
  (`qwen_vl_utils.process_vision_info`), so swapping model families needs code changes.
- **The VLM is lazy-loaded** (inside the `generate` closure) and single-shot `ask` sets
  `free_retriever=True`, which offloads colqwen (~4.5 GB) to CPU *before* the VLM loads.
  That ordering is load-bearing: without it, colqwen + the 8-bit 7B are co-resident at
  ~15.4 GB and OOM the card at load time. Interactive mode keeps colqwen resident (it's
  needed to embed each question), so use `--llm-4bit` there.
- Generation is greedy (`do_sample=False`) with `repetition_penalty=1.15`. The penalty is
  required: 8-bit greedy without it falls into runaway loops (e.g. counting torque values
  `10Nm, 20Nm, 30Nm…` until the token cap).
- MPS/CPU have no bitsandbytes, so `_load_local_vlm` loads full precision there (bf16 on
  MPS to avoid fp16 inf/nan in the vision tower).

## docs/ is local-only

`.gitignore` excludes all of `docs/` (PDFs are large + copyrighted) except `.gitkeep`,
plus `indexes/` and `.venv/`. The manuals never sync via git; each machine keeps its own
copies. Working flow is edit → commit → push (from the pop box) → `git pull` elsewhere.
