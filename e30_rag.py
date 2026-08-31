#!/usr/bin/env python3
"""
e30_rag.py — multimodal RAG over BMW E30 repair-manual PDFs.

Why this exists (instead of `python -m apps.colqwen_rag`):
  The stock LEANN ColQwen app is retrieval-only — its `ask` has a literal
  "TODO: add answer generation" and its `search` prints doc IDs, not page
  numbers. This wrapper reuses LEANN's ColQwen model + multi-vector index but
  adds the three pieces we need:
    1. Page-image rendering via PyMuPDF (no poppler/pdf2image system dep).
    2. Real answers: the retrieved *page images* are read by a vision LLM —
       either a local Qwen model (fully offline, default) or Anthropic Claude.
    3. Model-awareness + citations: `vehicle.md` is prepended to every prompt,
       and each answer lists the exact manual pages it used.

The local path needs no API key. For --provider anthropic, the key is read from
a local `.env` (gitignored). See `.env.example`.

Usage:
    uv run python e30_rag.py build                      # index docs/*.pdf (incremental: only new/changed)
    uv run python e30_rag.py ask "remove the rear wheel bearing"          # local Qwen
    uv run python e30_rag.py ask --provider anthropic "..."               # use Claude
    uv run python e30_rag.py ask --max-k 12 "..."                         # allow more pages (dynamic-k)
    uv run python e30_rag.py ask --top-k 5 "..."                          # force exactly 5 pages
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, NamedTuple

# --- transformers 4.53.x chat-template shim ------------------------------
# vidore/colqwen2-v1.0 ships additional_chat_templates/sentence_transformers.jinja;
# transformers 4.53.x mis-resolves it to None inside ProcessorMixin.from_pretrained
# and crashes: TypeError: expected str, bytes or os.PathLike object, not NoneType.
# ColQwen2/ColPali are retrieval encoders that never use a chat template, and we are
# pinned to 4.53.x (colpali needs >=4.53.1, LEANN caps <4.54), so we disable the
# spurious template discovery. Safe: no chat template is consumed at inference.
try:
    import transformers.processing_utils as _tf_processing_utils

    _tf_processing_utils.list_repo_templates = lambda *a, **k: []
except Exception:
    pass
# -------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "LEANN"
MULTIVEC = VENDOR / "apps" / "multimodal" / "vision-based-pdf-multi-vector"
DOCS = ROOT / "docs"
INDEX_DIR = ROOT / "indexes"
VEHICLE = ROOT / "vehicle.md"
ENV_FILE = ROOT / ".env"


def _load_env() -> None:
    """Load KEY=VALUE lines from ./.env into the environment (no dependency).

    A real shell env var wins over the file (setdefault), so `export
    ANTHROPIC_API_KEY=...` still overrides `.env`.
    """
    if not ENV_FILE.exists():
        return
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _setup_import_paths() -> None:
    """Put the vendored LEANN repo on sys.path so `apps.colqwen_rag` and
    `leann_multi_vector` import without installing the repo itself."""
    for p in (str(VENDOR), str(MULTIVEC)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _silence_deps() -> None:
    """Quiet noisy third-party chatter so the CLI output stays readable.

    Suppresses: colpali_engine's "FastPlaid is not installed" INFO log (fired on
    import) and transformers' "torch_dtype is deprecated" warning (the vendored
    ColQwen loader still passes torch_dtype). Call before importing the vendored
    LEANN / colpali stack. The vendored `ColQwenRAG.__init__` also does a bare
    print(); that's swallowed with redirect_stdout at each call site.
    """
    import logging

    logging.disable(logging.INFO)
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass


def _images_dir(index: str) -> Path:
    """Directory of persisted page images (matches LeannMultiVector's naming)."""
    return INDEX_DIR / f"{index}.images"


def _page_image_path(index: str, doc_id: int) -> Path:
    return _images_dir(index) / f"doc_{doc_id}.png"


def _sources_path(index: str) -> Path:
    """Manifest of indexed PDFs: content hash + assigned doc_id range per file.

    This is what makes builds incremental — it's the only record of *which* PDFs
    (and which exact bytes) are already in the index.
    """
    return INDEX_DIR / f"{index}.sources.json"


def _labels_path(index: str) -> Path:
    return INDEX_DIR / f"{index}.labels.json"


def _emb_path(index: str) -> Path:
    return INDEX_DIR / f"{index}.emb.npy"


def _meta_path(index: str) -> Path:
    return INDEX_DIR / f"{index}.meta.json"


def _file_sha256(path: Path) -> str:
    """Stream a file's SHA-256 so we detect content edits without trusting mtimes.

    Chunked so a few-hundred-MB manual never lands in RAM all at once.
    """
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json_atomic(path: Path, obj) -> None:
    """Write JSON via a temp file + os.replace so a crash can't truncate the file."""
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _save_npy_atomic(path: Path, arr) -> None:
    """np.save via a temp file + os.replace (atomic within the same directory).

    We hand np.save an open file object because, given a path that doesn't end in
    '.npy' (our '<name>.npy.tmp'), it would otherwise append a second '.npy'.
    """
    import numpy as np

    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


def _meta_dict(model: str, dim: int) -> dict:
    """Sidecar meta matching LeannMultiVector's schema (ask reads 'dimensions')."""
    return {
        "version": "1.0",
        "backend_name": "hnsw",
        "embedding_model": model,
        "embedding_mode": "custom",
        "dimensions": dim,
        "backend_kwargs": {
            "distance_metric": "mips",
            "M": 16,
            "efConstruction": 500,
            "is_compact": False,
            "is_recompute": False,
        },
        "is_compact": False,
        "is_pruned": False,
    }


def _load_manifest(index: str) -> dict | None:
    path = _sources_path(index)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _reset_index(index: str) -> None:
    """Delete every persisted artifact for this index (for a clean full rebuild).

    Covers the incremental sidecars we own plus the FAISS ANN files a legacy
    (pre-incremental) build may have left behind.
    """
    images_dir = _images_dir(index)
    if images_dir.exists():
        shutil.rmtree(images_dir)
    for p in (
        _emb_path(index),
        _labels_path(index),
        _meta_path(index),
        _sources_path(index),
        INDEX_DIR / f"{index}.index",
        INDEX_DIR / f"{index}.ids.txt",
    ):
        if p.exists():
            p.unlink()


def _render_pdf_pages(pdf_path: str, dpi: int):
    """Render every page of a PDF to a PIL image using PyMuPDF (fitz).

    PyMuPDF has no external system dependency (unlike pdf2image, which needs
    poppler), and handles the mixed text/scanned pages in these manuals fine.
    Yields (page_number, PIL.Image) lazily so we never hold a whole PDF in RAM.
    """
    import fitz  # PyMuPDF
    from PIL import Image

    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    try:
        for i in range(doc.page_count):
            pix = doc.load_page(i).get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            yield i + 1, img
    finally:
        doc.close()


def _embed_image(colqwen, img):
    """ColQwen multi-vector embedding for one page image -> (P, 128) float32.

    We embed page-by-page (not the stock batched torch.cat) because pages of
    different pixel sizes produce a different number of patch vectors P, which
    makes a single cat() across pages fail.
    """
    import torch

    with torch.no_grad():
        inputs = colqwen.processor.process_images([img]).to(colqwen.device)
        out = colqwen.model(**inputs)
    return out.cpu()[0].numpy().astype("float32")


def _embed_query(colqwen, text: str):
    """ColQwen multi-vector embedding for a text query -> (T, 128) float32."""
    import torch

    with torch.no_grad():
        inputs = colqwen.processor.process_queries([text]).to(colqwen.device)
        out = colqwen.model(**inputs)
    return out.cpu()[0].numpy().astype("float32")


def cmd_build(args) -> None:
    """Incrementally (re)build the page-image index from docs/*.pdf.

    Only PDFs that are new or whose bytes changed since the last build are
    rendered + embedded (the expensive GPU step). Unchanged PDFs are skipped and
    their existing vectors reused; PDFs removed from docs/ have their vectors and
    page images pruned. Change detection is by SHA-256 of the PDF, tracked in
    <index>.sources.json.

    Retrieval (`ask`) scores exact MaxSim over <index>.emb.npy, so we maintain
    only what that path reads — <index>.emb.npy, .labels.json, .meta.json and the
    page PNGs — and do NOT rebuild the FAISS HNSW ANN index (unused here, and the
    slow part of a from-scratch build).
    """
    _setup_import_paths()
    _silence_deps()
    import fitz  # PyMuPDF — cheap page-count pass for the progress bar
    import numpy as np
    from tqdm import tqdm

    docs_dir = Path(args.docs)
    pdfs = sorted(docs_dir.rglob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {docs_dir}/. Drop your manuals there and re-run.")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    emb_path, labels_path = _emb_path(args.index), _labels_path(args.index)

    # --- Load the prior manifest; invalidate it if it can't be trusted. ---
    manifest = _load_manifest(args.index)
    if manifest is not None and (
        manifest.get("model") != args.model
        or int(manifest.get("dpi", -1)) != int(args.dpi)
    ):
        print(
            f"Existing index was built with model={manifest.get('model')} "
            f"dpi={manifest.get('dpi')}; you asked for model={args.model} dpi={args.dpi}. "
            "Those change every embedding — rebuilding the whole index."
        )
        manifest = None

    # Load + validate the survivor sidecars. Any inconsistency => clean rebuild.
    old_labels: list[dict] = []
    old_emb = None
    if manifest is not None:
        if emb_path.exists() and labels_path.exists():
            old_labels = json.loads(labels_path.read_text(encoding="utf-8"))
            old_emb = np.load(emb_path, mmap_mode="r")  # header-only; rows read on demand
            if old_emb.shape[0] != len(old_labels):
                print("Index sidecars are inconsistent (row mismatch) — rebuilding.")
                manifest, old_labels, old_emb = None, [], None
        else:
            manifest = None  # manifest present but its data files are gone

    if manifest is None:
        # Fresh dir, invalidated index, or a legacy (pre-manifest) index: clean slate.
        _reset_index(args.index)
        prior_sources: dict = {}
        next_doc_id = 0
        dim: int | None = None
        old_labels, old_emb = [], None
    else:
        prior_sources = manifest.get("sources", {})
        next_doc_id = int(manifest.get("next_doc_id", 0))
        dim = int(manifest["dim"]) if manifest.get("dim") else None

    # --- Hash the current PDFs and diff against the manifest. ---
    print(f"Scanning {len(pdfs)} PDF(s) in {docs_dir}/ ...")
    current: dict[str, dict] = {
        pdf.name: {"path": pdf, "sha256": _file_sha256(pdf), "size": pdf.stat().st_size}
        for pdf in pdfs
    }
    new_names = sorted(n for n in current if n not in prior_sources)
    changed_names = sorted(
        n for n in current
        if n in prior_sources and current[n]["sha256"] != prior_sources[n].get("sha256")
    )
    deleted_names = sorted(n for n in prior_sources if n not in current)
    unchanged_names = sorted(
        n for n in current
        if n in prior_sources and current[n]["sha256"] == prior_sources[n].get("sha256")
    )
    to_embed = new_names + changed_names  # only these hit the GPU

    # doc_ids to drop: a changed source's stale range + every deleted source's range.
    retired_doc_ids: set[int] = set()
    for n in changed_names + deleted_names:
        s = prior_sources[n]
        start = int(s["doc_id_start"])
        retired_doc_ids.update(range(start, start + int(s["page_count"])))

    def _fmt(names: list[str]) -> str:
        return ", ".join(names) if names else "(none)"

    print(f"  New:       {_fmt(new_names)}")
    print(f"  Changed:   {_fmt(changed_names)}")
    print(f"  Deleted:   {_fmt(deleted_names)}")
    print(f"  Unchanged: {len(unchanged_names)} file(s), skipped")

    if not to_embed and not retired_doc_ids:
        print("Index is already up to date. Nothing to do.")
        return

    # --- Prune retired rows from the survivors (filters, never re-embeds). ---
    if old_emb is not None and retired_doc_ids:
        keep = [
            i for i, row in enumerate(old_labels)
            if int(row["doc_id"]) not in retired_doc_ids
        ]
        kept_labels = [old_labels[i] for i in keep]
        kept_emb = old_emb[keep] if keep else None  # fancy-index a memmap -> RAM copy of kept rows only
    else:
        kept_labels = old_labels
        kept_emb = old_emb  # memmap (or None); materialized by the vstack below

    # --- Embed new/changed sources; carry unchanged sources forward untouched. ---
    updated_sources: dict = {n: prior_sources[n] for n in unchanged_names}
    new_labels: list[dict] = []
    new_emb_parts: list = []

    if to_embed:
        page_counts = {}
        for n in to_embed:
            d = fitz.open(str(current[n]["path"]))
            page_counts[n] = d.page_count
            d.close()
        total_pages = sum(page_counts.values())
        print(
            f"Embedding {len(to_embed)} file(s), {total_pages} pages at {args.dpi} DPI "
            f"(reusing {len(unchanged_names)} unchanged):"
        )
        for n in to_embed:
            print(f"  {n}: {page_counts[n]} pages")

        print(f"Loading {args.model} (uses cached weights after the first run)...")
        from apps.colqwen_rag import ColQwenRAG  # pulls torch; only needed to embed
        with contextlib.redirect_stdout(io.StringIO()):
            colqwen = ColQwenRAG(args.model)

        _images_dir(args.index).mkdir(parents=True, exist_ok=True)

        # Stream: render -> save page PNG to disk -> embed -> keep vectors only.
        # Images go straight to disk (ask() reloads them by path) so full-res pages
        # never pile up in RAM. New/changed docs get fresh, monotonic doc_ids that
        # are never reused, so survivor rows and their PNG names stay valid.
        with tqdm(total=total_pages, desc="Embedding pages", unit="pg") as pbar:
            for n in to_embed:
                pdf = current[n]["path"]
                start_doc_id = next_doc_id
                pages = 0
                for page_no, img in _render_pdf_pages(str(pdf), dpi=args.dpi):
                    img.save(_page_image_path(args.index, next_doc_id), "PNG")
                    vecs = _embed_image(colqwen, img)  # (P, dim)
                    if dim is None:
                        dim = int(vecs.shape[-1])
                    img_path = str(_page_image_path(args.index, next_doc_id))
                    for seq_id, vec in enumerate(vecs):
                        new_emb_parts.append(np.asarray(vec, dtype=np.float32))
                        new_labels.append(
                            {
                                "id": f"{next_doc_id}:{seq_id}",
                                "doc_id": next_doc_id,
                                "seq_id": int(seq_id),
                                "filepath": f"{pdf.stem} — p.{page_no}",
                                "image_path": img_path,
                            }
                        )
                    next_doc_id += 1
                    pages += 1
                    pbar.update(1)
                    pbar.set_postfix_str(f"{pdf.stem} p.{page_no}")
                    del img
                updated_sources[n] = {
                    "sha256": current[n]["sha256"],
                    "size": current[n]["size"],
                    "page_count": pages,
                    "doc_id_start": start_doc_id,
                }

    # --- Assemble the final matrix + labels (survivors first, then new rows). ---
    new_emb = np.vstack(new_emb_parts) if new_emb_parts else None
    if kept_emb is not None and new_emb is not None:
        final_emb = np.vstack([np.asarray(kept_emb), new_emb])
    elif new_emb is not None:
        final_emb = new_emb
    elif kept_emb is not None:
        final_emb = np.asarray(kept_emb)
    else:
        final_emb = None
    final_labels = kept_labels + new_labels

    if final_emb is None or final_emb.shape[0] == 0:
        sys.exit("Refusing to write an empty index (no pages remain); left it untouched.")
    if final_emb.shape[0] != len(final_labels):
        sys.exit(
            f"Internal error: {final_emb.shape[0]} vectors vs {len(final_labels)} labels; "
            "aborted without touching the existing index."
        )
    if dim is None:
        dim = int(final_emb.shape[1])

    # --- Commit: atomic sidecar writes so a crash can't leave a half-updated index. ---
    _save_npy_atomic(emb_path, final_emb.astype(np.float32, copy=False))
    _write_json_atomic(labels_path, final_labels)
    _write_json_atomic(_meta_path(args.index), _meta_dict(args.model, dim))
    _write_json_atomic(
        _sources_path(args.index),
        {
            "version": 1,
            "model": args.model,
            "dpi": int(args.dpi),
            "dim": int(dim),
            "next_doc_id": int(next_doc_id),
            "sources": updated_sources,
        },
    )

    # --- Post-commit GC: only now drop pruned page images + any stale FAISS files. ---
    for doc_id in retired_doc_ids:
        img_file = _page_image_path(args.index, doc_id)
        if img_file.exists():
            img_file.unlink()
    for stale in (INDEX_DIR / f"{args.index}.index", INDEX_DIR / f"{args.index}.ids.txt"):
        if stale.exists():
            stale.unlink()

    total_pages_now = sum(int(s["page_count"]) for s in updated_sources.values())
    print(
        f"\nDone. Embedded {len(to_embed)} file(s), removed {len(deleted_names)}, "
        f"reused {len(unchanged_names)}. Index now: {len(updated_sources)} file(s), "
        f"{total_pages_now} pages, {final_emb.shape[0]} vectors -> {emb_path}"
    )
    print('Ask with:  pnpm ask "your question"')


def _b64_jpeg(img, *, max_edge: int = 1568, quality: int = 85) -> str:
    """Downscale to Anthropic's effective max resolution and JPEG-encode for transport.

    Claude downsamples any image whose long edge exceeds ~1568px anyway, and PNGs of
    scanned/photographic manual pages run multiple MB apiece — 8 of them blow past the
    ~32MB request cap (the 413 we hit at max_k). Resizing to 1568px + JPEG cuts each page
    to a few hundred KB with no loss of legibility, so a full max_k of pages fits in one
    request. Local inference is unaffected (the Qwen processor resizes in-memory itself).
    """
    from PIL import Image

    im = img.convert("RGB")  # JPEG has no alpha channel
    long_edge = max(im.size)
    if long_edge > max_edge:
        scale = max_edge / long_edge
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _vehicle_context() -> str:
    return VEHICLE.read_text(encoding="utf-8") if VEHICLE.exists() else "(no vehicle profile)"


def _build_prompt(question: str) -> str:
    return (
        "You are an expert BMW E30 mechanic assistant writing a repair guide. Answer the "
        "question using ONLY the attached repair-manual page images (each labelled with its "
        "source page). Be thorough and complete: give the full ordered procedure, every torque "
        "value, clearance, fluid type/capacity, tool size and special-tool number exactly as "
        "printed, plus any warnings or prerequisite steps shown. Do not omit steps. If the pages "
        "show several model variants, use the one that matches THIS car. If the attached pages "
        "don't fully cover it, say what's missing and which manual section to check. Always end "
        "with 'Pages used:' listing the page labels you relied on.\n\n"
        f"=== THIS CAR (vehicle profile) ===\n{_vehicle_context()}\n\n"
        f"=== QUESTION ===\n{question}\n"
    )


class RetrievedPage(NamedTuple):
    """One retrieved manual page plus everything the answer step needs.

    Bundles the page's id, its human-readable citation, its MaxSim score, and the
    loaded page image into a single named row. This replaces what used to be three
    parallel tuples — (score, doc_id), (citation, img), (citation, score) — whose
    field order flipped between producer and consumers and could silently desync.
    """

    doc_id: int
    citation: str  # e.g. "Bentley — p.214"
    score: float  # MaxSim, non-negative; higher = more relevant
    image: Any  # a PIL.Image.Image (typed loosely to avoid a top-level PIL import)


def _page_label(citation: str) -> dict:
    """The text block that captions each attached page image (same for both backends)."""
    return {"type": "text", "text": f"[Manual page: {citation}]"}


def _load_local_vlm(model_name: str, quant: str = "8bit"):
    """Load a local vision-language model for answer generation (once).

    Uses AutoModelForImageTextToText so any Qwen2-VL / Qwen2.5-VL checkpoint works
    via --llm-model. On CUDA the weights are loaded quantized via bitsandbytes so a
    7B model fits alongside the colqwen retriever on a 16 GB card:
      quant="8bit" (default) — ~8 GB, near-fp16 answer quality.
      quant="4bit"           — ~5.5 GB (NF4), faster + more VRAM for large --max-k.
    Apple-Silicon MPS / CPU have no bitsandbytes, so they load full precision.
    Returns (model, processor, device).
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)

    if torch.cuda.is_available() and quant in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig

        if quant == "4bit":
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            qcfg = BitsAndBytesConfig(load_in_8bit=True)
        print(f"Loading local model {model_name} on cuda in {quant} (first run downloads weights)...")
        # device_map places the quantized weights on the GPU; do NOT call .to() after.
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, quantization_config=qcfg, device_map={"": 0}
        )
        return model, processor, "cuda"

    # No CUDA quantization path (Apple-Silicon MPS / CPU): full precision. torch_dtype
    # (not dtype) is the kwarg transformers 4.53.x understands; bf16 on MPS avoids fp16
    # inf/nan in the vision tower, fp32 on CPU.
    if torch.backends.mps.is_available():
        device, dtype = "mps", torch.bfloat16
    elif torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    print(f"Loading local model {model_name} on {device} (first run downloads weights)...")
    model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=dtype).to(device)
    return model, processor, device


def _answer_local(model, processor, device, question: str, pages) -> str:
    """Generate an answer fully locally from the retrieved page images."""
    import torch
    from qwen_vl_utils import process_vision_info

    content = [{"type": "text", "text": _build_prompt(question)}]
    for page in pages:
        content.append(_page_label(page.citation))
        content.append({"type": "image", "image": page.image})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        # Greedy (do_sample=False): deterministic, reproducible answers for a factual
        # repair assistant, and it avoids the multinomial sampling path that crashes on
        # any residual inf/nan in the logits. repetition_penalty tames the greedy-decode
        # loops that quantized VLMs fall into (e.g. counting "10Nm, 20Nm, 30Nm…" forever);
        # 2048 new tokens is plenty for a page-grounded answer and caps any runaway.
        generated = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            repetition_penalty=1.15,
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def _answer_anthropic(question: str, pages, model: str) -> str:
    import anthropic

    content = [{"type": "text", "text": _build_prompt(question)}]
    for page in pages:
        content.append(_page_label(page.citation))
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64_jpeg(page.image)},
            }
        )
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    try:
        resp = client.messages.create(
            model=model, max_tokens=4096, messages=[{"role": "user", "content": content}]
        )
    except anthropic.RequestTooLargeError:
        # Even after downscaling, a large max_k of dense pages can exceed the request
        # cap. Return guidance instead of crashing (keeps interactive mode alive).
        return (
            f"[Anthropic request too large: {len(pages)} page images exceeded the size "
            "limit. Send fewer pages with a lower --max-k or a higher --keep-ratio.]"
        )
    text = "".join(block.text for block in resp.content if block.type == "text")
    if not text.strip():
        # No text block means the model returned only non-text content or hit a stop
        # condition before writing — surface it instead of printing a blank answer.
        print(
            f"  WARNING: Anthropic returned no text (stop_reason={resp.stop_reason!r}, "
            f"{len(resp.content)} block(s)).",
            file=sys.stderr,
        )
    return text


def _select_pages(results, *, min_k: int, max_k: int, keep_ratio: float):
    """Dynamically choose how many retrieved pages to hand to the answer LLM.

    `results` is the (score, doc_id) list from search_exact_all — MaxSim scores,
    sorted best-first, non-negative. Rather than a fixed top-k, we keep the best
    hit, then keep each following page only while its score stays within
    `keep_ratio` of the top score (an "elbow" cut on the score curve). A vague
    question whose pages all score similarly keeps more pages; a laser-specific
    question whose top hit dominates keeps few. Keeps at least `min_k` pages when
    that many were retrieved, and never more than `max_k` — the upper bound matters
    because every kept page is a full-res image the vision LLM must ingest, the real bottleneck
    (Claude token cost / local VLM memory), not retrieval.
    """
    if not results:
        return []
    top = results[0][0]
    kept = [results[0]]
    for score, doc_id in results[1:]:
        if len(kept) >= max_k:
            break
        below_floor = len(kept) < min_k
        # top == 0 means every page scored 0 (scores are non-negative) — skip the
        # ratio test so we fall back to the min_k floor instead of keeping max_k.
        close_to_best = top > 0 and score >= keep_ratio * top
        if below_floor or close_to_best:
            kept.append((score, doc_id))
        else:
            break  # scores fell off the cliff — stop here
    return kept


def cmd_ask(args) -> None:
    # Validate retrieval bounds up front, before the slow model load, so a bad flag
    # combo fails instantly. A fixed --top-k forces exactly that many pages (min=max);
    # otherwise the count is chosen dynamically within [min_k, max_k] from the score curve.
    if args.top_k is not None:
        if args.top_k < 1:
            sys.exit("--top-k must be >= 1.")
        min_k = max_k = args.top_k
    else:
        min_k, max_k = args.min_k, args.max_k
        if min_k < 1:
            sys.exit("--min-k must be >= 1.")
        if min_k > max_k:
            sys.exit(f"--min-k ({min_k}) cannot exceed --max-k ({max_k}).")
    if not 0.0 < args.keep_ratio <= 1.0:
        sys.exit("--keep-ratio must be in (0, 1].")

    _load_env()
    _setup_import_paths()
    _silence_deps()
    from PIL import Image
    from apps.colqwen_rag import ColQwenRAG
    from leann_multi_vector import LeannMultiVector

    index_path = INDEX_DIR / args.index
    meta_file = index_path.parent / f"{index_path.name}.meta.json"
    if not meta_file.exists():
        sys.exit(
            f"No index found at {index_path}.* — run `uv run python e30_rag.py build` first."
        )
    dim = int(json.loads(meta_file.read_text()).get("dimensions", 128))

    print(f"Loading {args.model}...")
    with contextlib.redirect_stdout(io.StringIO()):
        colqwen = ColQwenRAG(args.model)
    mv = LeannMultiVector(index_path=str(index_path), dim=dim, embedding_model_name=args.model)

    if args.provider == "local":
        llm_model = args.llm_model or "Qwen/Qwen2.5-VL-7B-Instruct"
        quant = "4bit" if args.llm_4bit else "8bit"
        # Lazy-load the big VLM: in single-shot mode answer_one frees the ~4.5 GB
        # retriever first (see free_retriever), so deferring the load until the first
        # generate() keeps colqwen and the 8-bit VLM from ever being co-resident — that's
        # what lets the 7B fit a 16 GB card. Cached in a dict so interactive loads once.
        _vlm = {}

        def generate(question, pages):
            if not _vlm:
                _vlm["h"] = _load_local_vlm(llm_model, quant=quant)
            model, processor, device = _vlm["h"]
            return _answer_local(model, processor, device, question, pages)
    else:
        llm_model = args.llm_model or "claude-sonnet-4-6"

        def generate(question, pages):
            return _answer_anthropic(question, pages, llm_model)

    def answer_one(question: str, free_retriever: bool = False) -> None:
        q_vecs = _embed_query(colqwen, question)  # (T, 128)
        # Exact MaxSim: every page is scored exactly (no ANN approximation);
        # search_exact_all returns the best max_k as the candidate pool, which
        # _select_pages then trims by score.
        results = mv.search_exact_all(q_vecs, topk=max_k)
        selected = _select_pages(results, min_k=min_k, max_k=max_k, keep_ratio=args.keep_ratio)
        pages, missing = [], 0
        for score, doc_id in selected:
            meta = mv.get_metadata(doc_id) or {}
            citation = meta.get("filepath") or f"doc {doc_id}"
            img_path = _page_image_path(args.index, doc_id)
            if not img_path.exists():
                # The retriever chose this page but its rendered image is gone
                # (stale/interrupted build). Don't silently answer from worse
                # pages — warn and skip, so a degraded answer isn't invisible.
                print(
                    f"  WARNING: skipping retrieved page '{citation}' — image missing at "
                    f"{img_path}. Re-run `build`.",
                    file=sys.stderr,
                )
                missing += 1
                continue
            with Image.open(img_path) as im:
                img = im.convert("RGB")  # load pixels now so we can close the file handle
            pages.append(RetrievedPage(doc_id, citation, score, img))
        if not pages:
            # Distinguish "retriever found nothing" from "images are missing on disk".
            if missing:
                print(f"All {missing} matching page(s) had missing images — re-run `build`.")
            else:
                print("No matching pages found.")
            return
        kind = "fixed" if args.top_k is not None else "dynamic-k"
        print(f"\nRetrieved {len(pages)} page(s) [{kind}, MaxSim-ranked]:")
        for page in pages:
            print(f"  [{page.score:7.2f}]  {page.citation}")
        if free_retriever:
            # Query's embedded and pages are retrieved, so the colqwen retriever (~4.5 GB)
            # is dead weight during generation — drop it to CPU so the local VLM gets the
            # whole card for the page-image prefill. Single-shot only; interactive keeps it
            # resident to embed the next question.
            import torch

            if torch.cuda.is_available():
                colqwen.model.to("cpu")
                torch.cuda.empty_cache()
        answer = generate(question, pages)
        print("\n=== ANSWER ===\n" + answer)
        print("\n=== SOURCES (open these manual pages) ===")
        for page in pages:
            print(f"  • {page.citation}")

    if args.question:
        answer_one(" ".join(args.question), free_retriever=(args.provider == "local"))
    else:
        print("Interactive mode. Ask about your E30; type 'quit' to exit.")
        while True:
            try:
                q = input("\n🔧 e30> ").strip()
            except EOFError:
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if q:
                answer_one(q)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal RAG over BMW E30 repair manuals (ColQwen retrieval + vision LLM)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser(
        "build",
        help="Incrementally (re)build the index from docs/*.pdf (embeds only new/changed PDFs)",
    )
    b.add_argument("--docs", default=str(DOCS), help="Directory of PDFs (default: docs/)")
    b.add_argument("--index", default="e30", help="Index name (default: e30)")
    b.add_argument("--model", choices=["colqwen2", "colpali"], default="colqwen2")
    b.add_argument(
        "--dpi", type=int, default=120,
        help="Page render DPI (default: 120 — keeps dense spec tables legible; Claude downsamples "
             "past ~1568px so higher mostly wastes disk/time)",
    )
    b.set_defaults(func=cmd_build)

    a = sub.add_parser("ask", help="Ask a question; get an answer + page citations")
    a.add_argument("question", nargs="*", help="Your question (omit for interactive mode)")
    a.add_argument("--index", default="e30")
    a.add_argument("--model", choices=["colqwen2", "colpali"], default="colqwen2")
    a.add_argument(
        "--provider", choices=["local", "anthropic"], default="local",
        help="Answer backend: local Qwen (offline) or anthropic Claude (default: local)",
    )
    a.add_argument(
        "--llm-model", default=None,
        help="Answer model (default: Qwen/Qwen2.5-VL-7B-Instruct for local, claude-sonnet-4-6 for anthropic)",
    )
    a.add_argument(
        "--llm-4bit", action="store_true",
        help="Load the local VLM in 4-bit NF4 instead of the 8-bit default (~5.5 GB vs ~8 GB): "
             "faster and leaves more VRAM for large --max-k, at a small quality cost.",
    )
    a.add_argument(
        "--top-k", type=int, default=None,
        help="Force a fixed number of pages (disables dynamic-k). Omit to choose dynamically.",
    )
    a.add_argument("--min-k", type=int, default=3, help="Dynamic-k floor: fewest pages to send (default: 3)")
    a.add_argument(
        "--max-k", type=int, default=8,
        help="Dynamic-k cap: most pages to send (default: 8). Kept low on purpose — every page is a "
             "full-res image the vision LLM must read, which is the real bottleneck, not retrieval.",
    )
    a.add_argument(
        "--keep-ratio", type=float, default=0.9,
        help="Dynamic-k threshold: keep pages scoring >= this fraction of the top page's MaxSim score "
             "(default: 0.9). Lower = more pages kept; higher = stricter.",
    )
    a.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
