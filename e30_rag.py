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
    uv run python e30_rag.py build                      # index docs/*.pdf
    uv run python e30_rag.py ask "remove the rear wheel bearing"          # local Qwen
    uv run python e30_rag.py ask --provider anthropic "..."               # use Claude
    uv run python e30_rag.py ask --top-k 8 "..."                          # more pages
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
    _setup_import_paths()
    _silence_deps()
    import fitz  # PyMuPDF — cheap page-count pass for the progress bar
    from tqdm import tqdm
    from apps.colqwen_rag import ColQwenRAG  # loads torch/pdf2image; model loads in __init__
    from leann_multi_vector import LeannMultiVector

    docs_dir = Path(args.docs)
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {docs_dir}/. Drop your manuals there and re-run.")

    # Cheap page counts (no rendering) so the progress bar has a total + ETA.
    page_counts = {}
    for pdf in pdfs:
        d = fitz.open(str(pdf))
        page_counts[pdf] = d.page_count
        d.close()
    total_pages = sum(page_counts.values())
    print(f"Found {len(pdfs)} PDF(s), {total_pages} pages total, rendering at {args.dpi} DPI:")
    for pdf in pdfs:
        print(f"  {pdf.name}: {page_counts[pdf]} pages")

    print(f"Loading {args.model} (uses cached weights after the first run)...")
    with contextlib.redirect_stdout(io.StringIO()):
        colqwen = ColQwenRAG(args.model)

    # Fresh rebuild: clear this index's persisted page images (regenerable).
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = str(INDEX_DIR / args.index)
    images_dir = _images_dir(args.index)
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    mv = LeannMultiVector(index_path=index_path, dim=128, embedding_model_name=args.model)
    mv.create_collection()

    # Stream: render -> save page image to disk -> embed -> insert vectors only.
    # We save images ourselves and pass image=None so full-res PIL pages are NOT
    # held in RAM (LeannMultiVector otherwise keeps every page image in memory
    # until create_index — ~20GB at high DPI). ask() loads pages back by path.
    doc_id, dim = 0, None
    with tqdm(total=total_pages, desc="Embedding pages", unit="pg") as pbar:
        for pdf in pdfs:
            for page_no, img in _render_pdf_pages(str(pdf), dpi=args.dpi):
                img.save(_page_image_path(args.index, doc_id), "PNG")
                vecs = _embed_image(colqwen, img)  # (P, 128)
                dim = vecs.shape[-1]
                mv.insert(
                    {
                        "doc_id": doc_id,
                        "filepath": f"{pdf.stem} — p.{page_no}",  # returned by get_metadata()
                        "colbert_vecs": vecs,
                        "image": None,  # already saved to disk; keep it out of RAM
                    }
                )
                doc_id += 1
                pbar.update(1)
                pbar.set_postfix_str(f"{pdf.stem} p.{page_no}")
                del img

    if dim is not None:
        mv.dim = dim  # ensure persisted meta matches real embedding dim
    print(f"\nBuilding index from {doc_id} pages (writing to disk)...")
    mv.create_index()
    print(f"Done. Index written to {index_path}.* ({doc_id} pages).")
    print('Ask with:  uv run python e30_rag.py ask "your question"')


def _b64_png(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
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


def _load_local_vlm(model_name: str):
    """Load a local vision-language model for answer generation (once).

    Uses AutoModelForImageTextToText so any Qwen2-VL / Qwen2.5-VL checkpoint (or
    other compatible VLM) works via --llm-model. Runs on Apple-Silicon MPS if
    available, else CUDA, else CPU. Returns (model, processor, device).
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    elif torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    print(f"Loading local model {model_name} on {device} (first run downloads weights)...")
    model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=dtype).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor, device


def _answer_local(model, processor, device, question: str, pages) -> str:
    """Generate an answer fully locally from the retrieved page images."""
    import torch
    from qwen_vl_utils import process_vision_info

    content = [{"type": "text", "text": _build_prompt(question)}]
    for citation, img in pages:
        content.append({"type": "text", "text": f"[Manual page: {citation}]"})
        content.append({"type": "image", "image": img})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=4096)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def _answer_anthropic(question: str, pages, model: str) -> str:
    import anthropic

    content = [{"type": "text", "text": _build_prompt(question)}]
    for citation, img in pages:
        content.append({"type": "text", "text": f"[Manual page: {citation}]"})
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": _b64_png(img)},
            }
        )
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    resp = client.messages.create(
        model=model, max_tokens=4096, messages=[{"role": "user", "content": content}]
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def cmd_ask(args) -> None:
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
        llm_model = args.llm_model or "Qwen/Qwen2.5-VL-3B-Instruct"
        model, processor, device = _load_local_vlm(llm_model)

        def generate(question, pages):
            return _answer_local(model, processor, device, question, pages)
    else:
        llm_model = args.llm_model or "claude-sonnet-4-6"

        def generate(question, pages):
            return _answer_anthropic(question, pages, llm_model)

    def answer_one(question: str) -> None:
        q_vecs = _embed_query(colqwen, question)  # (T, 128)
        # Exact MaxSim over ALL pages (no ANN approximation) for best retrieval.
        results = mv.search_exact_all(q_vecs, topk=args.top_k)
        pages = []
        for _score, doc_id in results:
            meta = mv.get_metadata(doc_id) or {}
            citation = meta.get("filepath") or f"doc {doc_id}"
            img_path = _page_image_path(args.index, doc_id)
            if img_path.exists():
                pages.append((citation, Image.open(img_path)))
        if not pages:
            print("No matching pages found.")
            return
        answer = generate(question, pages)
        print("\n=== ANSWER ===\n" + answer)
        print("\n=== SOURCES (open these manual pages) ===")
        for citation, _img in pages:
            print(f"  • {citation}")

    if args.question:
        answer_one(" ".join(args.question))
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

    b = sub.add_parser("build", help="Build the page-image index from docs/*.pdf")
    b.add_argument("--docs", default=str(DOCS), help="Directory of PDFs (default: docs/)")
    b.add_argument("--index", default="e30", help="Index name (default: e30)")
    b.add_argument("--model", choices=["colqwen2", "colpali"], default="colqwen2")
    b.add_argument(
        "--dpi", type=int, default=150,
        help="Page render DPI (default: 150 — saturates ColQwen + Claude; higher just wastes time)",
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
        help="Answer model (default: Qwen/Qwen2.5-VL-3B-Instruct for local, claude-sonnet-4-6 for anthropic; "
             "use Qwen/Qwen2.5-VL-7B-Instruct locally if you have the RAM)",
    )
    a.add_argument("--top-k", type=int, default=5, help="Pages to retrieve (default: 5)")
    a.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
