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


def _load_local_vlm(model_name: str):
    """Load a local vision-language model for answer generation (once).

    Uses AutoModelForImageTextToText so any Qwen2-VL / Qwen2.5-VL checkpoint (or
    other compatible VLM) works via --llm-model. Runs on Apple-Silicon MPS if
    available, else CUDA, else CPU. Returns (model, processor, device).
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if torch.backends.mps.is_available():
        # bfloat16, not float16: fp16's narrow exponent range overflows to inf/nan in
        # Qwen2.5-VL's vision tower on MPS, which crashes generation with "probability
        # tensor contains inf/nan". bf16 has fp32's exponent range and stays stable.
        device, dtype = "mps", torch.bfloat16
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
        # any residual inf/nan in the logits.
        generated = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
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
        answer = generate(question, pages)
        print("\n=== ANSWER ===\n" + answer)
        print("\n=== SOURCES (open these manual pages) ===")
        for page in pages:
            print(f"  • {page.citation}")

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
        help="Answer model (default: Qwen/Qwen2.5-VL-3B-Instruct for local, claude-sonnet-4-6 for anthropic; "
             "use Qwen/Qwen2.5-VL-7B-Instruct locally if you have the RAM)",
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
