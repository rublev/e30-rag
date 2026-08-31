<p align="center">
  <img src="logo.jpg" alt="e30 rag" width="420">
</p>

# e30 rag

local multimodal rag over my bmw e30 repair manuals. ask something like "how do i
remove the rear wheel bearing?" and it finds the right manual pages and gives you an
answer that cites them.

the manuals are scanned / image-heavy and not really ocr'd, so instead of searching
text it embeds each pdf page as an image (colqwen2 via leann) and searches those. the
top matching pages then get read by a vision model that writes the answer. you can run
that last step two ways:

- **fully local** (default) — a qwen vision model runs on your machine, no api, nothing
  leaves the box.
- **claude** — send the pages to anthropic instead (better answers, needs an api key).

retrieval + embeddings are always local either way.

## setup

you need [uv](https://github.com/astral-sh/uv). an anthropic api key is optional (only
if you want to use claude for the answer step).

    git clone <this repo> e30
    cd e30

    # the retrieval code lives in the leann repo, clone it into vendor/
    git clone https://github.com/StarTrail-org/LEANN.git vendor/LEANN

    uv sync                 # makes .venv with everything

    # only if you want to use claude:
    cp .env.example .env    # then put your key in it: ANTHROPIC_API_KEY=sk-ant-...

## use

    # put your own manual pdfs in docs/ (they're gitignored, bring your own)

    # build the index. first run downloads the colqwen2 model (~4gb), slow once.
    uv run python e30_rag.py build

    # ask, fully local (first run also downloads the 7b answer model, ~16gb)
    uv run python e30_rag.py ask "how do i remove the rear wheel bearing?"
    uv run python e30_rag.py ask --llm-4bit  # interactive (see notes on 4-bit)

    # or use claude for the answer step
    uv run python e30_rag.py ask --provider anthropic "front control arm bushing torque"

answers end with the pages they used, e.g. `bentley-e30 — p.412`, so you can open the
pdf there for the diagrams.

## shortcuts

if you have node, there's a package.json wrapping the common commands:

    npm run build                    # build the index
    npm run ask -- "your question"   # ask, fully local (default)
    npm run chat                     # interactive
    npm run ask:claude -- "..."      # answer with claude
    npm run ask:4bit -- "..."        # local 7b in 4-bit (leaner/faster; use for interactive)

the `--` is just how npm passes your question (and any extra flags) through.

## examples

fully local (the default) — loads the qwen model, then prints the answer + pages:

    $ uv run python e30_rag.py ask "how do i remove the rear wheel bearing?"
    Loading colqwen2...
    Loading local model Qwen/Qwen2.5-VL-7B-Instruct on cuda in 8bit...

    === ANSWER ===
    1. raise and support the rear of the car, remove the wheel.
    2. unstake and remove the axle/drive-flange nut, then pull the flange.
    3. press the hub out, then drive the bearing from the trailing arm...
    (torque values / special tools as printed on the pages below)
    Pages used: bentley-e30 — p.412, bentley-e30 — p.413

    === SOURCES (open these manual pages) ===
      • bentley-e30 — p.412
      • bentley-e30 — p.413

same question via claude (`--provider anthropic`):

    $ uv run python e30_rag.py ask --provider anthropic "front control arm bushing torque"
    Loading colqwen2...

    === ANSWER ===
    torque the control-arm-to-subframe bolts to the figure printed on the page,
    done up at normal ride height so the bushing sits neutral...
    Pages used: bentley-e30 — p.201

    === SOURCES (open these manual pages) ===
      • bentley-e30 — p.201

interactive (run `ask` with no question; `--llm-4bit` keeps vram free for the repl):

    $ uv run python e30_rag.py ask --llm-4bit
    Loading colqwen2...
    Loading local model Qwen/Qwen2.5-VL-7B-Instruct on cuda in 4bit...
    Interactive mode. Ask about your E30; type 'quit' to exit.

    🔧 e30> engine oil type and capacity?
    === ANSWER ===
    m20b25: use the grade and capacity listed for the 325i on the page...
    Pages used: bentley-e30 — p.020

    🔧 e30> quit

## options

`build` (make the index from docs/*.pdf):

    --docs DIR      folder of pdfs (default: docs/)
    --index NAME    name for the index (default: e30)
    --model M       retrieval model, colqwen2 (default) or colpali
    --dpi N         page render dpi (default: 120, keeps spec tables legible; higher just wastes time)

`ask` (query it):

    --provider P    local (default) or anthropic
    --llm-model M   answer model. default is Qwen/Qwen2.5-VL-7B-Instruct for local
                    and claude-sonnet-4-6 for anthropic (e.g. claude-opus-4-7 for the
                    best claude answers)
    --llm-4bit      load the local 7b in 4-bit (~5.5gb) instead of the 8-bit default
                    (~8gb): faster and leaves more vram for a big --max-k, at a small
                    quality cost. use it for interactive mode, where the retriever stays
                    resident alongside the answer model
    --max-k N       most pages to send the answer model (default: 8). by default the
                    count is dynamic: it keeps the best-matching page plus any others
                    scoring close to it, so a specific question uses few pages and a
                    broad one uses more (up to this cap)
    --min-k N       fewest pages to send when going dynamic (default: 3)
    --keep-ratio R  keep pages scoring >= R x the top page's score (default: 0.9;
                    lower = more pages, higher = stricter)
    --top-k N       force exactly N pages, turning the dynamic behavior off
    --index NAME    which index to query (default: e30)
    --model M       retrieval model, must match what you built with (default: colqwen2)

run `ask` with no question for interactive mode.

## make it match your car

edit vehicle.md with your car's details. it gets prepended to every question so the
model picks the right procedure (e.g. m20 six-cylinder vs m10 four-cylinder). there's
a spot for your vin, look it up on realoem.com and copy the decoded specs in.

## notes

- local answer model runs on an nvidia gpu (cuda) in 8-bit by default — the 7b fits a
  16gb card (e.g. a 4080) because single-shot `ask` frees the ~4.5gb retriever before
  loading the answer model. add `--llm-4bit` for more headroom, or for interactive mode
  where both models stay resident. apple silicon (mps) and cpu have no bitsandbytes and
  load full precision instead — there the 7b wants a lot of ram and the 3b
  (`--llm-model Qwen/Qwen2.5-VL-3B-Instruct`) is the saner choice.
- ask specific, component-level questions ("e30 front suspension strut removal", "m20
  valve clearance") — the retriever matches page *content*, so broad prompts ("teach me
  to build a race car") drift to the wrong book.
- local mode needs no api key. claude mode reads `ANTHROPIC_API_KEY` from `.env`.
- accuracy vs privacy: the local 7b is solid on dense spec tables, but a quantized vision
  model can still misread fine print (bore/stroke, torque, wheel sizes) now and then. for
  spec-critical lookups use `--provider anthropic` — claude reads the fine print more
  reliably. retrieval is identical either way; only the answer model differs.
- the leann text cli also gets installed if you ever want plain-text rag over ocr'd
  docs; this project uses the image path instead.
- see CLAUDE.md for the transformers version pin and the 4.53.x workarounds — read it
  before bumping deps or touching the model loader.

## license

mit
