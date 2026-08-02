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

    # ask, fully local (first run also downloads the qwen answer model, ~7gb)
    uv run python e30_rag.py ask "how do i remove the rear wheel bearing?"
    uv run python e30_rag.py ask            # interactive

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
    npm run ask:7b -- "..."          # local, bigger 7b model

the `--` is just how npm passes your question (and any extra flags) through.

## examples

fully local (the default) — loads the qwen model, then prints the answer + pages:

    $ uv run python e30_rag.py ask "how do i remove the rear wheel bearing?"
    Loading colqwen2...
    Loading local model Qwen/Qwen2.5-VL-3B-Instruct on mps...

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

interactive (run `ask` with no question):

    $ uv run python e30_rag.py ask
    Loading colqwen2...
    Loading local model Qwen/Qwen2.5-VL-3B-Instruct on mps...
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
    --llm-model M   answer model. default is Qwen/Qwen2.5-VL-3B-Instruct for local
                    and claude-sonnet-4-6 for anthropic. e.g. Qwen/Qwen2.5-VL-7B-Instruct
                    locally if you have the ram, or claude-opus-4-7 for the best claude answers
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

- everything runs on apple silicon (mps). local mode defaults to qwen2.5-vl-3b, which
  fits comfortably on a 32gb mac. the 7b is better but wants a lot more ram (it swaps
  hard on 32gb), so it's opt-in: `--llm-model Qwen/Qwen2.5-VL-7B-Instruct`.
- local mode needs no api key. claude mode reads `ANTHROPIC_API_KEY` from `.env`.
- accuracy vs privacy: the local 3b is fine for offline/private use, but on dense spec
  tables it can misread the fine print (small numbers like bore/stroke, torque, wheel
  sizes) and occasionally invent a value. for spec-critical lookups use `--provider
  anthropic` — claude reads the fine print far more reliably — or the local 7b, which is
  noticeably better at tables than the 3b (at the ram cost above). retrieval is the same
  either way; it's only the answer model that differs.
- the leann text cli also gets installed if you ever want plain-text rag over ocr'd
  docs; this project uses the image path instead.

## license

mit
