# parallax-fetcher

[![CI](https://github.com/Hamsinideshmukh/parallax-fetcher/actions/workflows/ci.yml/badge.svg)](https://github.com/Hamsinideshmukh/parallax-fetcher/actions/workflows/ci.yml)

A typed, tested command-line tool that fetches many pages at once, retries failures, respects
robots.txt and per-host rate limits, and saves clean text as JSONL. On top of that sits a small RAG
layer: embed the pages, search them by meaning, and ask an LLM questions that it must answer from the
fetched pages, as validated JSON with its sources.

## Quickstart (Windows, PowerShell)

Needs Python 3.12 or newer.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt -e .

python scripts\make_urls.py                        # 266 Python standard library pages -> data\urls.txt
fetcher fetch data\urls.txt -o data\pages.jsonl    # about 27 s at the default 10 requests/s
fetcher stats data\pages.jsonl
python scripts\to_chunks.py data\pages.jsonl       # overlapping chunks, ready to embed
```

If `activate` is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## Quickstart (macOS / Linux)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt -e .

python scripts/make_urls.py
fetcher fetch data/urls.txt -o data/pages.jsonl
fetcher stats data/pages.jsonl
python scripts/to_chunks.py data/pages.jsonl
```

## Your user agent

Say who you are (with a way to contact you) before crawling anything that isn't yours. Either pass
`--user-agent "your-bot/0.1 (+mailto:you@example.com)"`, or set it once per terminal:

```powershell
$env:FETCHER_USER_AGENT = "your-bot/0.1 (+mailto:you@example.com)"   # PowerShell
```

```bash
cp .env.example .env && set -a; source .env; set +a                 # macOS / Linux
```

## Search and ask (the RAG layer)

After the quickstart you have `data/chunks.jsonl`. Embed it once (the first run downloads a small
embedding model, about 130 MB, and runs on the CPU):

```powershell
fetcher index data\chunks.jsonl                          # -> data\index\
fetcher search "run a shell command and capture its output"
python scripts\eval_search.py                            # hit@1, hit@5 and MRR on 12 test questions
```

`search` ranks by meaning, not by matching words: the query above should find the `subprocess` page
even though it never says "subprocess". `eval_search.py` measures how often that works, using
`data/eval_queries.jsonl`: plain English questions, each paired with the page that should come back.

To ask questions, point it at any OpenAI-compatible chat API. Groq has a free tier
([get a key](https://console.groq.com/keys)); OpenAI, Gemini or a local Ollama work the same way:

```powershell
$env:LLM_API_KEY  = "gsk_your_key_here"
$env:LLM_BASE_URL = "https://api.groq.com/openai/v1"
$env:LLM_MODEL    = "openai/gpt-oss-120b"     # or any chat model your provider lists

fetcher ask "How many days until next year begins?"
```

```json
{
  "answer": "...",
  "sources": ["https://docs.python.org/3/library/datetime.html#4"],
  "confident": true
}
```

How the answer is kept honest (`rag.py`):

- the prompt tells the model to use only the retrieved passages and to say so when they don't
  contain the answer (`"confident": false`);
- the reply must be JSON that fits the `Answer` Pydantic model, and every source it cites must be one
  of the passages it was actually given, so it can't invent references;
- if the reply fails either check, the error is sent back and the model gets one more try.

## What's inside

```
src/fetcher/
  models.py     Page and Status: what one fetched page looks like (Pydantic)
  config.py     FetchConfig: every knob in one frozen dataclass
  fetch.py      Fetcher: asyncio + httpx, semaphore, retries, rate limit, robots.txt
  retry.py      which failures to retry, exponential backoff with full jitter, Retry-After
  ratelimit.py  per-host rate limiter
  robots.py     robots.txt cache (RFC 9309 rules for 4xx and 5xx)
  clean.py      HTML -> readable text
  log.py        JSON logs that print above the progress bar
  output.py     JSONL writer and reader
  cli.py        Typer CLI: `fetcher fetch`, `stats`, `index`, `search`, `ask`
  search.py     embeddings (fastembed, local) and cosine-similarity search, one hit per page
  rag.py        LLM answers as validated JSON, grounded in the retrieved passages, with a retry
scripts/        demos for each chapter, the flaky test server, the benchmark, RAG chunking,
                eval_search.py (retrieval quality: hit@k and MRR)
tests/          pytest with httpx.MockTransport and fake embedders/LLMs (no network, no API key)
```

## Options

```
fetcher fetch URLS_FILE [-o pages.jsonl] [--concurrency 10] [--rps 10] [--timeout 10]
                        [--retries 3] [--workers 4] [--no-robots] [--user-agent ...]
                        [--log-level info]
```

`--concurrency` caps requests in flight across all hosts; `--rps` caps requests per second to any
single host (0 turns it off); `--workers` is how many processes clean HTML (0 = one thread).
