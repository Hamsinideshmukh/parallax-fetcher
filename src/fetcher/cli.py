"""The command line: `fetch`, `stats`, and the RAG layer: `index`, `search`, `ask`."""

import asyncio
import time
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeElapsedColumn
from rich.table import Table

from fetcher.config import DEFAULT_USER_AGENT, FetchConfig
from fetcher.fetch import Fetcher
from fetcher.log import log_info, setup_logging
from fetcher.models import Status
from fetcher.output import JsonlWriter, read_pages
from fetcher.rag import ask as ask_llm
from fetcher.rag import openai_compatible_llm
from fetcher.search import DEFAULT_MODEL, FastEmbedder, SearchIndex, read_chunks

app = typer.Typer(add_completion=False, help="Fetch many pages at once, politely, as clean JSONL.")
console = Console(stderr=True)


def read_urls(path: Path) -> list[str]:
    """One URL per line. Blank lines and # comments are skipped, duplicates removed."""
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    urls = [line for line in lines if line.startswith(("http://", "https://"))]
    return list(dict.fromkeys(urls))  # keeps the order, drops repeats


async def run(urls: list[str], out: Path, config: FetchConfig) -> Counter[Status]:
    counts: Counter[Status] = Counter()
    columns = ("{task.description}", BarColumn(), MofNCompleteColumn(), TimeElapsedColumn())
    with Progress(*columns, console=console) as progress, JsonlWriter(out) as writer:
        task = progress.add_task("fetching", total=len(urls))
        async with Fetcher(config) as fetcher:
            async for page in fetcher.fetch_many(urls):
                writer.write(page)
                counts[page.status] += 1
                progress.advance(task)
    return counts


@app.command()
def fetch(
    urls_file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="One URL per line")
    ],
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("pages.jsonl"),
    concurrency: Annotated[int, typer.Option(min=1, envvar="FETCHER_CONCURRENCY")] = 10,
    rps: Annotated[float, typer.Option(min=0, help="Per host, 0 = no limit")] = 10.0,
    timeout: Annotated[float, typer.Option(min=0.1, help="Seconds per request")] = 10.0,
    retries: Annotated[int, typer.Option(min=0, help="Retries after the first try")] = 3,
    workers: Annotated[int, typer.Option(min=0, help="Processes for HTML cleaning")] = 4,
    robots: Annotated[bool, typer.Option(help="Respect robots.txt")] = True,
    user_agent: Annotated[str, typer.Option(envvar="FETCHER_USER_AGENT")] = DEFAULT_USER_AGENT,
    log_level: Annotated[str, typer.Option(envvar="FETCHER_LOG_LEVEL")] = "info",
) -> None:
    """Fetch every URL in URLS_FILE and write one JSON line per page to OUT."""
    setup_logging(console, log_level)
    urls = read_urls(urls_file)
    config = FetchConfig(
        concurrency=concurrency,
        per_host_rps=rps,
        timeout_s=timeout,
        max_attempts=retries + 1,
        clean_workers=workers,
        respect_robots=robots,
        user_agent=user_agent,
    )
    log_info("start", urls=len(urls), concurrency=concurrency, rps=rps)
    started = time.perf_counter()
    counts = asyncio.run(run(urls, out, config))
    seconds = time.perf_counter() - started
    summary = {"seconds": round(seconds, 1), "pages_per_s": round(len(urls) / seconds, 1)}
    log_info("done", **summary, **{status.value: n for status, n in counts.items()})
    typer.echo(f"wrote {sum(counts.values())} pages to {out}")


@app.command()
def stats(pages_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """Summarise a JSONL file: statuses, attempts, words, and the slowest pages."""
    pages = read_pages(pages_file)
    table = Table(title=f"{pages_file} ({len(pages)} pages)")
    for column in ("status", "pages", "avg attempts", "words"):
        table.add_column(column, justify="right" if column != "status" else "left")
    for status, n in Counter(p.status for p in pages).most_common():
        group = [p for p in pages if p.status == status]
        attempts = sum(p.attempts for p in group) / n
        table.add_row(
            status.value, str(n), f"{attempts:.2f}", f"{sum(p.word_count for p in group):,}"
        )
    Console().print(table)
    for p in sorted(pages, key=lambda p: p.elapsed_ms, reverse=True)[:3]:
        typer.echo(f"slowest: {p.elapsed_ms:>6} ms  {p.url}")


IndexDir = Annotated[Path, typer.Option("--index", help="Folder that holds the search index")]


@app.command()
def index(
    chunks_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    out: IndexDir = Path("data/index"),
    model: Annotated[str, typer.Option(help="fastembed model name")] = DEFAULT_MODEL,
) -> None:
    """Embed every chunk once and save the vectors, so searches are instant."""
    chunks = read_chunks(chunks_file)
    started = time.perf_counter()
    SearchIndex.build(chunks, FastEmbedder(model)).save(out)
    seconds = time.perf_counter() - started
    typer.echo(f"embedded {len(chunks):,} chunks in {seconds:.1f} s -> {out}")


@app.command()
def search(
    query: str,
    index_dir: IndexDir = Path("data/index"),
    k: Annotated[int, typer.Option(min=1)] = 5,
) -> None:
    """Find the chunks closest in meaning to QUERY (not just matching words)."""
    hits = SearchIndex.load(index_dir).search(FastEmbedder().embed_query(query), k=k)
    for rank, hit in enumerate(hits, 1):
        preview = hit.chunk.text[:120].replace("\n", " ")
        typer.echo(f"{rank}. {hit.score:.3f}  {hit.chunk.url}\n   {preview}...")


@app.command()
def ask(
    question: str,
    index_dir: IndexDir = Path("data/index"),
    k: Annotated[int, typer.Option(min=1, help="Passages to give the LLM")] = 4,
) -> None:
    """Answer QUESTION from the fetched pages with an LLM, citing its sources."""
    hits = SearchIndex.load(index_dir).search(FastEmbedder().embed_query(question), k=k)
    answer = ask_llm(question, hits, openai_compatible_llm())
    typer.echo(answer.model_dump_json(indent=2))
