#!/usr/bin/env python3
"""
Quick-and-dirty load-tester for any OpenAI-style LLM endpoint.

Modified in this fork from the upstream project.
Example:

# vLLM running on port 8000
python -m main \
       --base-url http://localhost:8000/v1 \
       --model Qwen/Qwen3-14B \
       --requests 200 --concurrency 12

# Ollama (must be 0.1.34+ which exposes /v1/chat/completions)
python -m main \
       --base-url http://localhost:11434/v1 \
       --model qwen3:14b-fp16 \
       --requests 200 --concurrency 16
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table


def _existing_file(path: str) -> str:
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"prompt file does not exist: {path}")
    return path


def _build_chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/chat/completions"


async def _chat_completion(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    payload: dict,
    capture_responses: bool,
) -> Tuple[Optional[float], Optional[int], Optional[Dict[str, Any]]]:
    """Send one completion request and return (latency, total_tokens, response)."""
    t0 = time.perf_counter()
    try:
        r = await client.post(url, headers=headers, json=payload, timeout=60)
        latency = time.perf_counter() - t0
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage", {})
        tokens = usage.get(
            "total_tokens",
            usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
        )
        return latency, tokens, data if capture_responses else None
    except Exception:
        return None, None, None


async def _run_once(args: argparse.Namespace) -> None:
    console = Console()
    url = _build_chat_completions_url(args.base_url)
    prompt_text = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt_text = f.read()

    headers = {
        "Content-Type": "application/json",
        **({"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}),
    }
    payload = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": False,
    }

    sem = asyncio.Semaphore(args.concurrency)
    latencies: List[float] = []
    tokens: List[int] = []
    responses: List[Dict[str, Any]] = []

    async def worker() -> None:
        async with sem:
            l, t, resp = await _chat_completion(
                client, url, headers, payload, args.capture_responses
            )
            if l is not None:
                latencies.append(l)
                tokens.append(t if t is not None else 0)
                if resp and args.capture_responses:
                    responses.append(resp)

    async with httpx.AsyncClient(http2=True, timeout=None) as client:
        console.print(
            Panel(
                f"[bold cyan]Warm-up request[/bold cyan]\n"
                f"[white]Endpoint:[/white] {url}\n"
                f"[white]Model:[/white] {args.model}",
                title="Preparing benchmark",
                expand=False,
                border_style="blue",
            )
        )

        warmup_started = time.perf_counter()
        try:
            warmup_response = await client.post(
                url, headers=headers, json=payload, timeout=60
            )
            warmup_response.raise_for_status()
            warmup_elapsed = time.perf_counter() - warmup_started
            console.print(
                Panel(
                    f"[bold green]Warm-up succeeded[/bold green]\n"
                    f"[white]Latency:[/white] {warmup_elapsed:.3f}s",
                    title="Ready",
                    expand=False,
                    border_style="green",
                )
            )
        except Exception as exc:
            warmup_elapsed = time.perf_counter() - warmup_started
            console.print(
                Panel(
                    f"[bold red]Warm-up failed[/bold red]\n"
                    f"[white]After:[/white] {warmup_elapsed:.3f}s\n"
                    f"[white]Error:[/white] {exc}",
                    title="Continuing anyway",
                    expand=False,
                    border_style="red",
                )
            )

        tic = time.perf_counter()
        tasks = [asyncio.create_task(worker()) for _ in range(args.requests)]
        if args.quiet:
            await asyncio.gather(*tasks)
        else:
            progress = Progress(
                SpinnerColumn(style="cyan"),
                TextColumn("[bold blue]Benchmarking[/bold blue]"),
                BarColumn(bar_width=None),
                TextColumn("[bold]{task.completed}/{task.total}[/bold]"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            )
            with progress:
                progress_task = progress.add_task("requests", total=len(tasks))
                for completed in asyncio.as_completed(tasks):
                    await completed
                    progress.advance(progress_task)
        toc = time.perf_counter()

    _report(latencies, tokens, args.requests, toc - tic)

    if args.capture_responses and responses:
        _write_responses_to_file(responses, args.output_file)


def _write_responses_to_file(responses: List[Dict[str, Any]], filename: str) -> None:
    """Write LLM responses to a file."""
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, "w") as f:
        json.dump(responses, f, indent=2)
    print(f"\nResponses written to {filename}")


def _report(
    latencies: List[float], tokens: List[int], total_req: int, wall: float
) -> None:
    console = Console()
    ok = len(latencies)
    safe_wall = wall if wall > 0 else 1e-9
    success_rate = (ok / total_req * 100) if total_req else 0.0

    header = Panel(
        f"[bold green]OpenAI LLM Benchmark[/bold green]\n"
        f"[white]Requests:[/white] {ok}/{total_req} succeeded  "
        f"[white]Success rate:[/white] {success_rate:.1f}%  "
        f"[white]Elapsed:[/white] {wall:.2f}s",
        expand=False,
        border_style="cyan",
    )
    console.print(header)

    if ok == 0:
        console.print("[bold red] No successful requests.[/bold red]\n")
        return

    rps = ok / safe_wall
    rows: List[Tuple[str, str]] = [("Requests/s", f"{rps:,.2f}")]

    if any(tokens):
        tps = sum(tokens) / safe_wall
        rows.append(("Tokens/s", f"{tps:,.2f}"))

    rows.extend(
        [
            ("Avg latency", f"{statistics.mean(latencies):.3f}s"),
            ("p50 latency", f"{np.percentile(latencies, 50):.3f}s"),
            ("p95 latency", f"{np.percentile(latencies, 95):.3f}s"),
        ]
    )

    table = Table(show_header=True, header_style="bold magenta", expand=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="green")
    for metric, value in rows:
        table.add_row(metric, value)

    console.print(table)
    console.print()


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Concurrent benchmark for OpenAI-style LLMs"
    )
    p.add_argument(
        "--base-url",
        required=True,
        help="Base URL or full chat completions URL (e.g. http://host:port/v1)",
    )
    p.add_argument(
        "--api-key", default="", help="Bearer token if your server needs one"
    )
    p.add_argument("--model", required=True, help="Model name")
    p.add_argument(
        "--prompt",
        default="Tell me a fun fact about the Roman Empire",
        help="User prompt",
    )
    p.add_argument(
        "--prompt-file",
        type=_existing_file,
        default=None,
        help="Load prompt text from file (overrides --prompt)",
    )
    p.add_argument("--requests", type=int, default=100, help="Total number of requests")
    p.add_argument("--concurrency", type=int, default=10, help="Parallel workers")
    p.add_argument("--max-tokens", type=int, default=32, help="max_tokens per request")
    p.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature for sampling (0.0 = deterministic)",
    )
    p.add_argument("--quiet", action="store_true", help="Hide progress bar")
    p.add_argument(
        "--capture-responses",
        action="store_true",
        help="Capture LLM responses and write to file",
    )
    p.add_argument(
        "--output-file",
        default="responses.json",
        help="File to write captured responses (used with --capture-responses)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse()
    asyncio.run(_run_once(args))


if __name__ == "__main__":
    main()
