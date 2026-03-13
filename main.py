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
import pathlib
import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
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


StreamEvent = Tuple[float, Optional[int], Dict[str, Any]]


def _extract_usage_counts(
    data: Dict[str, Any],
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None, None, None

    total_tokens = usage.get("total_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")

    total_value = total_tokens if isinstance(total_tokens, int) else None
    prompt_value = prompt_tokens if isinstance(prompt_tokens, int) else None
    completion_value = completion_tokens if isinstance(completion_tokens, int) else None

    if total_value is None and (
        prompt_value is not None or completion_value is not None
    ):
        total_value = int(prompt_value or 0) + int(completion_value or 0)

    return total_value, prompt_value, completion_value


def _extract_total_tokens(data: Dict[str, Any]) -> Optional[int]:
    total_tokens, _, _ = _extract_usage_counts(data)
    return total_tokens


def _extract_finish_reasons(data: Dict[str, Any]) -> List[str]:
    choices = data.get("choices")
    if not isinstance(choices, list):
        return []

    finish_reasons: List[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str):
            finish_reasons.append(finish_reason)

    return finish_reasons


async def _consume_sse_event(
    data_lines: List[str],
    started_at: float,
    events: List[StreamEvent],
    on_event: Optional[Callable[[float, Optional[int], Dict[str, Any]], None]],
) -> bool:
    if not data_lines:
        return False

    payload = "\n".join(data_lines).strip()
    if not payload:
        return False
    if payload == "[DONE]":
        return True

    data = json.loads(payload)
    latency = time.perf_counter() - started_at
    tokens = _extract_total_tokens(data)
    events.append((latency, tokens, data))
    if on_event:
        on_event(latency, tokens, data)
    return False


async def _chat_completion(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    payload: dict,
    on_event: Optional[Callable[[float, Optional[int], Dict[str, Any]], None]] = None,
) -> Optional[List[StreamEvent]]:
    """Send one streaming completion request and return per-event payloads."""
    started_at = time.perf_counter()
    events: List[StreamEvent] = []
    try:
        async with client.stream(
            "POST", url, headers=headers, json=payload, timeout=60
        ) as response:
            response.raise_for_status()

            data_lines: List[str] = []
            async for line in response.aiter_lines():
                if line == "":
                    should_stop = await _consume_sse_event(
                        data_lines, started_at, events, on_event
                    )
                    data_lines.clear()
                    if should_stop:
                        break
                    continue

                if line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())

            if data_lines:
                await _consume_sse_event(data_lines, started_at, events, on_event)

        return events
    except Exception:
        return None


async def _warmup_request(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    model: str,
    temperature: float,
) -> float:
    warmup_payload = {
        "model": model,
        "max_tokens": 32,
        "temperature": temperature,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }

    started_at = time.perf_counter()
    response = await client.post(url, headers=headers, json=warmup_payload, timeout=60)
    response.raise_for_status()
    response.json()
    return time.perf_counter() - started_at


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
        "stream": True,
    }
    if args.stream_include_usage:
        payload["stream_options"] = {"include_usage": True}

    sem = asyncio.Semaphore(args.concurrency)
    latencies: List[float] = []
    ttfts: List[float] = []
    total_tokens: List[int] = []
    completion_tokens: List[int] = []
    streams: List[List[StreamEvent]] = []
    truncated_requests = 0
    active_streams = 0
    total_chunks = 0
    live_total_tokens = 0
    last_reported_tokens = 0
    last_reported_tps = "-"
    latest_chunk_latency: Optional[float] = None
    stats_lock = asyncio.Lock()

    def _progress_fields(started_at: float) -> Dict[str, Any]:
        nonlocal last_reported_tokens, last_reported_tps
        if live_total_tokens > last_reported_tokens:
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            last_reported_tokens = live_total_tokens
            last_reported_tps = f"{live_total_tokens / elapsed:,.2f}"
        return {
            "active": active_streams,
            "chunks": total_chunks,
            "latest": f"{latest_chunk_latency:.3f}s" if latest_chunk_latency else "-",
            "tokens": f"{live_total_tokens:,}" if live_total_tokens else "-",
            "tps": last_reported_tps,
        }

    def _event_recorder(
        progress: Optional[Progress],
        progress_task: Optional[TaskID],
        started_at: float,
    ) -> Callable[[float, Optional[int], Dict[str, Any]], None]:
        seen_tokens = 0

        def record(
            latency: float, _tokens: Optional[int], _data: Dict[str, Any]
        ) -> None:
            nonlocal total_chunks, latest_chunk_latency, live_total_tokens, seen_tokens
            total_chunks += 1
            latest_chunk_latency = latency
            if _tokens is not None and _tokens > seen_tokens:
                live_total_tokens += _tokens - seen_tokens
                seen_tokens = _tokens
            if progress is not None and progress_task is not None:
                progress.update(progress_task, **_progress_fields(started_at))

        return record

    async def worker(
        progress: Optional[Progress] = None,
        progress_task: Optional[TaskID] = None,
    ) -> None:
        nonlocal active_streams, truncated_requests
        async with sem:
            async with stats_lock:
                active_streams += 1
                if progress is not None and progress_task is not None:
                    progress.update(progress_task, **_progress_fields(tic))

            stream = await _chat_completion(
                client,
                url,
                headers,
                payload,
                on_event=_event_recorder(progress, progress_task, tic),
            )

            async with stats_lock:
                active_streams -= 1

            if stream:
                streams.append(stream)
                ttfts.append(stream[0][0])
                latencies.append(stream[-1][0])

                if any(
                    finish_reason == "length"
                    for _, _, chunk_data in stream
                    for finish_reason in _extract_finish_reasons(chunk_data)
                ):
                    truncated_requests += 1

                final_total_tokens = 0
                final_completion_tokens = 0
                for _, _, chunk_data in reversed(stream):
                    chunk_total_tokens, _, chunk_completion_tokens = (
                        _extract_usage_counts(chunk_data)
                    )
                    if chunk_total_tokens is not None:
                        final_total_tokens = chunk_total_tokens
                        if chunk_completion_tokens is not None:
                            final_completion_tokens = chunk_completion_tokens
                        break

                total_tokens.append(final_total_tokens)
                completion_tokens.append(final_completion_tokens)

            if progress is not None and progress_task is not None:
                progress.advance(progress_task)
                progress.update(progress_task, **_progress_fields(tic))

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
            warmup_elapsed = await _warmup_request(
                client,
                url,
                headers,
                args.model,
                args.temperature,
            )
            console.print(
                Panel(
                    f"[bold green]Warm-up succeeded[/bold green]\n"
                    f"[white]Prompt:[/white] hello\n"
                    f"[white]Max tokens:[/white] 32\n"
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
        if args.quiet:
            tasks = [asyncio.create_task(worker()) for _ in range(args.requests)]
            await asyncio.gather(*tasks)
        else:
            progress = Progress(
                SpinnerColumn(style="cyan"),
                TextColumn("[bold blue]Benchmarking[/bold blue]"),
                BarColumn(bar_width=None),
                TextColumn("[bold]{task.completed}/{task.total}[/bold]"),
                TextColumn("active: [bold]{task.fields[active]}[/bold]"),
                TextColumn("chunks: [bold]{task.fields[chunks]}[/bold]"),
                TextColumn("tokens: [bold]{task.fields[tokens]}[/bold]"),
                TextColumn("tok/s: [bold]{task.fields[tps]}[/bold]"),
                TextColumn("latest: [bold]{task.fields[latest]}[/bold]"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            )
            with progress:
                progress_task = progress.add_task(
                    "requests",
                    total=args.requests,
                    active=0,
                    chunks=0,
                    tokens="-",
                    tps="-",
                    latest="-",
                )
                tasks = [
                    asyncio.create_task(worker(progress, progress_task))
                    for _ in range(args.requests)
                ]
                for completed in asyncio.as_completed(tasks):
                    await completed
        toc = time.perf_counter()

    _report(
        latencies,
        ttfts,
        total_tokens,
        completion_tokens,
        total_chunks,
        truncated_requests,
        args.requests,
        toc - tic,
    )

    if args.output:
        _write_responses_to_file(streams, args.output)


def _write_responses_to_file(
    responses: List[List[StreamEvent]], filename: pathlib.Path
) -> None:
    """Write raw streaming events to a file."""
    if filename.parent != pathlib.Path("."):
        filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=2)
    print(f"Responses written to {filename}")


def _report(
    latencies: List[float],
    ttfts: List[float],
    total_tokens: List[int],
    completion_tokens: List[int],
    total_chunks: int,
    truncated_requests: int,
    total_req: int,
    wall: float,
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

    rows.append(("Stream chunks", f"{total_chunks:,}"))

    if any(completion_tokens):
        rows.append(("Generated tokens", f"{sum(completion_tokens):,}"))
        tps = sum(completion_tokens) / safe_wall
        rows.append(("Tokens/s", f"{tps:,.2f}"))
    elif any(total_tokens):
        rows.append(("Total tokens", f"{sum(total_tokens):,}"))
        tps = sum(total_tokens) / safe_wall
        rows.append(("Tokens/s", f"{tps:,.2f}"))

    if ttfts:
        rows.extend(
            [
                ("Avg TTFT", f"{statistics.mean(ttfts):.3f}s"),
                ("p50 TTFT", f"{np.percentile(ttfts, 50):.3f}s"),
                ("p95 TTFT", f"{np.percentile(ttfts, 95):.3f}s"),
            ]
        )

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

    if truncated_requests:
        console.print(
            Panel(
                f"[bold yellow]{truncated_requests} request(s) ended with finish_reason='length'[/bold yellow]\n"
                "[white]This usually means the model hit the output token limit before finishing.[/white]\n"
                "[white]Consider increasing --max-tokens or shortening the prompt.[/white]",
                title="Warning",
                expand=False,
                border_style="yellow",
            )
        )


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
    p.add_argument("--concurrency", type=int, default=1, help="Parallel workers")
    p.add_argument(
        "--max-tokens", type=int, default=1024, help="max_tokens per request"
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature for sampling (0.0 = deterministic)",
    )
    p.add_argument("--quiet", action="store_true", help="Hide progress bar")
    p.add_argument(
        "--stream-include-usage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request usage in streaming responses when supported",
    )
    p.add_argument(
        "--output",
        default=None,
        type=pathlib.Path,
        help="File to write raw streaming events",
    )
    return p.parse_args()


def main() -> None:
    args = _parse()
    asyncio.run(_run_once(args))


if __name__ == "__main__":
    main()
