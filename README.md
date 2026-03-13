# OpenAI LLM Benchmark

> Notice: This fork contains modifications from the upstream `openai-llm-benchmark` project.

A quick-and-dirty load-tester for any OpenAI-style LLM endpoint. This tool allows you to benchmark the performance of various LLM models by sending concurrent requests and measuring metrics like latency and tokens per second.

## Features

- Test any OpenAI-compatible API endpoint
- Configure number of requests and concurrency level
- Measure key performance metrics (requests/sec, tokens/sec, latency)
- Support for various models and deployments (vLLM, Ollama, etc.)
- Progress bar visualization (with Rich)
- Streaming benchmarks with live chunk stats in Rich
- Optional raw stream capture for inspection and analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/robert-mcdermott/openai-llm-benchmark.git
cd openai-llm-benchmark

# Install dependencies
uv sync
```

## Usage

```bash
uv run openai-llm-benchmark \
       --base-url <API_ENDPOINT> \
       --model <MODEL_NAME> \
       --prompt-file <PROMPT_FILE> \
       --requests <NUM_REQUESTS> \
       --concurrency <CONCURRENCY_LEVEL>
```

### Example: Testing vLLM

```bash
uv run openai-llm-benchmark \
       --base-url http://localhost:8000/v1 \
       --model Qwen/Qwen3-14B \
       --requests 200 --concurrency 12
```

### Example: Testing Ollama

```bash
uv run openai-llm-benchmark \
       --base-url http://localhost:11434/v1 \
       --model qwen3:14b-fp16 \
       --requests 200 --concurrency 16
```

### Example: Capturing raw streaming events to a file

```bash
uv run openai-llm-benchmark \
       --base-url http://localhost:11434/v1 \
       --model qwen3:14b-fp16 \
       --requests 50 --concurrency 8 \
       --output results/ollama_streams.json
```

## Parameters

| Parameter       | Description                                                      | Default         |
| --------------- | ---------------------------------------------------------------- | --------------- |
| `--base-url`    | Base URL or full chat completions URL (required)                 | -               |
| `--api-key`     | Bearer token for authentication                                  | ""              |
| `--model`       | Model name to test                                               | "llama3.2"      |
| `--prompt`      | User prompt to send                                              | "Hello, world!" |
| `--prompt-file` | Path to a file containing the prompt text (overrides `--prompt`) | -               |
| `--requests`    | Total number of requests                                         | 100             |
| `--concurrency` | Number of parallel workers                                       | 10              |
| `--max-tokens`  | Maximum tokens per request                                       | 32              |
| `--temperature` | Temperature for sampling (0.0 = deterministic)                   | 0.2             |
| `--quiet`       | Hide progress bar                                                | False           |
| `--stream-include-usage` | Request usage in stream events when supported             | True            |
| `--output`      | File path for captured raw streaming events                      |                 |

## Output

The benchmark will output:

- Number of successful requests
- Total execution time
- Requests per second
- Stream chunks received
- Generated tokens and tokens per second (when usage is emitted, including final-only usage chunks)
- Average TTFT
- p50 TTFT
- p95 TTFT
- Average latency
- p50 latency (median)
- p95 latency

If any streamed request ends with `finish_reason="length"`, the benchmark prints a warning after the summary to indicate the response was truncated by the output token limit.

During the run, Rich also shows live request progress, active streams, total chunks received, streamed token totals, live tokens/sec, and the latest chunk latency.

Before the benchmark starts, the tool sends a small non-streaming warm-up request with the prompt `hello` and `max_tokens=32` to verify the model is loaded.

The benchmark now sends `stream_options.include_usage=true` by default so OpenAI-compatible backends can return usage in streamed responses. If your provider rejects that field, disable it with `--no-stream-include-usage`.

When `--output` is enabled, each request is written as a JSON array of streaming events, where every event is stored as `[latency_seconds, total_tokens_or_null, raw_chunk_json]`.

## Requirements

- Python 3.12+
- httpx[http2]
- numpy
- rich
