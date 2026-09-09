# workloads

Request workloads consumed by `python -m serving --dataset <...>` and by
`python -m bench run --dataset <...>`. Static `.jsonl` files live at the
top level; the `generators/` subpackage produces fresh ones on demand
and the `examples/` folder ships ready-to-edit invocation templates.

## Layout

```
workloads/
├── *.jsonl                    workload files (flat or agentic; see Format)
├── generators/                JSONL generators
│   ├── __main__.py            python -m workloads.generators <name> ...
│   ├── sharegpt.py            multi-turn ShareGPT parser (tokenizer + optional vLLM)
│   └── casr.py                synthetic prefix-reuse workload generator
└── examples/                  ready-to-edit per-model invocation templates
    ├── gen-llama-3.1-8b.sh
    ├── gen-qwen3-30b-a3b.sh
    └── gen-qwen3-32b.sh
```

## Format

Datasets are stored as `.jsonl` files (one JSON object per line). Two formats are supported:

### Flat requests (e.g., ShareGPT)

Each line is an independent request:

| Field | Type | Description |
| --- | --- | --- |
| `input_toks` | Integer | Number of input (prompt) tokens |
| `output_toks` | Integer | Number of output (generated) tokens |
| `arrival_time_ns` | Integer | Request arrival time in nanoseconds |
| `input_tok_ids` | List[Integer] | (optional) Token IDs of the input sequence for prefix cache matching |
| `output_tok_ids` | List[Integer] | (optional) Token IDs of the output sequence |

```json
{"input_toks": 128, "output_toks": 512, "arrival_time_ns": 0, "input_tok_ids": [1, 2, 3]}
```

### Agentic sessions (e.g., SWE-bench)

Each line is a session with chained LLM calls. The simulator respects dependency chains:
each sub-request is submitted only after the previous one completes plus the tool duration.

| Field | Type | Description |
| --- | --- | --- |
| `session_id` | String | Unique session identifier |
| `arrival_time_ns` | Integer | Session start time in nanoseconds |
| `sub_requests` | List[Object] | Ordered chain of LLM calls |

Each sub-request has:

| Field | Type | Description |
| --- | --- | --- |
| `input_toks` | Integer | Number of input tokens for this LLM call |
| `output_toks` | Integer | Number of output tokens for this LLM call |
| `tool_duration_ns` | Integer | Time to wait after this call completes before the next can start (0 for last) |
| `input_tok_ids` | List[Integer] | (optional) Token IDs for prefix cache matching |
| `output_tok_ids` | List[Integer] | (optional) Token IDs of the output |

```json
{
  "session_id": "task-0-run0",
  "arrival_time_ns": 4059740,
  "sub_requests": [
    {"input_toks": 1472, "output_toks": 133, "tool_duration_ns": 127348767},
    {"input_toks": 1582, "output_toks": 125, "tool_duration_ns": 0}
  ]
}
```

Both formats can coexist in the same file. Format is auto-detected by the presence
of the `sub_requests` key.

## Provided datasets

### ShareGPT traces

Generated on demand by `python -m workloads.generators sharegpt --model <hf-id>
--num-reqs <n> --sps <r>` (see `generators/`). Output files land directly in
this directory and follow the flat-request format above with `input_tok_ids`
populated for prefix-cache hashing.

### CASR synthetic traces

Generated on demand by `python -m workloads.generators casr`. These traces do
not need a tokenizer: they emit deterministic synthetic `input_tok_ids` and
`output_tok_ids`, plus ground-truth experiment labels (`hotspot_id`, `region`,
`link_state`, `decode_tier`) for offline CASR analysis. The generator supports
stable / Zipf / periodic / drifting hotspots, controllable reuse rate, burst
arrivals, edge/cloud region mixing, link degradation, and decode-tier
heterogeneity. A `.meta.json` sidecar records the parameter set and every
request's ground-truth label.


### SWE-bench agentic traces
Agentic sessions derived from real SWE-bench coding tasks with LLM calls chained
by tool calls (bash, grep, file edits). Each session is a complete coding task
consisting of multiple LLM sub-requests (6--20 per session) interleaved with tool
executions.

| File | Sessions | Sub-reqs | Avg sub-reqs/sess | Rate (sess/s) | Model |
| --- | --- | --- | --- | --- | --- |
| `swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl` | 50 | 765 | 15.3 | 0.2 | Qwen3-30B-A3B |

**There is no SWE-bench generator.** `workloads.generators` currently ships
`sharegpt` and `casr`; the SWE-bench file was produced out of band and
committed. To build your own agentic workload, emit the JSONL directly — the
schema is above, and the format reference is
[Workloads → JSONL format](https://llmservingsim.ai/docs/workloads/jsonl-format).

### Other
| File | Description |
| --- | --- |
| `example_trace.jsonl` | Small example trace for quick testing |
| `casr_hetero_hot_cold.jsonl` | Zipf hot/cold CASR trace used by `tests/run_casr_hetero_comparison.sh` |
| `casr_elasticity_low_high_low.jsonl` | Deterministic 1s low / 2s high / 1s low trace for fixed-vs-elastic experiments |

## Generating workloads

Workloads are produced via the `generators/` subpackage. The `sharegpt`
generator uses the target model's tokenizer to populate `input_tok_ids`
(so prefix-cache hashes are stable) and Poisson-distributed arrivals at
the requested rate. The default source dataset is
`shibing624/sharegpt_gpt4` (HF hub); pass `--source` to override with
another HF id or a local file. The `casr` generator creates synthetic
token IDs directly, without a tokenizer.

The simplest path is to copy one of the templates under `examples/`,
edit the model / sps / num-reqs as needed, and run it from inside the
vLLM Docker (`scripts/docker-vllm.sh`):

```bash
./workloads/examples/gen-qwen3-32b.sh
# or override the model on the command line:
MODEL="my-org/my-model" ./workloads/examples/gen-qwen3-32b.sh
```

For ad-hoc invocations:

```bash
python -m workloads.generators sharegpt \
    --model Qwen/Qwen3-32B \
    --num-reqs 300 --sps 10 --seed 42 \
    --output workloads/sharegpt-qwen3-32b-300-sps10.jsonl \
    --use-vllm --vllm-tp 2 --vllm-dtype bfloat16
```

A reproducible CASR workload with Zipf-distributed reusable prefixes and
labelled edge/cloud arrivals:

```bash
python -m workloads.generators casr \
    --output workloads/casr_zipf_demo.jsonl \
    --num-reqs 1000 --sps 10 --seed 42 \
    --hotspot-mode zipf --num-prefixes 8 --reuse-rate 0.8 \
    --edge-fraction 0.5 --link-degrade-at-sec 30 \
    --decode-tier-mode skewed --slow-fraction 0.3
```

For resource elasticity experiments, generate a controlled three-phase trace:

```bash
python -m workloads.generators casr \
    --output workloads/casr_elasticity_low_high_low.jsonl \
    --num-reqs 111 --seed 42 --hotspot-mode stable --reuse-rate 1.0 \
    --prefix-len 64 --input-len 64 --output-len 64 \
    --phase-rates 5,50,5 --phase-durations-sec 1,2,1
```

`--phase-rates` changes the deterministic inter-arrival interval at each
boundary. The phase workload used by
`tests/run_casr_elasticity_comparison.sh` has 11 low-load requests and 100
high-load requests. The script compares fixed 2P, fixed 1P, and resource-aware
dynamic CASR, reporting per-phase latency plus GPU-seconds, peak/average GPU
usage, startup cost, and resource release/rejection counts.

`--use-vllm` drives a real vLLM `LLM` engine in offline batched mode to
fill `output_tok_ids` with the model's natural responses (free
generation). Without it, `output_tok_ids` come straight from the
ShareGPT assistant turn.

Eight flags gate that mode:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--use-vllm` | off | enable free generation |
| `--vllm-tp` | `1` | `tensor_parallel_size` for the offline engine |
| `--vllm-dtype` | `bfloat16` | weight dtype |
| `--vllm-max-num-seqs` | `1024` | `max_num_seqs`, high on purpose — this is a throughput job |
| `--vllm-max-num-batched-tokens` | `16384` | `max_num_batched_tokens`, likewise |
| `--vllm-max-model-len` | model's max | override `max_model_len` |
| `--vllm-temperature` | `0.0` | `0` = greedy, so generation is reproducible for a seed |
| `--vllm-repetition-penalty` | `1.1` | `1.0` disables. Above 1.0 keeps free generation from rambling past natural EOS, so output lengths land at typical ShareGPT values (~500-1000 tokens) instead of every request hitting `--max-output-toks` |

These are settings for the generation job, not for the simulated run:
the generator records only token counts and ids, so a high
`--vllm-max-num-seqs` here says nothing about what `--max-num-seqs` the
workload should be simulated at.

Full flag reference:
[Workloads → ShareGPT generators](https://llmservingsim.ai/docs/workloads/sharegpt-generators).

To create a workload manually, write JSON objects to a `.jsonl` file
following the format above and pass the file path via `--dataset` to
`python -m serving` or `python -m bench run`.
