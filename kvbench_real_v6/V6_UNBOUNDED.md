# v6: natural generation to server context boundary

Changes from v5:

- `max_tokens` is omitted by default for MATH-500 and HumanEval.
- `min_tokens`, `ignore_eos`, and explicit `stop` are also omitted from requests.
- `serve.sh` already launches vLLM with `--generation-config vllm`; therefore no model `generation_config.json` max-new-token ceiling is inherited.
- A `finish_reason=length` event is recorded as a length/context-window termination and does **not** disqualify a pilot concurrency by default.
- Request-level caps remain available only as explicit debug options `--math-max-tokens` / `--human-max-tokens`.
- On this RTX 3090 workflow, the pipeline default pilot concurrency is now `8`, because the prior capped pilot completed C8 for all three methods and both datasets. Override with `PILOT_CONCURRENCY="4 8"` if a conservative re-check is desired.

Do not resume a v5 capped pilot into v6. Use a new result directory.
