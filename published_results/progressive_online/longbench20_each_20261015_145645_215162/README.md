# LongBench Progressive-KV Online Experiment

This directory contains the publishable summaries and logs from the
2026-10-15 online experiment.

- Six LongBench datasets, 20 cases per dataset
- 120 requests per arm and 360 requests total
- Arms: `parent`, `p211_full`, and `p211_adaptive`
- Model: Qwen3-4B
- GPU: RTX 3090
- Concurrency: 8

The complete local run directory also contains raw request/response artifacts
and generated runtime trees. Those large implementation artifacts are not
included here. The files under `summary/` and `comparison.*` are the retained
measurement and scoring summaries; `logs/` contains the launcher, run,
scoring, and report logs for all three arms.

Main files:

- `comparison.csv` and `comparison.json`: per-arm, per-dataset performance and quality
- `paired_quality.csv`: case-level quality pairing against `parent`
- `readout.md`: experiment readout and protocol limitations
- `plan.json`, `status.json`, and `summary/data_manifest.json`: experiment provenance
