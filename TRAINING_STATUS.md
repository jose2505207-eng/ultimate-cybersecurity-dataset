# Training Status

Last updated: 2026-05-30T09:31:35.189141+00:00

## Run

- Status: running
- Run name: `qwen25_coder_32b_production_r16_512_step8000`
- Model: `Qwen/Qwen2.5-Coder-32B-Instruct`
- Output dir: `/workspace/ultimate-cybersecurity-dataset/outputs/qlora/qwen25_coder_32b_production_r16_512_step8000`
- Latest checkpoint: `/workspace/ultimate-cybersecurity-dataset/outputs/qlora/qwen25_coder_32b_production_r16_512_step8000/checkpoint-100`
- Latest checkpoint step: 100
- Checkpoint validation: True
- Local checkpoints retained: 1
- Local checkpoints pruned at this milestone: 0

## Latest Metrics

- Loss: 0.5563253769651055
- Learning rate: 2.0833333333333336e-05
- Tokens/sec: 252.43411364659207
- VRAM allocated MB: 23514.95
- VRAM reserved MB: 56362.0
- Eval loss: 0.3332148776971735
- Eval batches: 16.0

## Artifact Policy

- Heavy adapter checkpoints remain local under ignored `outputs/` paths.
- Checkpoint metadata, logs, evaluation reports, and this status file are intended for git sync.
