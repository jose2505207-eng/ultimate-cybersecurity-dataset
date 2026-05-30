# Training Status

Last updated: 2026-05-30 08:45 UTC

## Run

- Status: running
- Launched: 2026-05-30 08:44 UTC
- tmux session: `qwen32b_prod`
- Training PID at launch: `17010`
- Model: `Qwen/Qwen2.5-Coder-32B-Instruct`
- Training method: QLoRA, 4-bit NF4, PEFT LoRA
- Trainable LoRA parameters: 134,217,728
- GPU target: single NVIDIA A100-SXM4-80GB
- Config: `config/qlora_production_qwen25_coder_32b.yml`
- Dataset: `outputs/sft_dataset/qwen_cyber_sft_v2_production_50k`
- Output dir: `outputs/qlora/qwen25_coder_32b_production_r16_512_step8000`
- Report dir: `reports/qwen32b_runs/qwen25_coder_32b_production_r16_512_step8000`

## Dataset

- Source silver rows: 302,317
- Prepared SFT rows after quality filtering before cap: 218,640
- Production SFT cap: 50,000
- Train rows: 31,852
- Eval rows: 9,113
- Test rows: 9,035
- Prepared train tokens at sequence length 512: 9,445,154
- Prepared eval tokens at sequence length 512: 2,644,312

## Training Plan

- Max optimizer steps: 8,000
- Micro batch size: 1
- Gradient accumulation: 4
- Approximate train examples consumed: 32,000
- Checkpoint cadence: every 100 optimizer steps
- Local checkpoint retention: latest 3 numeric checkpoints
- Resume: enabled from latest local `checkpoint-N/`
- Eval loss cadence: every 100 optimizer steps
- Eval batches per cadence: 16
- Post-run adapter evaluation limit: 128 examples

## Estimates

- Expected runtime on A100 80GB: 160-320 hours
- Estimated completion window: 2026-06-06 to 2026-06-12 UTC
- Estimated adapter checkpoint size: 250-400 MB
- Estimated optimizer/RNG training state per checkpoint: 1.0-1.5 GB
- Estimated full local checkpoint size: 1.3-2.0 GB
- Estimated retained local checkpoints: 4-6 GB for latest 3, plus final adapter
- Estimated base model cache need: 65-90 GB under `/workspace/.cache/huggingface`

## Artifact Policy

- Heavy model checkpoints stay local under ignored `outputs/` paths.
- Milestone checkpoint metadata, status, logs, and evaluation reports are committed and pushed.
- `TRAINING_STATUS.md` is updated at launch, at checkpoint milestones, and during final artifact sync.

## Latest Validation

- CUDA visible to PyTorch: true
- GPU: NVIDIA A100-SXM4-80GB
- VRAM: 79.25 GiB reported by PyTorch
- Workspace free space: about 87T
- Dependency import validation: passed in `.venv`
- Focused tests: 11 passed
- Prepare-only dataset/tokenizer validation: passed
