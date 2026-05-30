# Checkpoint Strategy

Date: 2026-05-30

This strategy applies to the Qwen2.5-Coder-32B QLoRA cloud path.

## Goals

- Save a recoverable local checkpoint every 100 optimizer steps.
- Keep only the latest 3 local checkpoints to control disk usage.
- Commit and push small milestone checkpoint metadata.
- Commit and push training logs.
- Commit and push evaluation reports.
- Resume after interruption without rebuilding the existing QLoRA pipeline.
- Keep heavy adapter checkpoint weights out of git.

## Local Checkpoints

Local checkpoints are written under the configured QLoRA output directory:

```text
outputs/qlora/qwen25_coder_32b_cloud_validation_r16_512_step100/
```

Each numeric checkpoint directory uses this shape:

```text
checkpoint-100/
  adapter_config.json
  adapter_model.safetensors
  training_state.json
  training_state.pt
```

The adapter files come from PEFT. The `training_state.pt` file stores optimizer state, RNG state, completed optimizer step, step metrics, checkpoint validation history, token count, and last loss. The `training_state.json` file is a small readable summary of the same checkpoint state.

The final adapter is still exported separately:

```text
final_adapter/
```

## Save Cadence

The 32B cloud config now sets:

```yaml
checkpoint_every: 100
checkpoint_keep_latest: 3
save_total_limit: 3
resume_from_checkpoint: true
```

The cloud wrapper also defaults to:

```bash
CHECKPOINT_EVERY=100
KEEP_CHECKPOINTS=3
```

## Retention

After each checkpoint save, the trainer scans numeric `checkpoint-N` directories and keeps only the latest 3 by step number.

Example:

```text
checkpoint-100
checkpoint-200
checkpoint-300
checkpoint-400
```

After pruning:

```text
checkpoint-200
checkpoint-300
checkpoint-400
```

`final_adapter/` is not pruned by this checkpoint retention rule.

## Resume

Resume is enabled for the manual QLoRA loop used by the 32B cloud wrapper.

On startup with resume enabled:

1. The trainer finds the latest numeric checkpoint in the output directory.
2. The base Qwen model is loaded in 4-bit.
3. The PEFT adapter is restored from that checkpoint with trainable LoRA weights.
4. The optimizer and RNG state are loaded from `training_state.pt` when present.
5. Training continues from the saved completed optimizer step.

If an older checkpoint lacks `training_state.pt`, the trainer can still reload the adapter and infer the completed step from the checkpoint directory name, but optimizer/scheduler state will be fresh. New checkpoints created by this strategy include the full state file.

## Git Artifact Sync

Heavy checkpoint weights remain local and ignored by git. Small artifacts are mirrored into:

```text
reports/qwen32b_runs/qwen25_coder_32b_cloud_validation_r16_512_step100/
```

Milestone checkpoint metadata is mirrored as:

```text
checkpoints/checkpoint-000100.json
```

Training logs and summaries are mirrored as:

```text
logs/train_events.jsonl
logs/cloud_train.log
logs/cloud_eval_adapter.log
metrics/prepare_summary.json
metrics/train_metrics.json
artifact_manifest.json
```

Evaluation reports are mirrored as:

```text
evaluation/evaluation_summary.json
evaluation/model_comparison_metrics.csv
```

The trainer commits and pushes checkpoint metadata at each milestone when artifact sync is enabled. The cloud wrapper runs `scripts.sync_run_artifacts` on exit to commit and push logs and evaluation reports.

## Controls

Default cloud behavior enables git artifact sync and push:

```bash
SYNC_GIT_ARTIFACTS=1
GIT_PUSH_ARTIFACTS=1
GIT_REMOTE=origin
GIT_BRANCH=main
```

To commit locally without pushing:

```bash
GIT_PUSH_ARTIFACTS=0 bash scripts/cloud_qwen32b_validation.sh
```

To disable git artifact sync entirely:

```bash
SYNC_GIT_ARTIFACTS=0 bash scripts/cloud_qwen32b_validation.sh
```

To change the tracked report directory:

```bash
REPORT_DIR=reports/qwen32b_runs/my_run bash scripts/cloud_qwen32b_validation.sh
```

## Non-Goals

- Do not commit `adapter_model.safetensors` to git.
- Do not commit full Hugging Face model cache files.
- Do not rebuild the existing trainer, dataset builder, or evaluator.
- Do not rely on git for restoring heavy checkpoint weights.

For disaster recovery beyond the local pod, sync the full ignored checkpoint directory to object storage or another persistent host with `rsync`, `rclone`, S3, R2, Backblaze, or provider storage.

## Verification Before Training

Before launching training, verify:

```bash
python -m py_compile scripts/train_qlora.py scripts/sync_run_artifacts.py
python -m pytest tests/test_train_qlora.py tests/test_prepare_sft_dataset.py tests/test_evaluate_qlora_adapters.py -q
```

No training is started by these checks.
