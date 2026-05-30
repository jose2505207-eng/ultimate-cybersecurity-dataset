# Qwen 32B Cloud Validation Runbook

This is the first cloud validation phase for `Qwen/Qwen2.5-Coder-32B-Instruct`. It is intentionally small: it validates infrastructure, checkpoint behavior, logging, memory stability, and early benchmark movement before any expensive long run.

## Provider Recommendation

Checked on 2026-05-28. Prices can change during deployment, especially on marketplace providers.

1. RunPod Secure Cloud, 1x H100 80GB if budget allows, otherwise 1x A100 80GB.
   - Best balance for this first run: predictable, per-second billing, simple persistent volumes.
   - Current public pod pricing shows A100 PCIe 80GB around 1.39 USD/hr, A100 SXM 80GB around 1.49 USD/hr, H100 PCIe 80GB around 2.89 USD/hr, and H100 SXM 80GB around 3.29 USD/hr.
2. Prime Intellect, 1x H100 80GB, only if availability and reliability look good at launch time.
   - Attractive budget option because it aggregates H100 availability and advertises H100 listings from about 1.00 USD/hr, with shown examples around 1.49 USD/hr.
3. Vast.ai, 1x H100 80GB or A100 80GB, only for very cost-sensitive validation.
   - Good for cost discovery, but host quality, disk, networking, and interruption behavior vary by listing.
4. Lambda, 1x H100 80GB if you want a more managed experience and availability is acceptable.
   - Current public instance pricing lists 1x H100 PCIe 80GB around 3.29 USD/hr and 1x H100 SXM 80GB around 4.29 USD/hr. Lambda's listed 1x options do not currently show A100 80GB single-GPU instances.

Minimum viable hardware:

- GPU: 1x A100 80GB.
- Preferred GPU: 1x H100 80GB.
- Avoid for first run: 48GB cards. A 32B 4-bit QLoRA run may fit with more compromises, but the first cloud validation should spend money on lowering failure risk.
- Host RAM: 120GB minimum, 180GB+ preferred.
- Disk: 250GB minimum persistent volume, 400GB preferred.
- Image: Ubuntu 22.04 or 24.04 with a CUDA 12.x PyTorch image.

## Training Architecture

Use a single 80GB GPU first. For QLoRA, this is lower risk than starting with FSDP or DeepSpeed because quantized PEFT models can be sensitive to sharding/offload behavior.

- Quantization: bitsandbytes NF4 4-bit, double quantization enabled.
- Adapter: LoRA only, rank 16, alpha 32, dropout 0.05.
- Sequence length: 512.
- Micro batch size: 1.
- Gradient accumulation: 4.
- Max steps: 100.
- Eval cadence: every 25 optimizer steps, 8 eval examples per cadence.
- Checkpoint cadence: every 100 optimizer steps plus final adapter export.
- Checkpoint retention: latest 3 local `checkpoint-N/` directories.

The Accelerate config in `config/accelerate_qwen32b_single_gpu.yml` records the intended first-run strategy. Do not use multi-GPU FSDP or DeepSpeed until this single-GPU validation passes.

## Cloud Setup

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs tmux rsync htop nvtop
git lfs install

git clone <YOUR_REPO_URL> ultimate-cybersecurity-dataset
cd ultimate-cybersecurity-dataset

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# If your cloud image does not already include a working CUDA PyTorch build,
# install the CUDA-specific PyTorch wheel recommended at https://pytorch.org/get-started/locally/.
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("available", torch.cuda.is_available())
PY

pip install -r requirements.txt
pip install -r requirements-cloud-qlora.txt
huggingface-cli login
```

Upload or regenerate the cleaned dataset:

```bash
rsync -av outputs/sft_dataset/qwen_cyber_sft_v2_20k/ <USER>@<HOST>:~/ultimate-cybersecurity-dataset/outputs/sft_dataset/qwen_cyber_sft_v2_20k/
```

If you prefer regenerating from the repository data on the cloud host, use the existing SFT preparation pipeline instead of copying the output directory.

## First Launch

First run may need network access to populate the cloud Hugging Face cache:

```bash
tmux new -s qwen32b
source .venv/bin/activate
export OUTPUT_DIR=outputs/qlora/qwen25_coder_32b_cloud_validation_r16_512_step100
bash scripts/cloud_qwen32b_validation.sh
```

After the model is cached on a persistent volume, rerun with local cache only:

```bash
source .venv/bin/activate
export LOCAL_FILES_ONLY=1
export OUTPUT_DIR=outputs/qlora/qwen25_coder_32b_cloud_validation_r16_512_step100_retry
bash scripts/cloud_qwen32b_validation.sh
```

Equivalent explicit training command:

```bash
python -m scripts.train_qlora \
  --config config/qlora_cloud_qwen25_coder_32b.yml \
  --dataset outputs/sft_dataset/qwen_cyber_sft_v2_20k \
  --output-dir outputs/qlora/qwen25_coder_32b_cloud_validation_r16_512_step100 \
  --train-rows 4096 \
  --eval-rows 512 \
  --max-steps 100 \
  --max-seq-length 512 \
  --checkpoint-every 100 \
  --keep-checkpoints 3 \
  --eval-every 25 \
  --eval-batches 8 \
  --resume \
  --sync-git-artifacts \
  --artifact-report-dir reports/qwen32b_runs/qwen25_coder_32b_cloud_validation_r16_512_step100 \
  --artifact-run-name qwen25_coder_32b_cloud_validation_r16_512_step100 \
  --manual-loop \
  2>&1 | tee outputs/qlora/qwen25_coder_32b_cloud_validation_r16_512_step100/cloud_train.log
```

## Evaluation

Evaluate the new adapter on a bounded benchmark subset:

```bash
python -m scripts.evaluate_qlora_adapters \
  --config config/qlora_cloud_qwen25_coder_32b.yml \
  --dataset auto \
  --out-dir outputs/qlora_eval/qwen25_32b_cloud_validation_limit64 \
  --adapter-path outputs/qlora/qwen25_coder_32b_cloud_validation_r16_512_step100/final_adapter \
  --adapter-name qwen32b_cleaned_sft_step100 \
  --limit 64 \
  --seed 42 \
  --max-new-tokens 32 \
  --max-seq-length 512 \
  --local-files-only
```

If base-plus-adapter evaluation runs out of memory, run base and adapter in separate fresh processes and combine with `--baseline-predictions`, matching the local evaluation workflow.

## Checkpoint And Recovery

Expected artifacts:

- `cloud_train.log`
- `train_metrics.json`
- `prepare_summary.json`
- `checkpoint-100/`
- `final_adapter/`
- tracked checkpoint metadata under `reports/qwen32b_runs/...`

Copy artifacts off the instance after each successful checkpoint:

```bash
rsync -av outputs/qlora/qwen25_coder_32b_cloud_validation_r16_512_step100/ <USER>@<BACKUP_HOST>:/backups/qwen32b-validation/
```

For object storage, use `rclone sync` to S3, R2, Backblaze, or your provider bucket.

The current manual training loop saves valid LoRA adapter checkpoints, optimizer/RNG state, metrics, and tracked checkpoint metadata. On restart with resume enabled, it loads the latest local numeric checkpoint, restores the trainable PEFT adapter, restores optimizer/RNG state from `training_state.pt` when present, and continues from the completed optimizer step. Heavy adapter weights stay in ignored local `outputs/`; small metadata/log/evaluation reports are mirrored to `reports/qwen32b_runs/...` and pushed through git.

## Expected Behavior

Memory:

- A100 80GB: expect high but stable VRAM usage, typically well under 80GB for this 4-bit LoRA configuration.
- H100 80GB: similar memory shape, faster steps.
- 48GB cards: not recommended for the first validation because allocator fragmentation and evaluation can push the run over the edge.

Throughput and runtime estimates:

- H100 80GB: roughly 25-60 train tokens/sec after model load; 100 steps should usually finish in 1-2.5 hours plus first model download.
- A100 80GB: roughly 12-30 train tokens/sec; 100 steps should usually finish in 2-4 hours plus first model download.
- First model download may add 20-60 minutes depending on provider bandwidth and Hugging Face cache location.

Likely bottlenecks:

- First model download and cache placement.
- bitsandbytes/CUDA mismatch.
- Non-persistent output directories.
- Disk too small for model cache plus checkpoints.
- Marketplace interruption or slow host storage.
- Evaluating base and adapter in the same process on a memory-constrained host.

Success criteria:

- Training reaches step 100.
- Loss and eval loss are finite.
- VRAM reserved does not climb step-over-step without settling.
- All cadence checkpoints validate.
- `final_adapter/` validates.
- Evaluation writes predictions and metrics with the existing benchmark schema.
