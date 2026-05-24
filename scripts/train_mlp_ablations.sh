#!/usr/bin/env bash
# =============================================================================
# Train PretrainMLP with 3 small action MLP heads (≈32K, 64K, 128K params)
#
# Usage:
#   bash scripts/train_mlp_ablations.sh          # run sequentially on GPU 0
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_mlp_ablations.sh  # parallel
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

LEROBOT_ENV="lerobot"

# If false, runs sequentially (one GPU); if true, runs parallel (one GPU each)
PARALLEL=${PARALLEL:-false}

# --------------- Configuration ---------------
DATASET_REPO="grasp_multi_10"
DATASET_ROOT="datasets/luo_proj/Blind_Grasping_LeRobot_No_HandPC"
POLICY_TYPE="pretrain_mlp"
STEPS=60000
BATCH_SIZE=512
NUM_WORKERS=12
USE_AMP=true
WARMUP_STEPS=2000
SAVE_FREQ=10000
LOG_FREQ=1000
WANDB_ENABLE=false
PUSH_TO_HUB=false

# Output directory base
OUTPUT_BASE="checkpoints/luo_proj"

# ---- Model variants (mlp_hidden_dims → approx param count) ----
# With global_cond_dim=896, action_dim=22, horizon=8:
#   [32]  → ~35K params
#   [64]  → ~69K params
#   [128] → ~138K params

VARIANTS=(
    "32:pretrain_mlp_32"
    "64:pretrain_mlp_64"
    "128:pretrain_mlp_128"
)

# --------------- GPU assignment ---------------
# Count available GPUs
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra GPU_ARR <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_ARR=(0)
fi
NUM_GPUS=${#GPU_ARR[@]}

echo "============================================"
echo "Starting PretrainMLP ablation training"
echo "GPUs available: ${GPU_ARR[*]} (count=$NUM_GPUS)"
echo "Variants: ${#VARIANTS[@]}"
echo "Steps per variant: $STEPS"
echo "============================================"

# --------------- Launch training jobs ---------------
if [ "$PARALLEL" = "true" ]; then
    # Parallel mode: launch all jobs in background, one GPU each
    PID_LIST=()
    IDX=0
    for VARIANT in "${VARIANTS[@]}"; do
        IFS=':' read -r HIDDEN_DIM OUTPUT_SUBDIR <<< "$VARIANT"
        OUTPUT_DIR="${OUTPUT_BASE}/${OUTPUT_SUBDIR}"
        JOB_NAME="mlp_${HIDDEN_DIM}_${STEPS}steps"
        REPO_ID="${OUTPUT_SUBDIR}"
        GPU_ID=${GPU_ARR[$((IDX % NUM_GPUS))]}
        IDX=$((IDX + 1))

        echo ""
        echo "--- Launching variant: mlp_hidden_dims=[${HIDDEN_DIM}] → ${OUTPUT_DIR} (GPU ${GPU_ID}) ---"

        CUDA_VISIBLE_DEVICES="$GPU_ID" \
        conda run -n lerobot --no-capture-output lerobot-train \
            --dataset.repo_id="${DATASET_REPO}" \
            --dataset.root="${DATASET_ROOT}" \
            --policy.type="${POLICY_TYPE}" \
            --output_dir="${OUTPUT_DIR}" \
            --policy.repo_id="${REPO_ID}" \
            --batch_size="${BATCH_SIZE}" \
            --num_workers="${NUM_WORKERS}" \
            --policy.use_amp="${USE_AMP}" \
            --steps="${STEPS}" \
            --log_freq="${LOG_FREQ}" \
            --wandb.enable="${WANDB_ENABLE}" \
            --job_name="${JOB_NAME}" \
            --policy.push_to_hub="${PUSH_TO_HUB}" \
            --policy.scheduler_warmup_steps="${WARMUP_STEPS}" \
            --save_freq="${SAVE_FREQ}" \
            --policy.mlp_hidden_dims="[${HIDDEN_DIM}]" &

        PID_LIST+=($!)
    done

    echo ""
    echo "============================================"
    echo "All ${#VARIANTS[@]} training jobs launched. Waiting for completion..."
    echo "============================================"

    FAIL_COUNT=0
    for i in "${!PID_LIST[@]}"; do
        wait "${PID_LIST[$i]}" || { echo "Job $i (PID ${PID_LIST[$i]}) failed!"; FAIL_COUNT=$((FAIL_COUNT + 1)); }
    done
else
    # Sequential mode: train one after another on GPU 0
    FAIL_COUNT=0
    for VARIANT in "${VARIANTS[@]}"; do
        IFS=':' read -r HIDDEN_DIM OUTPUT_SUBDIR <<< "$VARIANT"
        OUTPUT_DIR="${OUTPUT_BASE}/${OUTPUT_SUBDIR}"
        JOB_NAME="mlp_${HIDDEN_DIM}_${STEPS}steps"
        REPO_ID="${OUTPUT_SUBDIR}"

        echo ""
        echo "========== Training variant: mlp_hidden_dims=[${HIDDEN_DIM}] → ${OUTPUT_DIR} =========="

        CUDA_VISIBLE_DEVICES=0 \
        conda run -n lerobot --no-capture-output lerobot-train \
            --dataset.repo_id="${DATASET_REPO}" \
            --dataset.root="${DATASET_ROOT}" \
            --policy.type="${POLICY_TYPE}" \
            --output_dir="${OUTPUT_DIR}" \
            --policy.repo_id="${REPO_ID}" \
            --batch_size="${BATCH_SIZE}" \
            --num_workers="${NUM_WORKERS}" \
            --policy.use_amp="${USE_AMP}" \
            --steps="${STEPS}" \
            --log_freq="${LOG_FREQ}" \
            --wandb.enable="${WANDB_ENABLE}" \
            --job_name="${JOB_NAME}" \
            --policy.push_to_hub="${PUSH_TO_HUB}" \
            --policy.scheduler_warmup_steps="${WARMUP_STEPS}" \
            --save_freq="${SAVE_FREQ}" \
            --policy.mlp_hidden_dims="[${HIDDEN_DIM}]" || {
                echo "Job for variant ${HIDDEN_DIM} failed!"; FAIL_COUNT=$((FAIL_COUNT + 1));
            }

        echo "========== Finished variant: mlp_hidden_dims=[${HIDDEN_DIM}] =========="
    done
fi

echo ""
echo "============================================"
if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "All training jobs completed successfully!"
else
    echo "${FAIL_COUNT} job(s) failed. Check logs above."
fi
echo "Results saved under: ${OUTPUT_BASE}/pretrain_mlp_{32,64,128}/"
echo "============================================"
