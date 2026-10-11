#!/usr/bin/env bash
# Single-GPU PickXtimes runs. Stage 1 is the memory-free π0.5 baseline.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mode="${1:?Usage: bash scripts/t2mem/pickxtimes_a100.sh baseline|t2mem|baseline-matched}"
data_path="${PICKXTIMES_DATA:-data/t2mem_pickxtimes}"
run_root="${PICKXTIMES_RUNS:-runs/pickxtimes}"
base_weights="${PI05_BASE_WEIGHTS:-checkpoints/pi05_base_pytorch/model.safetensors}"
train_seed="${TRAIN_SEED:-0}"
memory_updates="${T2MEM_UPDATES:-2000}"
common=(--data "$data_path" --tasks PickXtimes --seed "$train_seed" --activation-checkpointing)
if [[ "${ACTIVATION_OFFLOAD:-0}" == 1 ]]; then
    common+=(--activation-offload)
fi

case "$mode" in
    baseline)
        uv run python scripts/t2mem/train.py "${common[@]}" \
            --stage stage1 --base-weights "$base_weights" \
            --output "$run_root/pi05_baseline" --updates 20000 --batch-size 64
        ;;
    t2mem)
        uv run python scripts/t2mem/train.py "${common[@]}" \
            --stage alternating --init "$run_root/pi05_baseline/step_020000.pt" \
            --output "$run_root/t2mem" --updates "$memory_updates" --batch-size 32
        ;;
    baseline-matched)
        # Optional control with the same number of outer optimizer updates.
        uv run python scripts/t2mem/train.py "${common[@]}" \
            --stage stage1 --resume "$run_root/pi05_baseline/step_020000.pt" \
            --output "$run_root/pi05_baseline_matched" \
            --updates "$((20000 + memory_updates))" --batch-size 64
        ;;
    *)
        printf 'Unknown mode: %s\n' "$mode" >&2
        exit 2
        ;;
esac
