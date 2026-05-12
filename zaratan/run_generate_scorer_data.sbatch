#!/usr/bin/env bash
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?REPO_ROOT not set}"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
CONFIG_PATH="${CONFIG_PATH:-configs/cd_formal_8B_VAE_conn.yaml}"
DATA_FILE="${DATA_FILE:-data/train.jsonl}"
DIFFUSION_CKPT="${DIFFUSION_CKPT:?DIFFUSION_CKPT not set}"
OUTPUT_DIR="${OUTPUT_DIR:-data/scorer}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"
source "${VENV_PATH}/bin/activate"

export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cmd=(
  python generate_scorer_data.py
  --config "${CONFIG_PATH}"
  --data-file "${DATA_FILE}"
  --diffusion-ckpt "${DIFFUSION_CKPT}"
  --output-dir "${OUTPUT_DIR}"
  --num-rollouts "${NUM_ROLLOUTS}"
)

if [[ -n "${MAX_EXAMPLES}" ]]; then cmd+=(--max-examples "${MAX_EXAMPLES}"); fi
if [[ -n "${EXTRA_ARGS}" ]]; then
  read -r -a extra_args <<< "${EXTRA_ARGS}"
  cmd+=("${extra_args[@]}")
fi

printf 'Running: %s\n' "${cmd[*]}"
srun "${cmd[@]}"
