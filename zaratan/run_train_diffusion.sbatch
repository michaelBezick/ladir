#!/usr/bin/env bash
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?REPO_ROOT not set}"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
CONFIG_PATH="${CONFIG_PATH:-configs/cd_formal_8B_VAE_conn.yaml}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
MASTER_PORT="${MASTER_PORT:-46089}"
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
  python -m torch.distributed.run
  --nproc_per_node "${NUM_GPUS}"
  --master_port "${MASTER_PORT}"
  train.py
  "${CONFIG_PATH}"
)

if [[ -n "${EXTRA_ARGS}" ]]; then
  read -r -a extra_args <<< "${EXTRA_ARGS}"
  cmd+=("${extra_args[@]}")
fi

printf 'Running: %s\n' "${cmd[*]}"
srun "${cmd[@]}"
