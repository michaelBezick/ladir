#!/usr/bin/env bash
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?REPO_ROOT not set}"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
METADATA="${METADATA:-data/scorer/metadata.jsonl}"
VAL_METADATA="${VAL_METADATA:-}"
LLM_MODEL="${LLM_MODEL:-/scratch/zt1/project/kaiqing-prj/user/mbezick/ladir/models/Llama-3.1-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/latent_scorer}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"
source "${VENV_PATH}/bin/activate"
python zaratan/check_python_env.py

export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cmd=(
  python train_scorer.py
  --metadata "${METADATA}"
  --model-name-or-path "${LLM_MODEL}"
  --output-dir "${OUTPUT_DIR}"
  --bf16
)

if [[ -n "${VAL_METADATA}" ]]; then cmd+=(--val-metadata "${VAL_METADATA}"); fi
if [[ -n "${EXTRA_ARGS}" ]]; then
  read -r -a extra_args <<< "${EXTRA_ARGS}"
  cmd+=("${extra_args[@]}")
fi

printf 'Running: %s\n' "${cmd[*]}"
srun "${cmd[@]}"
