# LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning

Official repository for the paper:  
**[LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning](https://arxiv.org/abs/2510.04573)**  

---

## 🧠 Overview

**LaDiR (Latent Diffusion Reasoner)** introduces a new reasoning framework that unifies the expressiveness of **continuous latent representations** with the **iterative refinement capability** of diffusion models for large language models (LLMs).

Instead of generating reasoning chains autoregressively, LaDiR performs **latent diffusion over thought tokens**, enabling:

- Iterative semantic self-refinement  
- Diverse parallel reasoning trajectories  
- A flexible trade-off between accuracy and test-time compute  

---


## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Training the VAE Model

1. **Prepare your dataset** in JSONL format with the following structure:
   ```json
   {"input": "question text", "output": "reasoning chain"}
   ```

2. **Configure training parameters** in `configs/cd_formal_8B_VAE_conn.yaml`

3. **Run VAE training**:
   ```bash
   cd vae
   bash ..scripts/train_vae.sh
   ```

### Training the Diffusion Model
   ```bash
   bash scripts/train_vae.sh
   ```

### Preparing DART-Math Data

The LaDiR paper trains on DART-Math and evaluates on held-out math benchmarks.
Prepare the JSONL files expected by this repo with:

```bash
python prepare_dart_math_data.py
```

By default this downloads `hkust-nlp/dart-math-hard`, writes deterministic
train/validation/test splits, and mirrors the training/validation splits to the
VAE filenames:

- `data/vae_train.jsonl` and `data/vae_val.jsonl` for VAE training.
- `data/train.jsonl` and `data/val.jsonl` for diffusion training.
- `data/test.jsonl` and `data/hard.jsonl` for held-out checks.

### Preparing GSM8K Data

If `data/gsm_train.json`, `data/gsm_valid.json`, `data/gsm_test.json`, and
`data/gsm_hard.json` are present, prepare the JSONL files expected by this repo:

```bash
python prepare_gsm8k_data.py
```

This writes:

- `data/vae_train.jsonl` and `data/vae_val.jsonl` for VAE training.
- `data/train.jsonl` and `data/val.jsonl` for diffusion training.
- `data/test.jsonl` and `data/hard.jsonl` for held-out evaluation.

### Training a Latent Scorer for Diffusion Steering

1. Generate on-policy scorer data from diffusion rollouts:
   ```bash
   python generate_scorer_data.py \
     --data-file data/train.jsonl \
     --diffusion-ckpt ckpt/vae_diffusion_experiment/checkpoint-2500 \
     --output-dir data/scorer \
     --num-rollouts 8
   ```

2. Train the scorer:
   ```bash
   python train_scorer.py \
     --metadata data/scorer/metadata.jsonl \
     --model-name-or-path meta-llama/Llama-3.1-8B \
     --output-dir checkpoints/latent_scorer
   ```

The scorer exposes `get_reward(...)` and `get_reward_gradient(...)` for use with
`LMFusionModel.denoise_with_reward_guidance(...)`.

## ⚙️ Configuration

The model can be configured through YAML files in the `configs/` directory. Key parameters include:

- **Model**: Base language model path, LoRA configuration
- **Training**: Learning rate, batch size, number of steps
- **VAE**: Compression rate, memory size, beta for KL loss
- **Dataset**: Training file paths, data processing options

---

If you find this work useful, please consider citing:

```bibtex
@article{kang2025ladir,
  title={LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning},
  author={Kang, Haoqiang and Zhang, Yizhe and Kuang, Nikki Lijing and Majamäki, Nicklas and Jaitly, Navdeep and Ma, Yi-An and Qin, Lianhui},
  journal={arXiv preprint arXiv:2510.08558},
  year={2025}
}
