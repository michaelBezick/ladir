import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf as om
from peft import LoraConfig
from safetensors.torch import load_file
from tqdm import tqdm
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from answer_utils import answers_match
from model import LMFusionModel
from scorer_data import write_shard
from vae.model_vae import VAE
from vae.vae_args import parse_args as parse_vae_args


def parse_args():
    parser = argparse.ArgumentParser(description="Generate on-policy latent scorer training data.")
    parser.add_argument("--config", default="configs/cd_formal_8B_VAE_conn.yaml")
    parser.add_argument("--data-file", required=True, help="JSONL math data with input and output/solutions")
    parser.add_argument("--diffusion-ckpt", default=None, help="Optional diffusion checkpoint file or directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    return parser.parse_args()


def add_thought_tokens(tokenizer):
    for special_token in ["<tht_s>", "<tht>", "</tht_s>", "<timestep>"]:
        tokenizer.add_special_tokens({"additional_special_tokens": [special_token]})
    tokenizer.bot_token_id = tokenizer.convert_tokens_to_ids("<tht_s>")
    tokenizer.tht_token_id = tokenizer.convert_tokens_to_ids("<tht>")
    tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids("</tht_s>")
    tokenizer.time_token_id = tokenizer.convert_tokens_to_ids("<timestep>")
    tokenizer.pad_token_id = tokenizer.eos_token_id


def load_diffusion_model(cfg, device):
    ae_lora_config = LoraConfig(
        r=cfg.ae.lora_r,
        lora_alpha=cfg.ae.lora_alpha,
        lora_dropout=cfg.ae.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    argv = sys.argv
    try:
        sys.argv = [argv[0]]
        ae_model_args, ae_training_args, _ = parse_vae_args()
    finally:
        sys.argv = argv
    ae = VAE(ae_model_args, ae_training_args, ae_lora_config)
    ae_state = load_file(cfg.ae.icae_ckpt)
    ae.load_state_dict(ae_state.get("state_dict", ae_state), strict=False)
    for param in ae.parameters():
        param.requires_grad = False
    ae = ae.to(device, dtype=torch.bfloat16).eval()

    text_config = AutoConfig.from_pretrained(
        cfg.model.llm_model_name_or_path,
        use_flash_attention=False,
        _flash_attn_2_enabled=False,
        local_files_only=True,
    )
    text_llama = AutoModelForCausalLM.from_pretrained(
        cfg.model.llm_model_name_or_path,
        config=text_config,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.llm_model_name_or_path, local_files_only=True)
    add_thought_tokens(tokenizer)

    model = LMFusionModel(
        text_llama=text_llama,
        thought_llama=None,
        autoencoder=ae,
        model_config=cfg,
        tokenizer=tokenizer,
        hidden_dim=text_config.hidden_size,
        freeze_text=False,
    ).to(device=device, dtype=torch.bfloat16)
    return model.eval(), tokenizer


def load_checkpoint(model, checkpoint):
    if checkpoint is None:
        return
    path = Path(checkpoint)
    if path.is_dir():
        for name in ["model.safetensors", "pytorch_model.bin"]:
            candidate = path / name
            if candidate.exists():
                path = candidate
                break
    if path.suffix == ".safetensors":
        state = load_file(str(path))
    else:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded diffusion checkpoint: missing={len(missing)} unexpected={len(unexpected)}")


def iter_examples(path, max_examples=None):
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_examples is not None and idx >= max_examples:
                break
            if line.strip():
                yield idx, json.loads(line)


def main():
    args = parse_args()
    cfg = om.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_diffusion_model(cfg, device)
    load_checkpoint(model, args.diffusion_ckpt)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = output_dir / "metadata.jsonl"
    if metadata.exists():
        raise FileExistsError(f"{metadata} already exists; remove it or choose a new output dir")

    shard_id = 0
    shard_latents = []
    shard_records = []

    for example_id, example in tqdm(list(iter_examples(args.data_file, args.max_examples)), desc="examples"):
        question = example["input"]
        target = example.get("solutions") or example.get("answer") or example.get("output")
        reasoning = example.get("reasoning_text") or f"{question}\n{example.get('output', '')}"
        input_ids_q = tokenizer(question + "\n", return_tensors="pt")["input_ids"].to(device)

        with torch.no_grad():
            gt_tokens = model.autoencoder.encode_text(reasoning)
        gt_tokens = gt_tokens.reshape(1, -1, model.tht_token_dim).to(device=device, dtype=torch.bfloat16)

        for rollout_id in range(args.num_rollouts):
            with torch.no_grad():
                states, timesteps = model.denoise_debug(
                    input_ids_q,
                    gt_tokens,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                )
                trajectory = states[0].detach().cpu().float()
                final_text = model.autoencoder.decode_text(states[0, -1].unsqueeze(0))
            label = 1.0 if answers_match(final_text, target) else 0.0

            for step_id, timestep in enumerate(timesteps):
                shard_latents.append(trajectory[step_id])
                shard_records.append(
                    {
                        "example_id": example_id,
                        "rollout_id": rollout_id,
                        "step_id": step_id,
                        "input": question,
                        "target": target,
                        "timestep": float(timestep),
                        "label": label,
                        "final_text": final_text,
                    }
                )

                if len(shard_latents) >= args.shard_size:
                    write_shard(str(output_dir), shard_id, shard_latents, shard_records)
                    shard_id += 1
                    shard_latents = []
                    shard_records = []

    if shard_latents:
        write_shard(str(output_dir), shard_id, shard_latents, shard_records)


if __name__ == "__main__":
    main()
