import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from scorer import LatentReasoningScorer, LatentScorerConfig
from scorer_data import LatentScorerCollator, LatentScorerDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train a latent reasoning correctness scorer.")
    parser.add_argument("--metadata", required=True, help="Training metadata JSONL from generate_scorer_data.py")
    parser.add_argument("--val-metadata", default=None, help="Optional validation metadata JSONL")
    parser.add_argument("--model-name-or-path", required=True, help="Tokenizer source for question tokenization")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--max-question-length", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, use_bf16=False):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    positives = 0.0
    for batch in loader:
        input_ids_q = batch["input_ids_q"].to(device)
        x_t = batch["x_t"].to(device)
        timestep = batch["timestep"].to(device)
        label = batch["label"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16 and device.type == "cuda"):
            logits = model(input_ids_q, x_t, timestep)
            loss = F.binary_cross_entropy_with_logits(logits, label)
        prob = torch.sigmoid(logits.float())
        pred = prob >= 0.5
        total_loss += loss.item() * label.numel()
        total += label.numel()
        correct += pred.eq(label.bool()).sum().item()
        positives += label.float().sum().item()
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "positive_rate": positives / max(total, 1),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dataset = LatentScorerDataset(args.metadata, tokenizer, args.max_question_length)
    collator = LatentScorerCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    val_loader = None
    if args.val_metadata:
        val_dataset = LatentScorerDataset(args.val_metadata, tokenizer, args.max_question_length)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=True,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = LatentScorerConfig(
        vocab_size=len(tokenizer),
        latent_dim=args.latent_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = LatentReasoningScorer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            input_ids_q = batch["input_ids_q"].to(device)
            x_t = batch["x_t"].to(device)
            timestep = batch["timestep"].to(device)
            label = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16 and device.type == "cuda"):
                logits = model(input_ids_q, x_t, timestep)
                loss = F.binary_cross_entropy_with_logits(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        metrics = evaluate(model, train_loader, device, use_bf16=args.bf16)
        print(f"train epoch={epoch + 1} {metrics}")
        if val_loader is not None:
            print(f"val epoch={epoch + 1} {evaluate(model, val_loader, device, use_bf16=args.bf16)}")
        model.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
