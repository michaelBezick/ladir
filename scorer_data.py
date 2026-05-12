import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class LatentScorerDataset(Dataset):
    """Loads scorer metadata JSONL records and latent tensor shards."""

    def __init__(
        self,
        metadata_file: str,
        tokenizer,
        max_question_length: int = 512,
    ):
        self.metadata_file = Path(metadata_file)
        self.root = self.metadata_file.parent
        self.tokenizer = tokenizer
        self.max_question_length = max_question_length
        with self.metadata_file.open("r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.records)

    @lru_cache(maxsize=8)
    def _load_shard(self, shard_name: str) -> Dict[str, torch.Tensor]:
        shard_path = Path(shard_name)
        if not shard_path.is_absolute():
            shard_path = self.root / shard_path
        return torch.load(shard_path, map_location="cpu")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        shard = self._load_shard(record["shard"])
        row = int(record["row"])
        question = record.get("input") or record.get("question") or ""
        input_ids = self.tokenizer(
            question,
            truncation=True,
            max_length=self.max_question_length,
            return_tensors="pt",
        )["input_ids"][0]
        return {
            "input_ids_q": input_ids,
            "x_t": shard["latents"][row].float(),
            "timestep": torch.tensor(float(record["timestep"]), dtype=torch.float32),
            "label": torch.tensor(float(record["label"]), dtype=torch.float32),
        }


class LatentScorerCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = list(batch)
        return {
            "input_ids_q": pad_sequence(
                [item["input_ids_q"] for item in batch],
                batch_first=True,
                padding_value=self.pad_token_id,
            ),
            "x_t": torch.stack([item["x_t"] for item in batch], dim=0),
            "timestep": torch.stack([item["timestep"] for item in batch], dim=0),
            "label": torch.stack([item["label"] for item in batch], dim=0),
        }


def write_shard(
    output_dir: str,
    shard_id: int,
    latents: List[torch.Tensor],
    records: List[Dict],
    metadata_name: str = "metadata.jsonl",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    shard_name = f"latents-{shard_id:06d}.pt"
    torch.save({"latents": torch.stack(latents, dim=0).cpu()}, output_path / shard_name)
    with (output_path / metadata_name).open("a", encoding="utf-8") as f:
        for row, record in enumerate(records):
            item = dict(record)
            item["shard"] = shard_name
            item["row"] = row
            f.write(json.dumps(item) + "\n")
