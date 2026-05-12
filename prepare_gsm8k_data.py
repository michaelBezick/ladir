import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def reasoning_from_example(example: Dict) -> str:
    steps = example.get("steps") or []
    answer = str(example.get("answer", "")).strip()
    lines: List[str] = []
    for step in steps:
        step = str(step).strip()
        if step:
            lines.append(step)
    if answer:
        lines.append(f"#### {answer}")
    return "\n".join(lines)


def convert_example(example: Dict) -> Dict:
    question = str(example["question"]).strip()
    answer = str(example.get("answer", "")).strip()
    output = reasoning_from_example(example)
    return {
        "input": question,
        "output": output,
        "answer": answer,
        "solutions": [answer] if answer else [],
    }


def load_json(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected {path} to contain a JSON list")
    return data


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare copied GSM8K JSON files for LaDiR training.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-source", default="gsm_train.json")
    parser.add_argument("--val-source", default="gsm_valid.json")
    parser.add_argument("--test-source", default="gsm_test.json")
    parser.add_argument("--hard-source", default="gsm_hard.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    splits = {
        "train": args.train_source,
        "val": args.val_source,
        "test": args.test_source,
        "hard": args.hard_source,
    }

    counts = {}
    for split, filename in splits.items():
        source = data_dir / filename
        rows = [convert_example(example) for example in load_json(source)]
        counts[f"{split}.jsonl"] = write_jsonl(data_dir / f"{split}.jsonl", rows)
        if split == "train":
            counts["vae_train.jsonl"] = write_jsonl(data_dir / "vae_train.jsonl", rows)
        elif split == "val":
            counts["vae_val.jsonl"] = write_jsonl(data_dir / "vae_val.jsonl", rows)

    print("Prepared GSM8K data:")
    for name, count in counts.items():
        print(f"  {data_dir / name}: {count}")


if __name__ == "__main__":
    main()
