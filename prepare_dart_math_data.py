import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from datasets import load_dataset


BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
ANSWER_RE = re.compile(r"(?:the\s+answer\s+is|answer)\s*:?\s*([^\n\.]+)", re.IGNORECASE)


def clean_answer(answer: str) -> str:
    answer = answer.strip().strip("$").strip()
    answer = answer.lstrip(":").strip().strip("$").strip()
    return answer


def extract_answer(response: str) -> str:
    boxed = BOXED_RE.findall(response)
    if boxed:
        return clean_answer(boxed[-1])

    matches = ANSWER_RE.findall(response)
    if matches:
        return clean_answer(matches[-1])

    return ""


def convert_example(example: Dict) -> Dict:
    question = str(example["query"]).strip()
    output = str(example["response"]).strip()
    answer = extract_answer(output)
    return {
        "input": question,
        "output": output,
        "answer": answer,
        "solutions": [answer] if answer else [],
    }


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def split_rows(rows: List[Dict], val_size: int, test_size: int, seed: int) -> Dict[str, List[Dict]]:
    if val_size + test_size >= len(rows):
        raise ValueError("val_size + test_size must be smaller than the dataset size")

    rows = list(rows)
    random.Random(seed).shuffle(rows)
    val = rows[:val_size]
    test = rows[val_size : val_size + test_size]
    train = rows[val_size + test_size :]
    return {"train": train, "val": val, "test": test}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DART-Math JSONL files for LaDiR training.")
    parser.add_argument("--dataset-name", default="hkust-nlp/dart-math-hard")
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name, split=args.split)
    rows = [convert_example(example) for example in dataset]
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    splits = split_rows(rows, args.val_size, args.test_size, args.seed)
    data_dir = Path(args.data_dir)

    counts = {
        "train.jsonl": write_jsonl(data_dir / "train.jsonl", splits["train"]),
        "val.jsonl": write_jsonl(data_dir / "val.jsonl", splits["val"]),
        "test.jsonl": write_jsonl(data_dir / "test.jsonl", splits["test"]),
        "hard.jsonl": write_jsonl(data_dir / "hard.jsonl", splits["test"]),
        "vae_train.jsonl": write_jsonl(data_dir / "vae_train.jsonl", splits["train"]),
        "vae_val.jsonl": write_jsonl(data_dir / "vae_val.jsonl", splits["val"]),
    }

    print(f"Prepared DART-Math data from {args.dataset_name}:")
    for name, count in counts.items():
        print(f"  {data_dir / name}: {count}")


if __name__ == "__main__":
    main()
