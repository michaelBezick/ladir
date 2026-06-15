#!/usr/bin/env python3
import json
import random
import re
import urllib.request
from pathlib import Path


DATA_DIR = Path("data")
SEED = 42
VALID_SIZE = 500
HARD_SIZE = 500
URLS = {
    "train": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
    "test": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
}


def split_answer(answer_text):
    if "####" in answer_text:
        reasoning, final = answer_text.rsplit("####", 1)
    else:
        reasoning, final = answer_text, ""

    steps = [line.strip() for line in reasoning.strip().splitlines() if line.strip()]
    final = final.strip()
    return steps, final


def load_jsonl_url(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode("utf-8")
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        steps, answer = split_answer(raw["answer"])
        rows.append(
            {
                "question": raw["question"].strip(),
                "steps": steps,
                "answer": answer,
            }
        )
    return rows


def answer_complexity(example):
    text = "\n".join(example["steps"])
    arithmetic_marks = len(re.findall(r"<<[^>]+>>", text))
    token_count = len(text.split())
    return (len(example["steps"]), arithmetic_marks, token_count)


def write_json(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main():
    train = load_jsonl_url(URLS["train"])
    test = load_jsonl_url(URLS["test"])

    indices = list(range(len(train)))
    random.Random(SEED).shuffle(indices)
    valid_indices = set(indices[:VALID_SIZE])
    valid = [row for idx, row in enumerate(train) if idx in valid_indices]
    train_split = [row for idx, row in enumerate(train) if idx not in valid_indices]
    hard = sorted(test, key=answer_complexity, reverse=True)[:HARD_SIZE]

    write_json(DATA_DIR / "gsm_train.json", train_split)
    write_json(DATA_DIR / "gsm_valid.json", valid)
    write_json(DATA_DIR / "gsm_test.json", test)
    write_json(DATA_DIR / "gsm_hard.json", hard)

    print("Downloaded GSM8K and wrote repo source splits:")
    print(f"  {DATA_DIR / 'gsm_train.json'}: {len(train_split)}")
    print(f"  {DATA_DIR / 'gsm_valid.json'}: {len(valid)}")
    print(f"  {DATA_DIR / 'gsm_test.json'}: {len(test)}")
    print(f"  {DATA_DIR / 'gsm_hard.json'}: {len(hard)}")


if __name__ == "__main__":
    main()
