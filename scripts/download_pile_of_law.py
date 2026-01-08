from datasets import load_dataset
import json
from pathlib import Path

OUT_DIR = Path("data/raw_docs/hf_law")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load a subset (safe start)
dataset = load_dataset(
    "pile-of-law/pile-of-law",
    split="train[:1%]"  # START WITH 1%
)

out_file = OUT_DIR / "pile_of_law_sample.jsonl"

with open(out_file, "w") as f:
    for item in dataset:
        text = item.get("text", "").strip()
        if len(text) > 500:
            f.write(json.dumps({"text": text}) + "\n")

print(f"Saved {out_file}")
