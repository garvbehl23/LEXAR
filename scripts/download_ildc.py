from datasets import load_dataset
from pathlib import Path
import json

OUT_DIR = Path("data/raw_docs/ildc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ds = load_dataset("shounakmulay/ILDC", split="train", streaming=True)

MAX_DOCS = 3000  # start controlled; we can scale to 10k+
kept = 0

out_file = OUT_DIR / "ildc_en.jsonl"
with open(out_file, "w") as f:
    for item in ds:
        text = (item.get("text") or "").strip()
        if len(text) > 800:
            f.write(json.dumps({
                "text": text,
                "source": "ILDC",
                "language": "en"
            }) + "\n")
            kept += 1
        if kept >= MAX_DOCS:
            break

print(f"Saved {kept} documents to {out_file}")
