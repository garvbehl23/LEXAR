from datasets import load_dataset
from pathlib import Path
import json

OUT_DIR = Path("data/raw_docs/indic_legal")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load ENGLISH split only (if available)
ds = load_dataset(
    "ai4bharat/indic-legal",
    split="train",
    streaming=True
)

MAX_DOCS = 2000   # start controlled; we can scale later
kept = 0

out_file = OUT_DIR / "indic_legal_en.jsonl"
with open(out_file, "w") as f:
    for item in ds:
        text = (item.get("text") or "").strip()
        lang = item.get("language", "").lower()

        # keep English only; skip short noise
        if lang in ("en", "english") and len(text) > 800:
            f.write(json.dumps({
                "text": text,
                "source": "IndicLegal",
                "language": "en"
            }) + "\n")
            kept += 1

        if kept >= MAX_DOCS:
            break

print(f"Saved {kept} documents to {out_file}")
