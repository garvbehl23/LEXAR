import subprocess
from pathlib import Path

OUT_DIR = Path("data/raw_docs/judgments_large")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "amitkumarjaiswal/indian-supreme-court-judgments"

cmd = [
    "kaggle", "datasets", "download",
    "-d", DATASET,
    "-p", str(OUT_DIR),
    "--unzip"
]

print("Downloading Indian Supreme Court judgments...")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode != 0:
    print("ERROR downloading dataset:")
    print(result.stderr)
    raise RuntimeError("Kaggle download failed")

print("Download completed successfully.")
