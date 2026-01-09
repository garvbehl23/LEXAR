import requests
from pathlib import Path

OUT = Path("data/raw_docs/statutes")
OUT.mkdir(parents=True, exist_ok=True)

URLS = {
    "ipc.pdf": "https://legislative.gov.in/sites/default/files/A1860-45.pdf",
    "crpc.pdf": "https://legislative.gov.in/sites/default/files/A1974-02.pdf",
    "cpc.pdf": "https://legislative.gov.in/sites/default/files/A1908-05.pdf",
    "evidence_act.pdf": "https://legislative.gov.in/sites/default/files/A1872-01.pdf"
}

for name, url in URLS.items():
    print(f"Downloading {name}...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(OUT / name, "wb") as f:
        f.write(r.content)

print("Indian statutes downloaded successfully.")
