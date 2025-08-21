import os, json
from PyPDF2 import PdfReader
from docx import Document

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def parse_docx(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def parse_pdf(path):
    reader = PdfReader(path)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    all_chunks = []
    pdf_path = os.path.join(data_dir, "scanned_sample.pdf")
    docx_path = os.path.join(data_dir, "sample.docx")
    pdf_text = parse_pdf(pdf_path)
    pdf_chunks = chunk_text(pdf_text)
    for i, c in enumerate(pdf_chunks):
        all_chunks.append({"text": c, "source": "scanned_sample.pdf", "page": i})
    docx_text = parse_docx(docx_path)
    docx_chunks = chunk_text(docx_text)
    for i, c in enumerate(docx_chunks):
        all_chunks.append({"text": c, "source": "sample.docx", "page": i})
    chunks_file = os.path.join(data_dir, "chunks.json")
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_chunks)} chunks to {chunks_file}")
