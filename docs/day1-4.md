# Project Notes (Day 1–4)

## Overview
By the end of Day 4, the project is able to:
- Parse **text-based PDFs**
- Parse **DOCX files**
- Parse **scanned/image-based PDFs** using OCR
- Run a **FastAPI test server**
- Maintain a clean repo structure and GitHub history

---

## scripts/parse_pdf.py
**Purpose:**  
Extract text from text-based PDFs.

**Key Functions & APIs:**  
- `fitz.open(pdf_path)`: Opens the PDF.  
- `page.get_text()`: Extracts selectable text from each page.  
- CLI usage with `sys.argv` to pass a filename.  

**Notes:**  
- Only works if the PDF actually contains embedded text.  
- Scanned/image-only PDFs will return empty output → that’s why OCR is needed.  

---

## scripts/parse_docx.py
**Purpose:**  
Extract text from Microsoft Word documents (.docx).  

**Key Functions & APIs:**  
- `docx.Document(docx_path)`: Loads the document.  
- `doc.paragraphs`: Returns all paragraphs in the file.  
- Filters out empty/whitespace lines.  

**Notes:**  
- Works for all modern DOCX files.  
- Does not support older `.doc` (binary format).  

---

## scripts/ocr_pdf.py
**Purpose:**  
Handle scanned/image-based PDFs using PaddleOCR. Falls back to OCR if no selectable text is found.

**Key Functions & APIs:**  
- `page.get_text()`: First attempt at direct extraction.  
- `page.get_pixmap(dpi=200)`: Renders page as an image at 200 DPI.  
- `ocr = PaddleOCR(use_angle_cls=True, lang='en')`: Initializes OCR model.  
- `ocr.ocr(img_path, cls=True)`: Runs OCR on rendered image.  

**Notes:**  
- Uses “fast path” (PyMuPDF) if text exists, otherwise “slow path” (OCR).  
- First run downloads OCR models (~200 MB). After that, they are cached in `.paddlex/`.  
- Temporary PNGs are deleted after processing each page.  
- Higher DPI → better accuracy but slower runtime.  

---

## Progress Summary (Day 1–4)
- **Day 1:** Repo structure created, Git initialized, pushed to GitHub.  
- **Day 2:** PyCharm venv setup, dependencies installed, FastAPI “Hello World” running.  
- **Day 3:** PDF and DOCX parsing scripts implemented and tested.  
- **Day 4:** OCR integration completed, robust parsing for both text-based and scanned PDFs.  

---

