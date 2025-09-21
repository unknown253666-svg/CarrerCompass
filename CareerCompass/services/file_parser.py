import re, PyPDF2, docx2txt

def extract_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        raw = " ".join(PyPDF2.PdfReader(path).pages[i].extract_text() for i in range(len(PyPDF2.PdfReader(path).pages)))
    else:
        raw = docx2txt.process(path)
    raw = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", raw)
    raw = re.sub(r"(?<=\w)\n(?=\w)", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()