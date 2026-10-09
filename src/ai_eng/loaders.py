"""Turn the files in the knowledge folder into LangChain Documents."""
from pathlib import Path

from langchain_community.document_loaders import (
    CSVLoader, PyMuPDFLoader, TextLoader, UnstructuredPDFLoader,
)
from langchain_core.documents import Document

SCAN_FOLDERS = {"scans"}      # PDFs kept here are images, so they need OCR


def _pdf(path: Path):
    if path.parent.name in SCAN_FOLDERS:
        return UnstructuredPDFLoader(str(path), mode="elements", strategy="ocr_only")
    return PyMuPDFLoader(str(path))


LOADER_FOR = {
    ".pdf": _pdf,
    ".md": lambda p: TextLoader(str(p), encoding="utf-8"),
    ".txt": lambda p: TextLoader(str(p), encoding="utf-8"),
    ".csv": lambda p: CSVLoader(str(p), encoding="utf-8", metadata_columns=["product", "category"]),
}


def load_file(path: Path) -> list[Document]:
    docs = LOADER_FOR[path.suffix.lower()](path).load()
    for doc in docs:
        doc.metadata.update(
            filename=path.name,
            area=path.parent.name,                 # policies, wiki, faq, scans ...
            doc_type=path.suffix.lstrip(".").lower(),
        )
    return docs


def load_corpus(root: Path) -> list[Document]:
    docs, failed = [], []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in LOADER_FOR:
            continue
        try:
            docs.extend(load_file(path))
        except Exception as exc:                   # keep going, report at the end
            failed.append(f"{path.name}: {exc}")
    print(f"Loaded {len(docs)} documents from {root}")
    for line in failed:
        print("  skipped", line)
    return docs
