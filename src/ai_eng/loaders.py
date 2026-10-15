"""Turn the files in the knowledge folder into LangChain Documents."""
from pathlib import Path
from langchain_community.document_loaders import (
    CSVLoader, PyMuPDFLoader, TextLoader, UnstructuredPDFLoader,
)
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

SCAN_FOLDERS = {"scans"}

def _pdf(path: Path):
    if path.parent.name in SCAN_FOLDERS:
        return UnstructuredPDFLoader(str(path), mode="elements", strategy="ocr_only")
    return PyMuPDFLoader(str(path))

# Define headers to extract
headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

def _markdown(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    return splitter.split_text(text)

LOADER_FOR = {
    ".pdf": _pdf,
    ".md": _markdown,  # <-- Use our custom markdown splitter function here
    ".txt": lambda p: TextLoader(str(p), encoding="utf-8"),
    ".csv": lambda p: CSVLoader(str(p), encoding="utf-8", metadata_columns=["product", "category"]),
}

def load_file(path: Path) -> list[Document]:
    loader_func = LOADER_FOR[path.suffix.lower()]
    
    # Markdown splitter returns a list of Documents directly; others need .load()
    if path.suffix.lower() == ".md":
        docs = loader_func(path)
    else:
        docs = loader_func(path).load()

    for doc in docs:
        doc.metadata.update(
            filename=path.name,
            area=path.parent.name,
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
        except Exception as exc:
            failed.append(f"{path.name}: {exc}")
    print(f"Loaded {len(docs)} documents from {root}")
    for line in failed:
        print("  skipped", line)
    return docs