from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
import re

from rag.config import COLLECTION_NAME, EMBEDDING_MODEL, PERSIST_DIR


def clean_text(text: str) -> str:
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"^\s*\d+\s*\n+bcb\.gov\.br\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+ª Reunião\s*\nCOPOM\s*\n.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_meeting_number(file_path: Path) -> str:
    match = re.search(r"\d+", file_path.stem)
    if match is None:
        raise ValueError(f"No meeting number in {file_path.name}")
    return match.group()


def build_copom_chunks(
    docs: list[Document],
    meeting: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> tuple[list[Document], list[str]]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    cleaned_docs = [
        Document(
            page_content=clean_text(doc.page_content),
            metadata=dict(doc.metadata),
        )
        for doc in docs
    ]

    chunks = text_splitter.split_documents(cleaned_docs)
    for chunk in chunks:
        chunk.metadata["meeting"] = meeting

    ids = [
        f"file_{Path(chunk.metadata['source']).name}_start_idx_{chunk.metadata['start_index']}"
        for chunk in chunks
    ]
    return chunks, ids


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    all_copom_files = sorted((project_root / "data").glob("Copom*.pdf"))

    all_chunks: list[Document] = []
    all_ids: list[str] = []

    for copom_file in all_copom_files:
        meeting = extract_meeting_number(copom_file)
        loader = PyMuPDFLoader(
            file_path=str(copom_file),
            extract_tables="markdown",
            mode="single",
        )
        docs = loader.load()
        chunks, ids = build_copom_chunks(docs=docs, meeting=meeting)
        print(
            f"Loaded document: {copom_file}\nCopom Meeting: {meeting}, len(chunks) = {len(chunks)}"
        )
        all_chunks.extend(chunks)
        all_ids.extend(ids)

    if not all_chunks:
        raise RuntimeError("No Copom PDFs found in data/")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(project_root / PERSIST_DIR.lstrip("/")),
    )

    vector_store.reset_collection()
    vector_store.add_documents(documents=all_chunks, ids=all_ids)

    print(
        f"Chunks: {len(all_ids)} | Stored in Chroma: {vector_store._collection.count()}"
    )


if __name__ == "__main__":
    main()
