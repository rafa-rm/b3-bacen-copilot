from langchain_community.document_loaders import PyMuPDFLoader
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


project_root = Path(__file__).resolve().parent.parent

all_copom_files = sorted((project_root / "data").glob("Copom*.pdf"))

all_chunks = []
all_ids = []


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

for copom_file in all_copom_files:
    match = re.search(r"\d+", copom_file.stem)
    if match is None:
        raise ValueError(f"No meeting number in {copom_file.name}")

    meeting = match.group()

    loader = PyMuPDFLoader(
        file_path=str(copom_file),
        extract_tables="markdown",
        mode="single",
    )

    docs = loader.load()
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    chunks = text_splitter.split_documents(docs)
    print(
        f"Loaded document: {str(copom_file)}\nCopom Meeting: {meeting}, len(chunks) = {len(chunks)}"
    )

    for chunk in chunks:
        chunk.metadata["meeting"] = meeting
    ids = [
        f"file_{Path(chunk.metadata['source']).name}_start_idx_{chunk.metadata['start_index']}"
        for chunk in chunks
    ]

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

## Verifying idempotency
print(f"Chunks: {len(all_ids)} | Stored in Chroma: {vector_store._collection.count()}")

## Verifying chunks structure
# print(all_chunks[0])
# print(all_chunks[0])
# print(type(all_chunks[0]))
# print(len(all_chunks))


## Testing if chunk_overlap is working
# print(all_chunks[8].page_content[-150:])
# print("\n\n")
# print(all_chunks[9].page_content[:150])


## Checking when page boundary flips
## Chunk 4 - Page 1 and Chunk 5 - Page 2
# chunks_page = [all_chunks[x].metadata["page"] for x in range(10)]
# print(chunks_page)

## Verifying that mode = "single" don't break the continuity between pages
# print(all_chunks[4].page_content[-150:])
# print("\n\n")
# print(all_chunks[5].page_content[:150])
