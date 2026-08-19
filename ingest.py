"""
Ingestion script.

Loads every .md / .docx file in data/, splits it into chunks, embeds each
chunk, and stores the result in a local Chroma vector database at
chroma_db/.

Run this once to build the knowledge base, and again any time you add or
edit files in data/:

    python ingest.py
"""
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
PERSIST_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "knowledge_bank"
EMBEDDING_MODEL = "gemini-embedding-001"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


def load_documents():
    """Load every .md and .docx file in DATA_DIR as a LangChain Document,
    tagging each with its source filename so citations work later."""
    docs = []
    files = sorted(DATA_DIR.glob("*.md")) + sorted(DATA_DIR.glob("*.docx"))

    if not files:
        print(f"No .md or .docx files found in {DATA_DIR}. Add some and re-run.")
        sys.exit(1)

    for path in files:
        loader = UnstructuredMarkdownLoader(str(path)) if path.suffix == ".md" else Docx2txtLoader(str(path))
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = path.name
        docs.extend(loaded)
        print(f"  loaded {path.name} ({len(loaded)} doc(s))")

    return docs


def main():
    print(f"Loading files from {DATA_DIR} ...")
    documents = load_documents()

    print("Splitting into chunks ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"  {len(documents)} document(s) -> {len(chunks)} chunk(s)")

    print("Embedding and storing in Chroma ...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
    )

    print(f"Done. Vector store saved to {PERSIST_DIR}")


if __name__ == "__main__":
    main()
