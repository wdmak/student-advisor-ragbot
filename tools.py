"""
The agent's one tool: search the knowledge base.

Wraps the Chroma vector store in a LangChain @tool so the agent can call it
by name and decide, on its own, when a question needs a lookup. Each result
carries its source filename so the agent (and the UI) can cite it.
"""
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PERSIST_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "knowledge_bank"
EMBEDDING_MODEL = "gemini-embedding-001"
TOP_K = 4

_embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
_vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=_embeddings,
    persist_directory=str(PERSIST_DIR),
)


@tool
def search_knowledge_base(query: str) -> str:
    """Search the student-shared job-hunting knowledge base for information
    relevant to the query. Use this whenever a question could be answered by
    students' shared interview experiences, job feelings, or techniques and
    skills. Always cite the source filename shown for each result."""
    results = _vector_store.similarity_search(query, k=TOP_K)

    if not results:
        return "No relevant results found in the knowledge base."

    formatted = []
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Result {i}, source: {source}]\n{doc.page_content}")

    return "\n\n".join(formatted)
