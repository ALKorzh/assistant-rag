import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "my_documents")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
DEFAULT_SEARCH_LIMIT = 3
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


class RAGService:
    """Document indexing and semantic retrieval over Qdrant."""

    def __init__(self) -> None:
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.client = QdrantClient(url=QDRANT_URL)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding=self.embeddings,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    def process_file(self, file_path: str) -> str:
        """Load, split, and index a text/PDF document."""
        path = Path(file_path)

        try:
            loader = self._get_loader(path)
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)

            if not chunks:
                return f"Файл {path.name} не содержит текста для индексации."

            self.vector_store.add_documents(chunks)
            return f"Файл {path.name} успешно проиндексирован."
        except Exception as exc:
            return f"Ошибка при обработке файла: {exc}"

    def query(self, question: str, limit: int = DEFAULT_SEARCH_LIMIT) -> str:
        """Return the most relevant document chunks for a user question."""
        try:
            docs = self.vector_store.similarity_search(question, k=limit)

            if not docs:
                return "Информация в локальных документах не найдена."

            return "\n\n".join(doc.page_content for doc in docs)
        except Exception as exc:
            return f"Ошибка при поиске: {exc}"

    @staticmethod
    def _get_loader(path: Path):
        if path.suffix.lower() == ".pdf":
            return PyPDFLoader(str(path))
        return TextLoader(str(path), encoding="utf-8")
