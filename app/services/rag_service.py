import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class RAGService:
    """Document indexing and semantic retrieval over Qdrant."""

    def __init__(self) -> None:
        settings = get_settings()
        self._search_limit = settings.rag_search_limit

        logger.info(
            "Initializing RAGService: qdrant=%s collection=%s embedding=%s",
            settings.qdrant_url,
            settings.qdrant_collection,
            settings.ollama_embedding_model,
        )

        self.embeddings = OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )
        self.client = QdrantClient(url=settings.qdrant_url)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=settings.qdrant_collection,
            embedding=self.embeddings,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def process_file(self, file_path: str) -> str:
        """Load, split, and index a text/PDF document."""
        path = Path(file_path)
        logger.info("Indexing file: %s", path)

        try:
            loader = self._get_loader(path)
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)

            if not chunks:
                logger.warning("No chunks produced for file %s", path)
                return f"Файл {path.name} не содержит текста для индексации."

            self.vector_store.add_documents(chunks)
            logger.info("Indexed %d chunks for %s", len(chunks), path.name)
            return f"Файл {path.name} успешно проиндексирован."
        except Exception:
            logger.exception("File indexing failed for %s", path)
            return f"Ошибка при обработке файла: {path.name}"

    def query(self, question: str, limit: int | None = None) -> str:
        """Return the most relevant document chunks for a user question."""
        top_k = limit or self._search_limit
        logger.info("RAG query: top_k=%d question_length=%d", top_k, len(question))

        try:
            docs = self.vector_store.similarity_search(question, k=top_k)
            if not docs:
                logger.info("RAG query returned no documents")
                return "Информация в локальных документах не найдена."

            logger.info("RAG query returned %d documents", len(docs))
            return "\n\n".join(doc.page_content for doc in docs)
        except Exception:
            logger.exception("RAG query failed")
            return "Ошибка при поиске в базе документов."

    @staticmethod
    def _get_loader(path: Path):
        if path.suffix.lower() == ".pdf":
            return PyPDFLoader(str(path))
        return TextLoader(str(path), encoding="utf-8")
