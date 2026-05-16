import logging
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})

# Matches file.ext referenced in prose (Russian/English filenames).
_HINT_FILE_EXT_PATTERN = re.compile(r"\b([A-Za-z0-9_-]+\.(?:pdf|txt|md))\b", re.IGNORECASE)
# после «файле», «файла», «документе» и т.п. без расширения
_HINT_AFTER_FILE_LABEL_PATTERN = re.compile(
    r"(?:файл\w*|документ\w*|конспект\w*|в\s+файле|по\s+файлу)\s+[«\"]?"
    r"([A-Za-z0-9_.-]{2,})\b",
    re.IGNORECASE,
)


def hints_from_question(question: str) -> list[str]:
    """Extract basename / stems the user explicitly names — used to bias retrieval."""
    if not question or not question.strip():
        return []
    seen: dict[str, None] = {}
    out: list[str] = []

    def push(token: str) -> None:
        t = token.strip()
        if len(t) < 2:
            return
        key = t.lower()
        if key not in seen:
            seen[key] = None
            out.append(t)

    for match in _HINT_FILE_EXT_PATTERN.finditer(question):
        name = match.group(1).strip()
        push(name)
        push(Path(name).stem)

    for match in _HINT_AFTER_FILE_LABEL_PATTERN.finditer(question):
        fragment = match.group(1).strip().rstrip(".,!?;:»«\"")
        if not fragment:
            continue
        if "." in fragment and not fragment.lower().endswith((".pdf", ".txt", ".md")):
            continue
        lower = fragment.lower()
        trivial = frozenset(
            {"который", "какой", "этот", "тот", "this", "the", "какое", "какая"}
        )
        if lower not in trivial:
            push(fragment)
            push(Path(fragment).stem)
    return out


def _metadata_basename(metadata: dict) -> str:
    fn = metadata.get("filename") or ""
    if isinstance(fn, str) and fn.strip():
        return Path(fn.strip()).name
    src = metadata.get("source") or ""
    if isinstance(src, str) and src.strip():
        return Path(src.strip().replace("\\", "/")).name
    return ""


def document_matches_hints(doc: Document, hints: list[str]) -> bool:
    """True when chunk metadata aligns with user-named file stems or names."""
    if not hints:
        return False
    base_name = _metadata_basename(doc.metadata).lower()
    stem = Path(base_name).stem.lower() if base_name else ""
    source = ""
    raw_src = doc.metadata.get("source")
    if isinstance(raw_src, str):
        source = raw_src.lower()

    for h in hints:
        h_strip = str(h).strip()
        if not h_strip:
            continue
        h_low = h_strip.lower()
        h_stem = Path(h_strip).stem.lower()
        if h_low == base_name or h_stem == stem:
            return True
        # substring: path often …/raw/e2e_upload_verify.txt
        if len(h_strip) >= 4 and (
            h_low in base_name
            or h_stem in base_name
            or h_low in source
            or h_stem in source
        ):
            return True
    return False


def _uniq_documents_ordered(docs: list[Document]) -> list[Document]:
    """Remove duplicates while preserving similarity rank order."""
    seen: set[str] = set()
    uniq: list[Document] = []
    for doc in docs:
        key = ""
        rid = doc.metadata.get("_id")
        if rid is not None:
            key = f"id:{rid}"
        else:
            base = doc.page_content.strip()[:200]
            bn = _metadata_basename(doc.metadata)
            key = f"body:{bn}:{base}"
        if key not in seen:
            seen.add(key)
            uniq.append(doc)
    return uniq


def _ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """Create the Qdrant collection if missing (fresh volume / first deploy)."""
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        return
    logger.info(
        "Creating Qdrant collection %s vector_size=%d distance=COSINE",
        collection_name,
        vector_size,
    )
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


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
        _ensure_collection(
            self.client,
            settings.qdrant_collection,
            settings.qdrant_vector_size,
        )
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
        """Load, split, and index a text/PDF/image document."""
        path = Path(file_path)
        logger.info("Indexing file: %s", path)

        try:
            loader = self._get_loader(path)
            documents = loader.load()
            canonical_source = str(path.resolve())
            for doc in documents:
                meta = dict(doc.metadata or {})
                meta.setdefault("source", canonical_source)
                meta["filename"] = path.name
                doc.metadata = meta
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
        hints = hints_from_question(question)
        fetch_k = top_k
        if hints:
            fetch_k = min(48, max(top_k * 8, max(24, top_k * 4)))

        logger.info(
            "RAG query: top_k=%d fetch_k=%d hints=%s question_length=%d",
            top_k,
            fetch_k,
            hints if hints else "—",
            len(question),
        )

        try:
            raw: list[Document] = []
            raw.extend(self.vector_store.similarity_search(question, k=fetch_k))
            if hints:
                booster_k = max(top_k * 3, min(24, fetch_k))
                raw.extend(self.vector_store.similarity_search(" ".join(hints), k=booster_k))

            ordered = _uniq_documents_ordered(raw)
            matched = [d for d in ordered if document_matches_hints(d, hints)]
            unmatched = [d for d in ordered if not document_matches_hints(d, hints)]

            merged: list[Document] = matched + unmatched
            merged = merged[:top_k] if merged else ordered[:top_k]

            if not merged:
                logger.info("RAG query returned no documents")
                return "Информация в локальных документах не найдена."

            hint_hits = any(document_matches_hints(d, hints) for d in merged) if hints else None
            logger.info("RAG query yielded %d chunks (hint_overlap=%s)", len(merged), hint_hits)

            def _label(doc: Document) -> str:
                label = doc.metadata.get("filename") if isinstance(doc.metadata.get("filename"), str) else ""
                if not label or not label.strip():
                    bn = _metadata_basename(doc.metadata)
                    label = bn or "(неизвестный файл)"
                return label.strip()

            return "\n\n".join(f"[Файл: {_label(doc)}]\n{doc.page_content}" for doc in merged)
        except Exception:
            logger.exception("RAG query failed")
            return "Ошибка при поиске в базе документов."

    @staticmethod
    def _get_loader(path: Path):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return PyPDFLoader(str(path))
        if suffix in _IMAGE_SUFFIXES:
            return _ImageOCRLoader(path)
        return TextLoader(str(path), encoding="utf-8")


class _ImageOCRLoader:
    """Minimal loader: OCR image to text for indexing."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> list[Document]:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            logger.warning("pytesseract/PIL not installed; skipping OCR for %s", self._path.name)
            return [
                Document(
                    page_content="",
                    metadata={"source": str(self._path)},
                )
            ]

        try:
            text = pytesseract.image_to_string(Image.open(self._path), lang="rus+eng")
        except Exception:
            logger.exception("OCR failed for %s", self._path)
            text = ""

        cleaned = (text or "").strip()
        body = cleaned if cleaned else "[Изображение без распознаваемого текста]"
        return [Document(page_content=body, metadata={"source": str(self._path)})]
