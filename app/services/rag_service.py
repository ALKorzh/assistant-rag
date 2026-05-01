import os
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Настройки
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "my_documents"


class RAGService:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        # Создаем клиента
        self.client = QdrantClient(url=QDRANT_URL)

        # Инициализируем хранилище (оно само создаст коллекцию)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding=self.embeddings,
        )

    def process_file(self, file_path: str):
        """Загружает файл, режет на куски и сохраняет в базу"""
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding='utf-8')

            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100
            )
            docs = text_splitter.split_documents(documents)

            # Добавляем документы
            self.vector_store.add_documents(docs)
            return f"Файл {os.path.basename(file_path)} успешно проиндексирован."
        except Exception as e:
            return f"Ошибка при обработке файла: {str(e)}"

    def query(self, question: str):
        """Поиск похожих кусков текста в базе"""
        try:
            # Выполняем поиск
            docs = self.vector_store.similarity_search(question, k=3)

            if not docs:
                return "Информация в локальных документах не найдена."

            context = "\n\n".join([doc.page_content for doc in docs])
            return context
        except Exception as e:
            return f"Ошибка при поиске: {str(e)}"