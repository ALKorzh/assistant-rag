import os
import shutil

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.agent import app_graph
from langchain_core.messages import HumanMessage

from app.services.rag_service import RAGService

app = FastAPI(title="Agentic RAG Assistant")


class Query(BaseModel):
    text: str


rag_service = RAGService()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Сохраняем файл временно
    temp_path = f"data/raw/{file.filename}"
    os.makedirs("data/raw", exist_ok=True)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Индексируем
    result = rag_service.process_file(temp_path)
    return {"status": result}


@app.post("/chat")
async def chat(query: Query):
    # Начальное состояние
    initial_state = {
        "messages": [HumanMessage(content=query.text)]
    }

    # Запуск графа
    result = await app_graph.ainvoke(initial_state)

    # Берем последнее сообщение из истории (ответ модели)
    final_message = result["messages"][-1].content
    return {"answer": final_message}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)