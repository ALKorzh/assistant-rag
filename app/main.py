import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.agent import app_graph
from app.services.rag_service import RAGService


UPLOAD_DIR = Path("data/raw")

app = FastAPI(
    title="Agentic RAG Assistant",
    description="Personal assistant with routing, RAG, and external tools.",
)


class Query(BaseModel):
    text: str = Field(..., min_length=1, description="User message")


class ChatResponse(BaseModel):
    answer: str


class UploadResponse(BaseModel):
    status: str


rag_service = RAGService()


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_filename = Path(file.filename or "uploaded_file").name
    temp_path = UPLOAD_DIR / safe_filename

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = rag_service.process_file(str(temp_path))
    return {"status": result}


@app.post("/chat", response_model=ChatResponse)
async def chat(query: Query):
    initial_state = {"messages": [HumanMessage(content=query.text)]}
    result = await app_graph.ainvoke(initial_state)
    final_message = result["messages"][-1].content
    return {"answer": str(final_message)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
