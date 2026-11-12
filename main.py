from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from rag_engine import RAGEngine

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")

app = FastAPI(title="Course RAG System")

# Initialize Engine
engine = RAGEngine(hf_api_token=HF_TOKEN)

@app.on_event("startup")
def startup_event():
    msg = engine.build_or_load_index()
    print(f"[RAG System Startup] {msg}")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/api/ask")
async def ask_question(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    response = engine.generate_answer(question=req.question, k=req.top_k)
    return response

# Serve UI Static Assets
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")