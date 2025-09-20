# /app/main.py
import os, io
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import openai
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from typing import List

# --- Config ---
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")  # optional, needed for search fallback
if not OPENAI_KEY:
    raise Exception("Set OPENAI_API_KEY in env")

openai.api_key = OPENAI_KEY

# Chroma persistent client (stores DB in ./chroma_db)
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
collection = client.get_or_create_collection(name="notes")

app = FastAPI(title="TutorBot (Level1+Level2)")

# --- helpers ---
def extract_text_from_pdf_bytes(b: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(b))
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception:
        return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

# --- ingest endpoint (upload notes) ---
@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    added = []
    for f in files:
        data = await f.read()
        text = ""
        if f.filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf_bytes(data)
        else:
            try:
                text = data.decode("utf-8")
            except:
                text = ""
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            emb = openai.Embedding.create(model="text-embedding-3-small", input=chunk)
            vector = emb["data"][0]["embedding"]
            doc_id = f"{f.filename}-{i}"
            collection.add(documents=[chunk], embeddings=[vector], metadatas=[{"source": f.filename}], ids=[doc_id])
        added.append({"file": f.filename, "chunks": len(chunks)})
    client.persist()
    return {"status": "ok", "added": added}

# --- query/chat endpoint ---
class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    question = query.question
    # 1) RAG: query Chroma for top-k
    try:
        res = collection.query(query_texts=[question], n_results=3, include=["documents", "metadatas"])
        docs = res.get("documents", [[]])[0]
    except Exception:
        docs = []

    context = "\n\n".join(docs) if docs else ""

    # 2) Fallback: web search if no context
    if not context and SERPAPI_KEY:
        try:
            from serpapi import GoogleSearch
            params = {"q": question, "engine": "google", "api_key": SERPAPI_KEY, "num": 3}
            search = GoogleSearch(params)
            result = search.get_dict()
            snippets = []
            for r in result.get("organic_results", [])[:3]:
                text_snip = r.get("snippet") or r.get("title") or ""
                snippets.append(text_snip)
            context = "\n\n".join(snippets)
        except Exception:
            context = ""

    # 3) Call LLM with context
    system_prompt = (
        "You are a helpful tutor. Use the provided context (if any) to answer the user's question. "
        "If the context doesn't answer it, use your knowledge. Provide explanation and a short practice problem when helpful."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
    ]
    resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, max_tokens=600)
    answer = resp["choices"][0]["message"]["content"]
    return {"answer": answer, "used_context": bool(context)}

# Health
@app.get("/")
def health():
    return {"status": "ok"}
