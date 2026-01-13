import os
import base64
from io import BytesIO
from typing import Optional

import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

import chromadb
from sentence_transformers import SentenceTransformer

from pypdf import PdfReader
import docx
from PIL import Image

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
TEXT_MODEL = os.getenv("TEXT_MODEL", "qwen2.5:7b-instruct")
VISION_MODEL = os.getenv("VISION_MODEL", "moondream:v2")

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedder = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PATH)
col = client.get_or_create_collection(name="uploads")

def chunk_text(text: str, max_chars: int = 1200):
    text = text.replace("\r", "")
    parts, buf, size = [], [], 0
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if size + len(line) + 1 > max_chars and buf:
            parts.append("\n".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += len(line) + 1
    if buf:
        parts.append("\n".join(buf))
    return parts

def add_to_rag(text: str, source: str):
    chunks = chunk_text(text)
    if not chunks:
        return 0
    embs = embedder.encode(chunks).tolist()
    ids = [f"{source}-{i}" for i in range(len(chunks))]
    metas = [{"source": source} for _ in chunks]
    col.add(ids=ids, documents=chunks, embeddings=embs, metadatas=metas)
    return len(chunks)

def retrieve(query: str, k: int = 6):
    q_emb = embedder.encode([query]).tolist()[0]
    res = col.query(query_embeddings=[q_emb], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return list(zip(docs, metas))

def ollama_generate(model: str, prompt: str, images_b64: Optional[list] = None):
    payload = {"model": model, "prompt": prompt, "stream": False}
    if images_b64:
        payload["images"] = images_b64
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["response"]

def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    out = []
    for page in reader.pages:
        out.append(page.extract_text() or "")
    return "\n".join(out).strip()

def read_docx(file_bytes: bytes) -> str:
    tmp = BytesIO(file_bytes)
    d = docx.Document(tmp)
    return "\n".join(p.text for p in d.paragraphs).strip()

def read_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore").strip()

def image_to_b64(file_bytes: bytes) -> str:
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    name = (file.filename or "").lower()

    if name.endswith(".pdf"):
        text = read_pdf(data)
    elif name.endswith(".docx"):
        text = read_docx(data)
    else:
        text = read_txt(data)

    n = add_to_rag(text, source=file.filename or "upload")
    return {"indexed_chunks": n, "filename": file.filename}

@app.post("/chat")
async def chat(message: str = Form(...)):
    hits = retrieve(message, k=6)
    context = "\n\n".join([f"- ({m.get('source','upload')}) {d}" for d, m in hits]) if hits else "None"

    prompt = f"""You are a helpful assistant.
Use the CONTEXT to answer. If CONTEXT is None or irrelevant, answer normally.

CONTEXT:
{context}

USER:
{message}
"""
    answer = ollama_generate(TEXT_MODEL, prompt)
    return {"answer": answer, "sources_used": [m.get("source") for _, m in hits]}

@app.post("/chat-image")
async def chat_image(message: str = Form(...), image: UploadFile = File(...)):
    img_bytes = await image.read()
    img_b64 = image_to_b64(img_bytes)

    prompt = f"""You are a helpful assistant.
Answer the user's question about the image.

USER:
{message}
"""
    answer = ollama_generate(VISION_MODEL, prompt, images_b64=[img_b64])
    return {"answer": answer, "image_filename": image.filename}
