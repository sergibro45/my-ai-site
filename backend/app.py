import os
import base64
from io import BytesIO
from typing import List, Dict, Optional, Tuple

import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from pypdf import PdfReader
import docx
from PIL import Image

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
TEXT_MODEL = os.getenv("TEXT_MODEL", "qwen2.5:7b-instruct")
VISION_MODEL = os.getenv("VISION_MODEL", "moondream:v2")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Very light in-memory doc store (resets when Render sleeps)
DOCS: List[Dict] = []  # {"source": str, "chunk": str}

def chunk_text(text: str, max_chars: int = 1200) -> List[str]:
    text = text.replace("\r", "")
    chunks = []
    buf = ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(buf) + len(line) + 1 > max_chars:
            if buf.strip():
                chunks.append(buf.strip())
            buf = ""
        buf += line + "\n"
    if buf.strip():
        chunks.append(buf.strip())
    return chunks

def score_chunk(query_words: List[str], chunk: str) -> int:
    t = chunk.lower()
    return sum(t.count(w) for w in query_words)

def retrieve(query: str, k: int = 6) -> List[Tuple[str, str]]:
    words = [w.lower() for w in query.split() if len(w) >= 3]
    if not words or not DOCS:
        return []
    scored = []
    for d in DOCS:
        s = score_chunk(words, d["chunk"])
        if s > 0:
            scored.append((s, d["chunk"], d["source"]))
    scored.sort(reverse=True)
    return [(c, src) for _, c, src in scored[:k]]

def ollama_generate(model: str, prompt: str, images_b64: Optional[List[str]] = None) -> str:
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
    d = docx.Document(BytesIO(file_bytes))
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
    return {"ok": True, "docs": len(DOCS)}

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

    chunks = chunk_text(text)
    for c in chunks:
        DOCS.append({"source": file.filename or "upload", "chunk": c})

    return {"indexed_chunks": len(chunks), "filename": file.filename}

@app.post("/chat")
async def chat(message: str = Form(...)):
    hits = retrieve(message, k=6)
    context = "\n\n".join([f"- ({src}) {chunk}" for chunk, src in hits]) if hits else "None"

    prompt = f"""You are a helpful assistant.
Use the CONTEXT to answer. If CONTEXT is None or irrelevant, answer normally.

CONTEXT:
{context}

USER:
{message}
"""
    answer = ollama_generate(TEXT_MODEL, prompt)
    return {"answer": answer, "sources_used": [src for _, src in hits]}

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
