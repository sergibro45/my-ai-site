import os
import base64
from io import BytesIO
from typing import List, Dict, Tuple, Optional

import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from pypdf import PdfReader
import docx
from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# In-memory doc store + TF-IDF index (low memory)
DOCS: List[Dict] = []  # {"source": str, "chunk": str}
_vectorizer = TfidfVectorizer(stop_words="english")
_matrix = None

def chunk_text(text: str, max_chars: int = 1200) -> List[str]:
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

def rebuild_index():
    global _matrix
    if not DOCS:
        _matrix = None
        return
    texts = [d["chunk"] for d in DOCS]
    _matrix = _vectorizer.fit_transform(texts)

def retrieve(query: str, k: int = 6) -> List[Tuple[str, str]]:
    if _matrix is None or not DOCS:
        return []
    qv = _vectorizer.transform([query])
    sims = cosine_similarity(qv, _matrix)[0]
    best = sims.argsort()[::-1][:k]
    return [(DOCS[i]["chunk"], DOCS[i]["source"]) for i in best]

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

    rebuild_index()
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
