# app.py
import os
import base64
from typing import List, Dict, Any, Optional

import requests
from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Optional text extractors (only used if installed)
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import docx  # python-docx
except Exception:
    docx = None

try:
    from PIL import Image
except Exception:
    Image = None


app = FastAPI(title="My AI Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Groq (OpenAI-compatible)
# -------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()

# If you want to try vision with Groq (if your chosen model supports it),
# set this env var in Render:
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", GROQ_MODEL).strip()

# Very small in-memory "session memory" (resets when service restarts)
DOC_STORE: List[Dict[str, Any]] = []  # each: {"name": str, "text": str}


def _require_key():
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY (set it in Render Environment).")


def groq_chat(messages: List[Dict[str, Any]], model: str) -> str:
    _require_key()
    url = f"{GROQ_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "stream": False,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Groq request failed: {e}")

    if r.status_code >= 400:
        # Keep it short but useful
        raise HTTPException(status_code=502, detail=f"Groq error {r.status_code}: {r.text[:300]}")
    data = r.json()
    return data["choices"][0]["message"]["content"]


def simple_rag_context(query: str, max_chars: int = 2500) -> str:
    """Tiny 'RAG': pick docs that share words with query (no heavy libs)."""
    if not DOC_STORE:
        return ""
    qwords = set(w.lower() for w in query.split() if len(w) > 3)
    scored = []
    for d in DOC_STORE:
        text = d.get("text", "")
        if not text:
            continue
        twords = set(w.lower() for w in text.split()[:2000])  # cap scan
        score = len(qwords & twords)
        if score > 0:
            scored.append((score, d["name"], text))
    scored.sort(reverse=True, key=lambda x: x[0])
    chunks = []
    total = 0
    for score, name, text in scored[:3]:
        snippet = text.strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "..."
        block = f"[Source: {name}]\n{snippet}\n"
        if total + len(block) > max_chars:
            break
        chunks.append(block)
        total += len(block)
    return "\n".join(chunks).strip()


def extract_text_from_upload(file: UploadFile, raw: bytes) -> str:
    name = (file.filename or "upload").lower()

    # PDF
    if name.endswith(".pdf") and PdfReader is not None:
        try:
            from io import BytesIO
            reader = PdfReader(BytesIO(raw))
            pages = []
            for p in reader.pages[:25]:
                pages.append(p.extract_text() or "")
            return "\n".join(pages).strip()
        except Exception:
            return ""

    # DOCX
    if name.endswith(".docx") and docx is not None:
        try:
            from io import BytesIO
            d = docx.Document(BytesIO(raw))
            return "\n".join(p.text for p in d.paragraphs).strip()
        except Exception:
            return ""

    # TXT
    if name.endswith(".txt"):
        try:
            return raw.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    # Images: no OCR (we store placeholder text)
    if any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]) and Image is not None:
        return f"(Image uploaded: {file.filename})"

    return ""


# -------------------------
# API
# -------------------------
@app.get("/health")
def health():
    return {"ok": True, "docs": len(DOC_STORE)}


class ChatJSON(BaseModel):
    message: str


@app.post("/chat")
def chat(
    # Support BOTH form (what your OpenAPI showed) and JSON (more convenient).
    message_form: Optional[str] = Form(default=None),
    body: Optional[ChatJSON] = Body(default=None),
):
    message = message_form or (body.message if body else None)
    if not message:
        raise HTTPException(status_code=422, detail="Missing message")

    context = simple_rag_context(message)
    system = (
        "You are a helpful assistant. "
        "If CONTEXT is provided, use it as supporting info. "
        "If you don't see the answer in context, answer normally."
    )

    user_content = message
    if context:
        user_content = f"CONTEXT:\n{context}\n\nUSER:\n{message}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    answer = groq_chat(messages, model=GROQ_MODEL)
    return {"answer": answer, "sources_used": []}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    text = extract_text_from_upload(file, raw)

    DOC_STORE.append(
        {
            "name": file.filename or "upload",
            "text": text or "",
        }
    )
    return {"ok": True, "filename": file.filename, "stored_chars": len(text or "")}


@app.post("/chat-image")
async def chat_image(message: str = Form(...), image: UploadFile = File(...)):
    raw = await image.read()
    b64 = base64.b64encode(raw).decode("utf-8")
    mime = image.content_type or "image/png"

    # This uses the OpenAI-style "image_url" content format.
    # It will ONLY work if your GROQ_VISION_MODEL supports images.
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Describe and answer based on the image."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        },
    ]

    try:
        answer = groq_chat(messages, model=GROQ_VISION_MODEL)
    except HTTPException as e:
        # If Groq/model doesn't support images, return a clear message
        raise HTTPException(
            status_code=502,
            detail=f"Vision request failed (your model may not support images). "
                   f"Set GROQ_VISION_MODEL to a vision-capable Groq model. Details: {e.detail}",
        )

    return {"answer": answer, "sources_used": []}

