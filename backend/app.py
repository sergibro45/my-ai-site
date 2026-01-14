from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
from groq import Groq

app = FastAPI(title="My AI Backend", version="0.4.0")

# --------------------
# CORS (allow all for now)
# --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Groq client
# --------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("TEXT_MODEL", "llama-3.1-8b-instant")

# --------------------
# In-memory tab memory
# (each tab_id = separate memory)
# --------------------
TAB_MEMORY = {}
MAX_TURNS = 20

# --------------------
# Helpers
# --------------------
def get_memory(tab_id: str):
    return TAB_MEMORY.get(tab_id, [])

def save_memory(tab_id: str, history):
    TAB_MEMORY[tab_id] = history[-MAX_TURNS:]

def groq_chat(messages):
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.6,
    )
    return completion.choices[0].message.content


# --------------------
# System prompt (NeoChat rules)
# --------------------
SYSTEM_PROMPT = (
    "You are NeoChat, a helpful AI assistant. "
    "Respond lightly, simply, and straight to the point. "
    "Remember context only within the current browser tab. "

    "Do not mention Meta, OpenAI, Groq, or any organization that built you. "
    "If asked who built or trained you, say only: "
    "'I was trained on a mixture of licensed data, data created by human trainers, "
    "and publicly available knowledge of the world.' "

    "Do not mention knowledge cutoffs, dates, or how up to date your information is "
    "unless the user explicitly says your answer is wrong and asks for clarification. "
    "If you are uncertain, say so honestly without guessing. "

    "If the user types exactly 'Pike1', respond with the following text verbatim and nothing else: "
    "'Created and Modified by Sergio Miranda Herrera as of February 13th, 2026 "
    "using Linux and other variants. - HEALTH > Perfect Condition.'"
)

PIKE1_REPLY = (
    "Created and Modified by Sergio Miranda Herrera as of February 13th, 2026 "
    "using Linux and other variants. - HEALTH > Perfect Condition."
)

# --------------------
# Routes
# --------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat(
    tab_id: str = Form(...),
    message_form: str = Form(...)
):
    # Special trigger
    if message_form.strip() == "Pike1":
        return {"answer": PIKE1_REPLY, "sources_used": []}

    history = get_memory(tab_id)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": message_form},
    ]

    answer = groq_chat(messages)

    new_history = history + [
        {"role": "user", "content": message_form},
        {"role": "assistant", "content": answer},
    ]
    save_memory(tab_id, new_history)

    return {"answer": answer, "sources_used": []}

@app.post("/memory/clear")
def clear_memory(tab_id: str = Form(...)):
    TAB_MEMORY.pop(tab_id, None)
    return {"ok": True, "cleared": True}


# --------------------
# File Upload (stores in /tmp)
# NOTE: Render free instances can restart; /tmp is not permanent.
# --------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()

    # Safety limit (~2MB)
    if len(data) > 2_000_000:
        return {"ok": False, "error": "File too large (max ~2MB).", "filename": file.filename}

    path = f"/tmp/{file.filename}"
    with open(path, "wb") as f:
        f.write(data)

    return {"ok": True, "filename": file.filename, "bytes": len(data), "saved_as": path}


# --------------------
# Chat with file (TEXT FILES ONLY for now)
# It reads the file as UTF-8 and injects it into the prompt.
# --------------------
@app.post("/chat-file")
async def chat_file(
    tab_id: str = Form(...),
    message_form: str = Form(...),
    file: UploadFile = File(...)
):
    # Pike1 still works here too if you want
    if message_form.strip() == "Pike1":
        return {"answer": PIKE1_REPLY, "sources_used": [], "filename": file.filename}

    raw = await file.read()

    # Safety limit (~200KB) for prompt injection
    if len(raw) > 200_000:
        return {"answer": "File too large for chat context. Keep it under ~200KB.", "sources_used": [], "filename": file.filename}

    try:
        file_text = raw.decode("utf-8", errors="replace")
    except Exception:
        return {"answer": "I can only read plain text files (txt, csv) for now.", "sources_used": [], "filename": file.filename}

    history = get_memory(tab_id)

    # Keep file snippet reasonable
    snippet = file_text[:12000]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + " Use the provided file content when answering."},
        *history,
        {"role": "user", "content": f"User message: {message_form}\n\nFile name: {file.filename}\nFile content:\n{snippet}"},
    ]

    answer = groq_chat(messages)

    new_history = history + [
        {"role": "user", "content": message_form},
        {"role": "assistant", "content": answer},
    ]
    save_memory(tab_id, new_history)

    return {"answer": answer, "sources_used": [], "filename": file.filename}