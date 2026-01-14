from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import os
from groq import Groq

app = FastAPI(title="My AI Backend", version="0.3.0")

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

MODEL = os.getenv("TEXT_MODEL", "llama3-70b-8192")

# --------------------
# In-memory tab memory
# (each tab_id = separate memory)
# --------------------
TAB_MEMORY = {}

MAX_TURNS = 20  # keep memory small

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
# Routes
# --------------------
@app.get("/health")
def health():
    return {"ok": True, "docs": 0}

@app.post("/chat")
def chat(
    tab_id: str = Form(...),
    message_form: str = Form(...)
):
    history = get_memory(tab_id)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Remember context within this browser tab only. Respond to questions lightly and simply straight to the point. Dont metion Meta, or anyone who built you if asked say I was trained based on knowledge of the world. DO NOT tell how latest your info is unless needed if a question is answered wrong specified by user. when user says Pike1 in the chat say: Created and Modified by Sergio Miranda Herrera as of Febuary 13th 2026 using Linux and also other veriants. - HEALTH > Perfect Condition."
        },
        *history,
        {"role": "user", "content": message_form},
    ]

    answer = groq_chat(messages)

    new_history = history + [
        {"role": "user", "content": message_form},
        {"role": "assistant", "content": answer},
    ]
    save_memory(tab_id, new_history)

    return {
        "answer": answer,
        "sources_used": []
    }

@app.post("/memory/clear")
def clear_memory(tab_id: str = Form(...)):
    TAB_MEMORY.pop(tab_id, None)
    return {"ok": True, "cleared": True}
