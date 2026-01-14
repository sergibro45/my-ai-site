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

MODEL = os.getenv("TEXT_MODEL", "llama-3.1-8b-instant")

# --------------------
# In-memory tab memory
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
    history = get_memory(tab_id)

    messages = [
        {
            "role": "system",
            "content": (
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