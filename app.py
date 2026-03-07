import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import os
import json
import uuid
import jwt as pyjwt
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import bcrypt as _bcrypt

from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.pgvector import PgVector2
from phi.llm.anthropic.claude import Claude
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder

load_dotenv()
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

# ── Config ────────────────────────────────────────────────
DB_URL = "postgresql+psycopg://ai:ai@localhost:5532/ai"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
METADATA_FILE = Path("chat_metadata.json")
USERS_FILE = Path("users.json")

SECRET_KEY = "pdf-assistant-jwt-secret-2024-change-in-prod"
ALGORITHM = "HS256"
TOKEN_EXPIRY_DAYS = 30

def hash_password(password: str) -> str:
    return _bcrypt.hashpw(password.encode("utf-8"), _bcrypt.gensalt()).decode("utf-8")

def verify_password(password: str, hashed: str) -> bool:
    return _bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

# ── Singleton embedder (avoid reloading model each request) ──
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformerEmbedder(dimensions=384)
    return _embedder

def get_storage():
    return PgAssistantStorage(table_name="pdf_assistant_web", db_url=DB_URL)

# ── User helpers ─────────────────────────────────────────
def load_users() -> dict:
    if USERS_FILE.exists():
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    return {}

def save_users(data: dict):
    USERS_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

# ── Metadata helpers ─────────────────────────────────────
def load_metadata() -> dict:
    if METADATA_FILE.exists():
        return json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    return {}

def save_metadata(data: dict):
    METADATA_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

# ── Auth helpers ─────────────────────────────────────────
def create_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=TOKEN_EXPIRY_DAYS)
    return pyjwt.encode({"sub": user_id, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(request: Request) -> dict:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = auth_header.split(" ")[1]
    try:
        payload = pyjwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        users = load_users()
        if user_id not in users:
            raise HTTPException(status_code=401, detail="User not found")
        u = users[user_id]
        return {"user_id": user_id, "name": u["name"], "email": u["email"]}
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ── FastAPI app ──────────────────────────────────────────
app = FastAPI(title="PDF Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Auth Routes ──────────────────────────────────────────

class SignUpRequest(BaseModel):
    name: str
    email: str
    password: str

class SignInRequest(BaseModel):
    email: str
    password: str

@app.post("/api/auth/signup")
async def signup(req: SignUpRequest):
    users = load_users()
    for uid, u in users.items():
        if u["email"].lower() == req.email.lower():
            raise HTTPException(status_code=400, detail="Email already registered")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")

    user_id = str(uuid.uuid4())
    users[user_id] = {
        "name": req.name.strip(),
        "email": req.email.lower().strip(),
        "password": hash_password(req.password),
        "created_at": datetime.now().isoformat(),
    }
    save_users(users)
    token = create_token(user_id)
    return {"token": token, "user": {"user_id": user_id, "name": req.name.strip(), "email": req.email.lower().strip()}}

@app.post("/api/auth/signin")
async def signin(req: SignInRequest):
    users = load_users()
    for uid, u in users.items():
        if u["email"].lower() == req.email.lower():
            if verify_password(req.password, u["password"]):
                token = create_token(uid)
                return {"token": token, "user": {"user_id": uid, "name": u["name"], "email": u["email"]}}
            raise HTTPException(status_code=401, detail="Invalid password")
    raise HTTPException(status_code=401, detail="No account found with this email")

@app.get("/api/auth/me")
async def get_me(user: dict = Depends(get_current_user)):
    return {"user": user}

# ── App Routes ───────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/index.html")

class ChatRequest(BaseModel):
    message: str

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """Upload a PDF and create a new chat session."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    chat_id = str(uuid.uuid4())
    file_id = chat_id[:8]
    safe_name = file.filename.replace(" ", "_")
    filepath = UPLOAD_DIR / f"{file_id}_{safe_name}"

    content = await file.read()
    filepath.write_bytes(content)

    collection = f"pdf_{file_id}"
    embedder = get_embedder()
    kb = PDFKnowledgeBase(
        path=str(filepath),
        vector_db=PgVector2(collection=collection, db_url=DB_URL, embedder=embedder),
    )
    kb.load()

    metadata = load_metadata()
    metadata[chat_id] = {
        "user_id": user["user_id"],
        "pdf_name": file.filename,
        "pdf_path": str(filepath),
        "collection": collection,
        "created_at": datetime.now().isoformat(),
        "title": file.filename.rsplit(".", 1)[0],
    }
    save_metadata(metadata)

    return {
        "chat_id": chat_id,
        "pdf_name": file.filename,
        "title": metadata[chat_id]["title"],
        "created_at": metadata[chat_id]["created_at"],
    }

@app.get("/api/chats")
async def list_chats(user: dict = Depends(get_current_user)):
    """List all chat sessions for the authenticated user."""
    metadata = load_metadata()
    chats = []
    for cid, info in metadata.items():
        if info.get("user_id") == user["user_id"]:
            chats.append({
                "chat_id": cid,
                "pdf_name": info["pdf_name"],
                "title": info.get("title", info["pdf_name"]),
                "created_at": info["created_at"],
            })
    chats.sort(key=lambda x: x["created_at"], reverse=True)
    return {"chats": chats}

@app.get("/api/chats/{chat_id}/messages")
async def get_messages(chat_id: str, user: dict = Depends(get_current_user)):
    """Return stored messages for a chat."""
    metadata = load_metadata()
    if chat_id not in metadata:
        raise HTTPException(status_code=404, detail="Chat not found")
    if metadata[chat_id].get("user_id") != user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    storage = get_storage()
    rows = storage.get_all_runs(user_id=user["user_id"])
    for row in rows:
        if row.run_id == chat_id and row.memory and "chat_history" in row.memory:
            messages = []
            for msg in row.memory["chat_history"]:
                if msg.get("role") in ("user", "assistant") and msg.get("content"):
                    messages.append({"role": msg["role"], "content": msg["content"]})
            return {"messages": messages}
    return {"messages": []}

@app.post("/api/chat/{chat_id}")
async def chat(chat_id: str, request_body: ChatRequest, user: dict = Depends(get_current_user)):
    """Send a message and stream the response via SSE."""
    metadata = load_metadata()
    if chat_id not in metadata:
        raise HTTPException(status_code=404, detail="Chat not found")
    if metadata[chat_id].get("user_id") != user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    info = metadata[chat_id]
    embedder = get_embedder()

    kb = PDFKnowledgeBase(
        path=info["pdf_path"],
        vector_db=PgVector2(collection=info["collection"], db_url=DB_URL, embedder=embedder),
    )

    storage = get_storage()
    assistant = Assistant(
        run_id=chat_id,
        user_id=user["user_id"],
        llm=Claude(model="claude-sonnet-4-20250514"),
        knowledge_base=kb,
        storage=storage,
        show_tool_calls=False,
        search_knowledge=True,
        read_chat_history=True,
        markdown=True,
    )

    def generate():
        try:
            for chunk in assistant.run(request_body.message, stream=True):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str, user: dict = Depends(get_current_user)):
    """Delete a chat session and its uploaded PDF."""
    metadata = load_metadata()
    if chat_id not in metadata:
        raise HTTPException(status_code=404, detail="Chat not found")
    if metadata[chat_id].get("user_id") != user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    info = metadata[chat_id]
    pdf_path = Path(info["pdf_path"])
    if pdf_path.exists():
        pdf_path.unlink()

    del metadata[chat_id]
    save_metadata(metadata)
    return {"status": "deleted"}

@app.patch("/api/chats/{chat_id}")
async def rename_chat(chat_id: str, body: dict, user: dict = Depends(get_current_user)):
    """Rename a chat session."""
    metadata = load_metadata()
    if chat_id not in metadata:
        raise HTTPException(status_code=404, detail="Chat not found")
    if metadata[chat_id].get("user_id") != user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    if "title" in body:
        metadata[chat_id]["title"] = body["title"]
        save_metadata(metadata)
    return {"status": "ok"}

# ── Entry point ──────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
