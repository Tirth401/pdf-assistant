import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import os
import json
import uuid
import logging
import traceback
import jwt as pyjwt
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import bcrypt as _bcrypt

from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.pgvector import PgVector2
from phi.llm.anthropic.claude import Claude
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Set API key from env
_api_key = os.getenv("ANTHROPIC_API_KEY")
if _api_key:
    os.environ["ANTHROPIC_API_KEY"] = _api_key

# ── Config ────────────────────────────────────────────────
_raw_db_url = os.getenv("DATABASE_URL", "postgresql+psycopg://ai:ai@localhost:5532/ai")
if _raw_db_url.startswith("postgresql://"):
    DB_URL = _raw_db_url.replace("postgresql://", "postgresql+psycopg://", 1)
elif _raw_db_url.startswith("postgres://"):
    DB_URL = _raw_db_url.replace("postgres://", "postgresql+psycopg://", 1)
else:
    DB_URL = _raw_db_url

# SQLAlchemy needs postgresql+psycopg2:// for psycopg2 or postgresql+psycopg:// for psycopg3
# We'll use psycopg2 for SQLAlchemy ORM tables since it's more widely compatible
_sa_db_url = DB_URL.replace("postgresql+psycopg://", "postgresql://", 1)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

SECRET_KEY = os.getenv("SECRET_KEY", "pdf-assistant-jwt-secret-2024-change-in-prod")
ALGORITHM = "HS256"
TOKEN_EXPIRY_DAYS = 30

# ── SQLAlchemy setup for users & chat metadata ────────────
Base = declarative_base()

class UserRow(Base):
    __tablename__ = "app_users"
    user_id    = Column(String, primary_key=True)
    name       = Column(String, nullable=False)
    email      = Column(String, unique=True, nullable=False)
    password   = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatMetadataRow(Base):
    __tablename__ = "chat_metadata"
    chat_id    = Column(String, primary_key=True)
    user_id    = Column(String, nullable=False, index=True)
    pdf_name   = Column(String, nullable=False)
    pdf_path   = Column(String, nullable=False)
    collection = Column(String, nullable=False)
    title      = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

_engine = create_engine(_sa_db_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=_engine)

def hash_password(password: str) -> str:
    return _bcrypt.hashpw(password.encode("utf-8"), _bcrypt.gensalt()).decode("utf-8")

def verify_password(password: str, hashed: str) -> bool:
    return _bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

# ── Singleton embedder ────────────────────────────────────
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformerEmbedder(dimensions=384)
    return _embedder

def get_storage():
    return PgAssistantStorage(table_name="pdf_assistant_web", db_url=DB_URL)

# ── DB-backed user helpers ────────────────────────────────
def db_get_user_by_email(email: str):
    with SessionLocal() as s:
        return s.query(UserRow).filter(UserRow.email == email.lower().strip()).first()

def db_get_user(user_id: str):
    with SessionLocal() as s:
        return s.query(UserRow).filter(UserRow.user_id == user_id).first()

def db_create_user(user_id, name, email, password_hash):
    with SessionLocal() as s:
        u = UserRow(user_id=user_id, name=name, email=email.lower().strip(), password=password_hash)
        s.add(u)
        s.commit()

# ── DB-backed metadata helpers ────────────────────────────
def db_save_chat_meta(chat_id, user_id, pdf_name, pdf_path, collection, title):
    with SessionLocal() as s:
        row = ChatMetadataRow(
            chat_id=chat_id, user_id=user_id, pdf_name=pdf_name,
            pdf_path=pdf_path, collection=collection, title=title,
            created_at=datetime.utcnow(),
        )
        s.add(row)
        s.commit()

def db_get_chat_meta(chat_id):
    with SessionLocal() as s:
        row = s.query(ChatMetadataRow).filter(ChatMetadataRow.chat_id == chat_id).first()
        if row:
            return {"chat_id": row.chat_id, "user_id": row.user_id, "pdf_name": row.pdf_name,
                    "pdf_path": row.pdf_path, "collection": row.collection, "title": row.title,
                    "created_at": row.created_at.isoformat() if row.created_at else ""}
        return None

def db_list_chats(user_id):
    with SessionLocal() as s:
        rows = s.query(ChatMetadataRow).filter(ChatMetadataRow.user_id == user_id)\
                .order_by(ChatMetadataRow.created_at.desc()).all()
        return [{"chat_id": r.chat_id, "pdf_name": r.pdf_name,
                 "title": r.title, "created_at": r.created_at.isoformat() if r.created_at else ""} for r in rows]

def db_delete_chat(chat_id):
    with SessionLocal() as s:
        s.query(ChatMetadataRow).filter(ChatMetadataRow.chat_id == chat_id).delete()
        s.commit()

def db_rename_chat(chat_id, new_title):
    with SessionLocal() as s:
        row = s.query(ChatMetadataRow).filter(ChatMetadataRow.chat_id == chat_id).first()
        if row:
            row.title = new_title
            s.commit()

# ── Auth helpers ──────────────────────────────────────────
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
        u = db_get_user(user_id)
        if not u:
            raise HTTPException(status_code=401, detail="User not found")
        return {"user_id": u.user_id, "name": u.name, "email": u.email}
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

# ── Startup: create tables + pgvector extension ──────────
@app.on_event("startup")
async def startup_event():
    logger.info(f"DB_URL (phi): {DB_URL[:40]}...")
    logger.info(f"DB_URL (sa):  {_sa_db_url[:40]}...")
    try:
        import psycopg
        raw_url = DB_URL.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(raw_url) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
            logger.info("pgvector extension ready")
    except Exception as e:
        logger.error(f"Could not create pgvector extension: {e}")

    # Create app tables (users, chat_metadata)
    try:
        Base.metadata.create_all(_engine)
        logger.info("App tables (app_users, chat_metadata) ready")
    except Exception as e:
        logger.error(f"Could not create app tables: {e}")

# ── Health check ─────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    checks = {"status": "ok", "db_url_set": bool(os.getenv("DATABASE_URL")),
              "api_key_set": bool(os.getenv("ANTHROPIC_API_KEY"))}
    try:
        import psycopg
        raw_url = DB_URL.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(raw_url) as conn:
            conn.execute("SELECT 1")
            checks["db_connection"] = "ok"
            result = conn.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'").fetchone()
            checks["pgvector"] = "installed" if result else "missing"
    except Exception as e:
        checks["db_connection"] = f"error: {str(e)}"
        checks["status"] = "degraded"
    return checks

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
    existing = db_get_user_by_email(req.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")

    user_id = str(uuid.uuid4())
    db_create_user(user_id, req.name.strip(), req.email, hash_password(req.password))
    token = create_token(user_id)
    return {"token": token, "user": {"user_id": user_id, "name": req.name.strip(), "email": req.email.lower().strip()}}

@app.post("/api/auth/signin")
async def signin(req: SignInRequest):
    u = db_get_user_by_email(req.email)
    if not u:
        raise HTTPException(status_code=401, detail="No account found with this email")
    if not verify_password(req.password, u.password):
        raise HTTPException(status_code=401, detail="Invalid password")
    token = create_token(u.user_id)
    return {"token": token, "user": {"user_id": u.user_id, "name": u.name, "email": u.email}}

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
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        chat_id = str(uuid.uuid4())
        file_id = chat_id[:8]
        safe_name = file.filename.replace(" ", "_")
        filepath = UPLOAD_DIR / f"{file_id}_{safe_name}"

        content = await file.read()
        filepath.write_bytes(content)
        logger.info(f"Saved PDF: {filepath} ({len(content)} bytes)")

        collection = f"pdf_{file_id}"
        embedder = get_embedder()
        logger.info(f"Creating knowledge base for collection: {collection}")
        kb = PDFKnowledgeBase(
            path=str(filepath),
            vector_db=PgVector2(collection=collection, db_url=DB_URL, embedder=embedder),
        )
        logger.info("Loading knowledge base (embedding + storing)...")
        kb.load()
        logger.info("Knowledge base loaded successfully")

        title = file.filename.rsplit(".", 1)[0]
        db_save_chat_meta(chat_id, user["user_id"], file.filename, str(filepath), collection, title)

        return {
            "chat_id": chat_id,
            "pdf_name": file.filename,
            "title": title,
            "created_at": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"detail": f"Upload processing error: {str(e)}"})

@app.get("/api/chats")
async def list_chats(user: dict = Depends(get_current_user)):
    chats = db_list_chats(user["user_id"])
    return {"chats": chats}

@app.get("/api/chats/{chat_id}/messages")
async def get_messages(chat_id: str, user: dict = Depends(get_current_user)):
    meta = db_get_chat_meta(chat_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Chat not found")
    if meta["user_id"] != user["user_id"]:
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
    meta = db_get_chat_meta(chat_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Chat not found")
    if meta["user_id"] != user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    embedder = get_embedder()
    kb = PDFKnowledgeBase(
        path=meta["pdf_path"],
        vector_db=PgVector2(collection=meta["collection"], db_url=DB_URL, embedder=embedder),
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
        add_references_to_prompt=True,
        markdown=True,
        description=f"You are a helpful PDF assistant. Today's date is {datetime.now().strftime('%B %d, %Y')}. You have access to the content of '{meta['pdf_name']}'. Always use the document content to answer questions. If the user asks about the document, refer to the actual content from the knowledge base.",
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
    meta = db_get_chat_meta(chat_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Chat not found")
    if meta["user_id"] != user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    pdf_path = Path(meta["pdf_path"])
    if pdf_path.exists():
        pdf_path.unlink()

    db_delete_chat(chat_id)
    return {"status": "deleted"}

@app.patch("/api/chats/{chat_id}")
async def rename_chat(chat_id: str, body: dict, user: dict = Depends(get_current_user)):
    meta = db_get_chat_meta(chat_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Chat not found")
    if meta["user_id"] != user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    if "title" in body:
        db_rename_chat(chat_id, body["title"])
    return {"status": "ok"}

# ── Entry point ──────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8501"))
    uvicorn.run(app, host="0.0.0.0", port=port)
