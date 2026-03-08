import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import asyncio
import base64
import json
import logging
import os
import re
import traceback
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import anthropic
import bcrypt as _bcrypt
import fitz as pymupdf
import jwt as pyjwt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pageindex import PageIndexClient
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ── Config ────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")

VISION_PAGE_THRESHOLD = 15
VISION_DPI = 150
VISION_MODEL = "claude-sonnet-4-20250514"

_raw_db_url = os.getenv("DATABASE_URL", "postgresql+psycopg2://ai:ai@localhost:5532/ai")
if _raw_db_url.startswith("postgresql+psycopg://"):
    DB_URL = _raw_db_url.replace("postgresql+psycopg://", "postgresql+psycopg2://", 1)
elif _raw_db_url.startswith("postgres://"):
    DB_URL = _raw_db_url.replace("postgres://", "postgresql+psycopg2://", 1)
elif _raw_db_url.startswith("postgresql://") and "+psycopg2" not in _raw_db_url:
    DB_URL = _raw_db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
else:
    DB_URL = _raw_db_url

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

SECRET_KEY = os.getenv("SECRET_KEY", "pdf-assistant-jwt-secret-2024-change-in-prod")
ALGORITHM = "HS256"
TOKEN_EXPIRY_DAYS = 30

MAX_POLL_SECONDS = 180
POLL_INTERVAL = 2

# ── SQLAlchemy setup ──────────────────────────────────────
Base = declarative_base()


class UserRow(Base):
    __tablename__ = "app_users"
    user_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatMetadataRow(Base):
    __tablename__ = "chat_metadata_v2"
    chat_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    pdf_name = Column(String, nullable=False)
    pdf_path = Column(String, nullable=False)
    mode = Column(String, nullable=False, default="vision")
    doc_id = Column(String, nullable=True)
    page_count = Column(Integer, nullable=False, default=1)
    title = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatMessageRow(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


_engine = create_engine(DB_URL, pool_pre_ping=True, connect_args={"connect_timeout": 10})
SessionLocal = sessionmaker(bind=_engine)


def hash_password(password: str) -> str:
    return _bcrypt.hashpw(password.encode("utf-8"), _bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return _bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


# ── API clients ───────────────────────────────────────────
_pi_client = None
_anthropic_client = None


def get_pi_client() -> PageIndexClient:
    global _pi_client
    if _pi_client is None:
        if not PAGEINDEX_API_KEY:
            raise HTTPException(status_code=500, detail="PAGEINDEX_API_KEY not configured")
        _pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    return _pi_client


def get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        if not ANTHROPIC_API_KEY:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")
        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


# ── PDF → images helper ──────────────────────────────────
def pdf_to_page_images(pdf_path: str, output_dir: Path, file_id: str, dpi: int = VISION_DPI) -> int:
    """Convert each page of a PDF to a JPEG image. Returns page count."""
    doc = pymupdf.open(pdf_path)
    page_count = doc.page_count
    for i in range(page_count):
        pix = doc[i].get_pixmap(dpi=dpi, alpha=False)
        img_bytes = pix.tobytes("jpeg")
        img_path = output_dir / f"{file_id}_page_{i}.jpg"
        img_path.write_bytes(img_bytes)
    doc.close()
    return page_count


def load_page_images_b64(pdf_path: str, file_id: str, page_count: int) -> list[dict]:
    """Load previously saved page images as base64-encoded content blocks."""
    output_dir = Path(pdf_path).parent
    blocks = []
    for i in range(page_count):
        img_path = output_dir / f"{file_id}_page_{i}.jpg"
        if not img_path.exists():
            continue
        b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
        blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            }
        )
    return blocks


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


# ── DB-backed chat metadata helpers ───────────────────────
def db_save_chat_meta(chat_id, user_id, pdf_name, pdf_path, mode, doc_id, page_count, title):
    with SessionLocal() as s:
        row = ChatMetadataRow(
            chat_id=chat_id,
            user_id=user_id,
            pdf_name=pdf_name,
            pdf_path=pdf_path,
            mode=mode,
            doc_id=doc_id or "",
            page_count=page_count,
            title=title,
            created_at=datetime.utcnow(),
        )
        s.add(row)
        s.commit()


def db_get_chat_meta(chat_id):
    with SessionLocal() as s:
        row = s.query(ChatMetadataRow).filter(ChatMetadataRow.chat_id == chat_id).first()
        if row:
            return {
                "chat_id": row.chat_id,
                "user_id": row.user_id,
                "pdf_name": row.pdf_name,
                "pdf_path": row.pdf_path,
                "mode": row.mode,
                "doc_id": row.doc_id,
                "page_count": row.page_count,
                "title": row.title,
                "created_at": row.created_at.isoformat() if row.created_at else "",
            }
        return None


def db_list_chats(user_id):
    with SessionLocal() as s:
        rows = (
            s.query(ChatMetadataRow)
            .filter(ChatMetadataRow.user_id == user_id)
            .order_by(ChatMetadataRow.created_at.desc())
            .all()
        )
        return [
            {
                "chat_id": r.chat_id,
                "pdf_name": r.pdf_name,
                "title": r.title,
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in rows
        ]


def db_delete_chat(chat_id):
    with SessionLocal() as s:
        s.query(ChatMessageRow).filter(ChatMessageRow.chat_id == chat_id).delete()
        s.query(ChatMetadataRow).filter(ChatMetadataRow.chat_id == chat_id).delete()
        s.commit()


def db_rename_chat(chat_id, new_title):
    with SessionLocal() as s:
        row = s.query(ChatMetadataRow).filter(ChatMetadataRow.chat_id == chat_id).first()
        if row:
            row.title = new_title
            s.commit()


# ── DB-backed message helpers ─────────────────────────────
def db_save_message(chat_id, role, content):
    with SessionLocal() as s:
        msg = ChatMessageRow(
            chat_id=chat_id,
            role=role,
            content=content,
            created_at=datetime.utcnow(),
        )
        s.add(msg)
        s.commit()


def db_get_messages(chat_id, limit=50):
    with SessionLocal() as s:
        rows = (
            s.query(ChatMessageRow)
            .filter(ChatMessageRow.chat_id == chat_id)
            .order_by(ChatMessageRow.created_at.asc())
            .limit(limit)
            .all()
        )
        return [{"role": r.role, "content": r.content} for r in rows]


# ── Auth helpers ──────────────────────────────────────────
def create_token(user_id: str) -> str:
    expire = datetime.now(UTC) + timedelta(days=TOKEN_EXPIRY_DAYS)
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
        raise HTTPException(status_code=401, detail="Token expired") from None
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token") from None


# ── FastAPI app ──────────────────────────────────────────
app = FastAPI(title="PDF Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup_event():
    logger.info(f"DB_URL: {DB_URL[:40]}...")
    try:
        Base.metadata.create_all(_engine)
        logger.info("App tables ready")
    except Exception as e:
        logger.error(f"Could not create app tables: {e}")


# ── Health check ─────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    checks = {
        "status": "ok",
        "db_url_set": bool(os.getenv("DATABASE_URL")),
        "anthropic_key_set": bool(ANTHROPIC_API_KEY),
        "pageindex_key_set": bool(PAGEINDEX_API_KEY),
        "vision_threshold": VISION_PAGE_THRESHOLD,
    }
    try:
        import psycopg2

        raw_url = DB_URL.replace("postgresql+psycopg2://", "postgresql://")
        conn = psycopg2.connect(raw_url, connect_timeout=5)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        checks["db_connection"] = "ok"
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

        doc = pymupdf.open(str(filepath))
        page_count = doc.page_count
        doc.close()
        logger.info(f"PDF has {page_count} pages")

        if page_count <= VISION_PAGE_THRESHOLD:
            mode = "vision"
            logger.info(f"Using VISION path ({page_count} pages <= {VISION_PAGE_THRESHOLD})")
            actual_pages = await asyncio.to_thread(pdf_to_page_images, str(filepath), UPLOAD_DIR, file_id)
            logger.info(f"Converted {actual_pages} pages to images")
            doc_id = ""
        else:
            mode = "pageindex"
            logger.info(f"Using PAGEINDEX path ({page_count} pages > {VISION_PAGE_THRESHOLD})")
            pi_client = get_pi_client()
            result = pi_client.submit_document(str(filepath))
            doc_id = result["doc_id"]
            logger.info(f"Document submitted, doc_id: {doc_id}")

            elapsed = 0
            while elapsed < MAX_POLL_SECONDS:
                doc_info = pi_client.get_document(doc_id)
                status = doc_info.get("status", "unknown")
                logger.info(f"Document status: {status} (elapsed: {elapsed}s)")
                if status == "completed":
                    break
                if status == "failed":
                    raise HTTPException(status_code=500, detail="PageIndex document processing failed")
                await asyncio.sleep(POLL_INTERVAL)
                elapsed += POLL_INTERVAL
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Document processing timed out after {MAX_POLL_SECONDS}s",
                )
            logger.info("Document processed by PageIndex successfully")

        title = file.filename.rsplit(".", 1)[0]
        db_save_chat_meta(chat_id, user["user_id"], file.filename, str(filepath), mode, doc_id, page_count, title)

        return {
            "chat_id": chat_id,
            "pdf_name": file.filename,
            "title": title,
            "mode": mode,
            "page_count": page_count,
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

    messages = db_get_messages(chat_id)
    return {"messages": messages}


# ── Vision chat: stream via Anthropic SDK ─────────────────
def _vision_generate(chat_id: str, meta: dict, user_message: str, history: list):
    file_id = meta["chat_id"][:8]
    image_blocks = load_page_images_b64(meta["pdf_path"], file_id, meta["page_count"])

    if not image_blocks:
        yield f"data: {json.dumps({'error': 'Could not load PDF page images'})}\n\n"
        return

    system_prompt = (
        f"You are a helpful PDF assistant. The user has uploaded '{meta['pdf_name']}' "
        f"({meta['page_count']} pages). The page images are provided below. "
        "Answer questions based ONLY on what you can see in these pages. "
        "Be precise with names, dates, numbers, and spelling — read them exactly as shown. "
        "If something is unclear in the image, say so rather than guessing."
    )

    messages = []
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})

    user_content = list(image_blocks)
    user_content.append({"type": "text", "text": user_message})
    messages.append({"role": "user", "content": user_content})

    client = get_anthropic_client()
    full_response = ""
    try:
        with client.messages.stream(
            model=VISION_MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                full_response += text
                yield f"data: {json.dumps({'content': text})}\n\n"

        db_save_message(chat_id, "assistant", full_response)
        yield f"data: {json.dumps({'done': True})}\n\n"
    except Exception as e:
        logger.error(f"Vision chat error: {traceback.format_exc()}")
        if full_response:
            db_save_message(chat_id, "assistant", full_response)
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# ── PageIndex metadata filter ─────────────────────────────
_PI_META_RE = re.compile(r'\{[^{}]*"doc_name"\s*:\s*"[^"]*"[^{}]*\}')


def _filter_pageindex_stream(raw_chunks):
    """Strip inline retrieval metadata from PageIndex streaming chunks."""
    buf = ""
    for chunk in raw_chunks:
        buf += chunk
        open_idx = buf.rfind("{")
        if open_idx != -1 and "}" not in buf[open_idx:]:
            safe = buf[:open_idx]
            if safe:
                yield _PI_META_RE.sub("", safe)
            buf = buf[open_idx:]
            continue
        cleaned = _PI_META_RE.sub("", buf)
        if cleaned:
            yield cleaned
        buf = ""
    if buf:
        cleaned = _PI_META_RE.sub("", buf)
        if cleaned:
            yield cleaned


# ── PageIndex chat: stream via PageIndex SDK ──────────────
def _pageindex_generate(chat_id: str, meta: dict, user_message: str, history: list):
    pi_messages = [{"role": m["role"], "content": m["content"]} for m in history]
    pi_messages.append({"role": "user", "content": user_message})

    pi_client = get_pi_client()
    full_response = ""
    try:
        raw_stream = pi_client.chat_completions(
            messages=pi_messages,
            doc_id=meta["doc_id"],
            stream=True,
        )
        for chunk in _filter_pageindex_stream(raw_stream):
            full_response += chunk
            yield f"data: {json.dumps({'content': chunk})}\n\n"

        db_save_message(chat_id, "assistant", full_response)
        yield f"data: {json.dumps({'done': True})}\n\n"
    except Exception as e:
        logger.error(f"PageIndex chat error: {traceback.format_exc()}")
        if full_response:
            db_save_message(chat_id, "assistant", full_response)
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post("/api/chat/{chat_id}")
async def chat(chat_id: str, request_body: ChatRequest, user: dict = Depends(get_current_user)):
    meta = db_get_chat_meta(chat_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Chat not found")
    if meta["user_id"] != user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    history = db_get_messages(chat_id)
    db_save_message(chat_id, "user", request_body.message)

    if meta["mode"] == "vision":
        gen = _vision_generate(chat_id, meta, request_body.message, history)
    else:
        gen = _pageindex_generate(chat_id, meta, request_body.message, history)

    return StreamingResponse(gen, media_type="text/event-stream")


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str, user: dict = Depends(get_current_user)):
    meta = db_get_chat_meta(chat_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Chat not found")
    if meta["user_id"] != user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    if meta["mode"] == "pageindex" and meta.get("doc_id"):
        try:
            pi_client = get_pi_client()
            pi_client.delete_document(meta["doc_id"])
        except Exception:
            logger.warning(f"Could not delete PageIndex doc {meta['doc_id']}")

    file_id = meta["chat_id"][:8]
    for i in range(meta.get("page_count", 0)):
        img = UPLOAD_DIR / f"{file_id}_page_{i}.jpg"
        if img.exists():
            img.unlink()

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
