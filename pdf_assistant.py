import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import base64
import os
import re
import time
from pathlib import Path

import anthropic
import fitz as pymupdf
import typer
from dotenv import load_dotenv
from pageindex import PageIndexClient

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")

VISION_PAGE_THRESHOLD = 15
VISION_DPI = 150
VISION_MODEL = "claude-sonnet-4-20250514"


def _load_page_images_b64(pdf_path: str) -> list[dict]:
    """Render each page of a PDF to JPEG and return as base64 content blocks."""
    doc = pymupdf.open(pdf_path)
    blocks = []
    for i in range(doc.page_count):
        pix = doc[i].get_pixmap(dpi=VISION_DPI)
        img_bytes = pix.tobytes("jpeg")
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            }
        )
    page_count = doc.page_count
    doc.close()
    return blocks, page_count


def _vision_chat_loop(pdf_path: str, pdf_name: str, page_images: list[dict], page_count: int):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    system_prompt = (
        f"You are a helpful PDF assistant. The user has uploaded '{pdf_name}' "
        f"({page_count} pages). The page images are provided below. "
        "Answer questions based ONLY on what you can see in these pages. "
        "Be precise with names, dates, numbers, and spelling — read them exactly as shown. "
        "If something is unclear in the image, say so rather than guessing."
    )

    messages = []

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "bye"):
            typer.echo("Goodbye!")
            break

        user_content = list(page_images)
        user_content.append({"type": "text", "text": query})
        messages.append({"role": "user", "content": user_content})

        typer.echo("\nAssistant: ", nl=False)
        full_response = ""
        try:
            with client.messages.stream(
                model=VISION_MODEL,
                max_tokens=4096,
                system=system_prompt,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    full_response += text
        except Exception as e:
            typer.echo(f"\nError: {e}", err=True)
            messages.pop()
            continue

        print("\n")
        messages.append({"role": "assistant", "content": full_response})


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


def _pageindex_chat_loop(doc_id: str, pi_client: PageIndexClient):
    messages = []

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "bye"):
            typer.echo("Goodbye!")
            break

        messages.append({"role": "user", "content": query})

        typer.echo("\nAssistant: ", nl=False)
        full_response = ""
        try:
            raw_stream = pi_client.chat_completions(
                messages=messages,
                doc_id=doc_id,
                stream=True,
            )
            for chunk in _filter_pageindex_stream(raw_stream):
                print(chunk, end="", flush=True)
                full_response += chunk
        except Exception as e:
            typer.echo(f"\nError: {e}", err=True)
            messages.pop()
            continue

        print("\n")
        messages.append({"role": "assistant", "content": full_response})


def pdf_assistant(
    pdf_path: str = typer.Argument(..., help="Path to a local PDF file"),
):
    if not os.path.isfile(pdf_path):
        typer.echo(f"Error: File not found: {pdf_path}", err=True)
        raise typer.Exit(1)

    pdf_name = Path(pdf_path).name

    doc = pymupdf.open(pdf_path)
    page_count = doc.page_count
    doc.close()

    typer.echo(f"PDF: {pdf_name} ({page_count} pages)")

    if page_count <= VISION_PAGE_THRESHOLD:
        if not ANTHROPIC_API_KEY:
            typer.echo("Error: ANTHROPIC_API_KEY not set in .env", err=True)
            raise typer.Exit(1)

        typer.echo(f"Using Claude Vision path (pages <= {VISION_PAGE_THRESHOLD})")
        typer.echo("Converting pages to images...")
        page_images, _ = _load_page_images_b64(pdf_path)
        typer.echo("Ready! Ask questions below (type 'exit' to quit).\n")
        _vision_chat_loop(pdf_path, pdf_name, page_images, page_count)
    else:
        if not PAGEINDEX_API_KEY:
            typer.echo("Error: PAGEINDEX_API_KEY not set in .env", err=True)
            raise typer.Exit(1)

        typer.echo(f"Using PageIndex path (pages > {VISION_PAGE_THRESHOLD})")
        pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

        typer.echo("Submitting document to PageIndex...")
        result = pi_client.submit_document(pdf_path)
        doc_id = result["doc_id"]
        typer.echo(f"Document submitted (doc_id: {doc_id}). Processing...")

        elapsed = 0
        while elapsed < 180:
            doc_info = pi_client.get_document(doc_id)
            status = doc_info.get("status", "unknown")
            if status == "completed":
                break
            if status == "failed":
                typer.echo("Error: Document processing failed.", err=True)
                raise typer.Exit(1)
            time.sleep(2)
            elapsed += 2
        else:
            typer.echo("Error: Document processing timed out.", err=True)
            raise typer.Exit(1)

        typer.echo("Ready! Ask questions below (type 'exit' to quit).\n")
        _pageindex_chat_loop(doc_id, pi_client)


if __name__ == "__main__":
    typer.run(pdf_assistant)
