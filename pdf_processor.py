import hashlib
import re
import time
from pathlib import Path

import fitz  # PyMuPDF
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PDF_CACHE_DIR = Path(__file__).parent / "PDF_output"
DOWNLOAD_TIMEOUT = 30
MAX_PDF_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "what", "which", "who", "how", "when", "where",
    "that", "this", "these", "those", "it", "its", "i", "me", "my", "we",
    "you", "he", "she", "they", "them", "their", "about", "find", "theses",
    "thesis", "research", "study", "work", "using", "based",
}


def get_cache_path(record: dict) -> Path:
    thesis_no = record.get("thesis_no", "").strip()
    handle = record.get("handle", "").replace("/", "_")
    if thesis_no and handle:
        filename = f"{thesis_no}_{handle}.pdf"
    elif handle:
        filename = f"{handle}.pdf"
    else:
        url_hash = hashlib.md5(record.get("pdf_url", "").encode()).hexdigest()[:12]
        filename = f"pdf_{url_hash}.pdf"
    return PDF_CACHE_DIR / filename


def is_cached(record: dict) -> bool:
    return get_cache_path(record).exists()


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; ThesisRAG/1.0; research use)"
    })
    return session


def download_pdf(pdf_url: str, cache_path: Path) -> bool:
    if not pdf_url:
        return False
    session = _make_session()
    try:
        with session.get(pdf_url, stream=True, timeout=DOWNLOAD_TIMEOUT) as resp:
            if resp.status_code != 200:
                return False
            content_length = int(resp.headers.get("Content-Length", 0))
            if content_length > MAX_PDF_SIZE_BYTES:
                return False

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(".tmp")
            downloaded = 0
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    downloaded += len(chunk)
                    if downloaded > MAX_PDF_SIZE_BYTES:
                        tmp_path.unlink(missing_ok=True)
                        return False
                    f.write(chunk)

            # Verify PDF magic bytes
            with open(tmp_path, "rb") as f:
                header = f.read(5)
            if header[:4] != b"%PDF":
                tmp_path.unlink(missing_ok=True)
                return False

            tmp_path.rename(cache_path)
            return True
    except Exception:
        return False


def extract_text_pymupdf(pdf_path: Path) -> tuple[str, int]:
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return "", 0

    pages_text = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if not text.strip():
            text = page.get_text("rawdict")
            if isinstance(text, dict):
                text = " ".join(
                    span["text"]
                    for block in text.get("blocks", [])
                    for line in block.get("lines", [])
                    for span in line.get("spans", [])
                )
        if text.strip():
            pages_text.append(f"--- Page {page_num} ---\n{text.strip()}")

    total_pages = len(doc)
    doc.close()
    return "\n\n".join(pages_text), total_pages


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    if not text:
        return []
    paragraphs = re.split(r"\n{2,}", text)
    chunks = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            if len(para) <= chunk_size:
                # carry overlap from previous chunk
                if chunks:
                    tail = chunks[-1][-overlap:] if len(chunks[-1]) > overlap else chunks[-1]
                    current = (tail + "\n\n" + para).strip()
                else:
                    current = para
            else:
                # Split long paragraph by character
                for start in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[start : start + chunk_size])
                current = ""
    if current:
        chunks.append(current)
    return chunks


def score_chunks_by_query(chunks: list, query: str, top_n: int = 5) -> list:
    query_terms = {
        t.lower() for t in re.findall(r"\b\w+\b", query) if t.lower() not in STOPWORDS
    }
    if not query_terms:
        return chunks[:top_n]

    scored = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = sum(chunk_lower.count(term) for term in query_terms)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_n]]


def get_pdf_text(record: dict, query: str, top_chunks: int = 5) -> dict:
    pdf_url = record.get("pdf_url", "")
    if not pdf_url:
        return {
            "success": False, "cached": False, "pdf_path": None,
            "text_chunks": [], "total_pages": 0, "total_chars": 0,
            "error": "No pdf_url in record",
        }

    cache_path = get_cache_path(record)
    cached = cache_path.exists()

    if not cached:
        ok = download_pdf(pdf_url, cache_path)
        if not ok:
            return {
                "success": False, "cached": False, "pdf_path": None,
                "text_chunks": [], "total_pages": 0, "total_chars": 0,
                "error": "PDF download failed",
            }

    full_text, total_pages = extract_text_pymupdf(cache_path)
    if not full_text.strip():
        return {
            "success": False, "cached": cached, "pdf_path": str(cache_path),
            "text_chunks": [], "total_pages": total_pages, "total_chars": 0,
            "error": "No extractable text (possibly scanned PDF)",
        }

    chunks = chunk_text(full_text)
    top = score_chunks_by_query(chunks, query, top_n=top_chunks)

    return {
        "success": True, "cached": cached, "pdf_path": str(cache_path),
        "text_chunks": top, "total_pages": total_pages,
        "total_chars": len(full_text), "error": None,
    }
