import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from dotenv import load_dotenv
import anthropic

from retrieval import hybrid_search
from pdf_processor import get_pdf_text

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-6"
MAX_PDF_CONTEXT_CHARS = 8000
PDF_FETCH_TIMEOUT = 45  # seconds per PDF

SYSTEM_PROMPT = """You are a research assistant specializing in IIT Delhi theses from the institutional repository (ir.iitd.ac.in).

You help researchers find and understand relevant thesis work. You will receive:
1. Metadata for the most relevant theses found via hybrid search (vector + knowledge graph)
2. Extracted text chunks from the actual thesis PDFs (when available)

Guidelines:
- Synthesize the provided information to directly answer the user's query
- ALWAYS cite your sources using the thesis title, author, year, and pdf_url
- When PDF text is available, reference specific findings, methods, or conclusions from it
- When PDF text is unavailable or empty, answer based on metadata alone and note the limitation
- Be concise yet thorough; group related theses when discussing similar topics
- Do NOT fabricate information not present in the provided context
- End every response with a "Sources" section listing all cited theses

Source citation format:
[N] {Author(s)} ({Year}). "{Title}". PDF: {pdf_url}"""


def format_metadata_context(results: list) -> str:
    lines = []
    for i, r in enumerate(results, start=1):
        authors = r.get("authors", [])
        advisors = r.get("advisors", [])
        keywords = r.get("keywords", [])
        year = r.get("publication_year", "Unknown")
        lines.append(
            f"[{i}] Title: {r.get('title', 'N/A')}\n"
            f"    Author(s): {', '.join(authors) if authors else 'N/A'}\n"
            f"    Advisor(s): {', '.join(advisors) if advisors else 'N/A'}\n"
            f"    Year: {year} | Thesis No: {r.get('thesis_no', 'N/A')}\n"
            f"    Keywords: {', '.join(keywords) if keywords else 'N/A'}\n"
            f"    PDF URL: {r.get('pdf_url', 'N/A')}\n"
            f"    Relevance Score: {r.get('fused_score', 0):.4f}"
        )
    return "\n\n".join(lines)


def format_pdf_context(pdf_results: dict, results: list) -> str:
    sections = []
    total_chars = 0

    for r in results:
        handle = r.get("handle", "")
        if handle not in pdf_results:
            continue
        pdf = pdf_results[handle]
        title = r.get("title", "Unknown")
        authors = r.get("authors", [])
        year = r.get("publication_year", "")

        header = f'--- "{title}" | {", ".join(authors)} ({year}) ---'

        if not pdf["success"]:
            sections.append(f"{header}\n[PDF unavailable: {pdf.get('error', 'unknown error')}]")
            continue

        chunks = pdf.get("text_chunks", [])
        if not chunks:
            sections.append(f"{header}\n[No extractable text — possibly a scanned PDF]")
            continue

        chunk_text = "\n\n".join(chunks)
        if total_chars + len(chunk_text) > MAX_PDF_CONTEXT_CHARS:
            remaining = MAX_PDF_CONTEXT_CHARS - total_chars
            if remaining < 200:
                break
            chunk_text = chunk_text[:remaining] + "\n[...truncated]"

        sections.append(f"{header}\n{chunk_text}")
        total_chars += len(chunk_text)

        if total_chars >= MAX_PDF_CONTEXT_CHARS:
            break

    return "\n\n".join(sections) if sections else "[No PDF text was successfully extracted]"


def build_messages(query: str, metadata_ctx: str, pdf_ctx: str) -> list:
    content = (
        "<metadata_search_results>\n"
        f"{metadata_ctx}\n"
        "</metadata_search_results>\n\n"
        "<pdf_text_extracts>\n"
        f"{pdf_ctx}\n"
        "</pdf_text_extracts>\n\n"
        f"User question: {query}"
    )
    return [{"role": "user", "content": content}]


def call_llm(client: anthropic.Anthropic, messages: list, stream_callback=None) -> str:
    for attempt in range(3):
        try:
            full_text = ""
            with client.messages.stream(
                model=MODEL,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    full_text += text
                    if stream_callback:
                        stream_callback(text)
            return full_text
        except anthropic.RateLimitError:
            if attempt < 2:
                time.sleep(60)
            else:
                raise
        except anthropic.APIError:
            raise


def run_query(
    query: str,
    collection,
    driver,
    client: anthropic.Anthropic,
    top_k_retrieve: int = 10,
    top_k_pdf: int = 5,
    stream_callback=None,
) -> dict:
    t0 = time.time()

    # Step 1: Hybrid retrieval
    results = hybrid_search(collection, driver, query, top_k=top_k_retrieve)
    retrieval_time = time.time() - t0

    # Step 2: Fetch PDFs for top results concurrently
    t1 = time.time()
    pdf_candidates = [r for r in results[:top_k_pdf] if r.get("pdf_url")]
    pdf_results = {}
    pdf_fetched = []
    pdf_failed = []

    def fetch(record):
        return record["handle"], get_pdf_text(record, query, top_chunks=5)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fetch, r): r["handle"] for r in pdf_candidates}
        for future in as_completed(futures, timeout=PDF_FETCH_TIMEOUT + 5):
            try:
                handle, result = future.result(timeout=PDF_FETCH_TIMEOUT)
                pdf_results[handle] = result
                if result["success"]:
                    pdf_fetched.append(handle)
                else:
                    pdf_failed.append(handle)
            except Exception:
                handle = futures[future]
                pdf_failed.append(handle)
                pdf_results[handle] = {
                    "success": False, "error": "Fetch timeout or error",
                    "text_chunks": [], "total_pages": 0, "total_chars": 0,
                }

    pdf_time = time.time() - t1

    # Step 3: Format context and call LLM
    t2 = time.time()
    metadata_ctx = format_metadata_context(results)
    pdf_ctx = format_pdf_context(pdf_results, results)
    messages = build_messages(query, metadata_ctx, pdf_ctx)
    answer = call_llm(client, messages, stream_callback=stream_callback)
    llm_time = time.time() - t2

    return {
        "answer": answer,
        "sources": results,
        "pdf_fetched": pdf_fetched,
        "pdf_failed": pdf_failed,
        "retrieval_time_s": round(retrieval_time, 2),
        "pdf_time_s": round(pdf_time, 2),
        "llm_time_s": round(llm_time, 2),
    }
