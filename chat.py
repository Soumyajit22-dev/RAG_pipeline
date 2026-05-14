import os
import sys
import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from retrieval import get_chroma_collection
from rag_pipeline import run_query, OLLAMA_URL, OLLAMA_MODEL

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║         IIT Delhi Thesis Research Assistant (RAG)            ║
║   Hybrid Search: ChromaDB (semantic) + Neo4j (graph)         ║
╠══════════════════════════════════════════════════════════════╣
║  Commands: 'quit'/'exit' to exit | 'clear' to reset history  ║
║            'help' for usage tips                             ║
╚══════════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
Usage tips:
  - Ask about topics:    "What are recent theses on solar energy?"
  - Filter by year:      "Theses on deep learning from 2022 and 2023"
  - Search by advisor:   "Find theses supervised by Prof. Shankar Ravi"
  - Search by author:    "Theses authored by Sanjay Kumar"
  - Follow-up:           Ask a follow-up without repeating context

The system will:
  1. Search 6,493 IIT Delhi theses using semantic + graph search
  2. Download and extract text from the top-5 matching PDFs
  3. Generate a grounded answer with source citations (PDF URLs)
"""


def initialize_clients():
    print("Connecting to services...")

    # ChromaDB
    try:
        collection = get_chroma_collection(CHROMA_PERSIST_DIR)
        count = collection.count()
        print(f"  ChromaDB: {count} theses indexed")
    except RuntimeError as e:
        print(f"  ChromaDB ERROR: {e}")
        sys.exit(1)

    # Neo4j
    driver = None
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        with driver.session() as session:
            n = session.run("MATCH (t:Thesis) RETURN count(t) AS n").single()["n"]
        print(f"  Neo4j:    {n} Thesis nodes")
    except ServiceUnavailable:
        print("  Neo4j:    [OFFLINE] — running in semantic-only mode")
        driver = None
    except Exception as e:
        print(f"  Neo4j:    [WARNING] {e} — running in semantic-only mode")
        driver = None

    # Ollama
    try:
        resp = requests.get(OLLAMA_URL.replace("/api/generate", "/api/tags"), timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            if any(OLLAMA_MODEL in m for m in models):
                print(f"  Ollama:   {OLLAMA_MODEL} ready")
            else:
                print(f"  Ollama:   WARNING — model '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}")
        else:
            print(f"  Ollama:   WARNING — server responded with {resp.status_code}")
    except Exception:
        print(f"  Ollama:   WARNING — cannot reach {OLLAMA_URL}. Is Ollama running?")

    return collection, driver


def print_sources_table(sources: list, pdf_fetched: list):
    print("\n" + "─" * 72)
    print(f"{'#':<3} {'Title':<42} {'Author':<18} {'Yr':<5} {'PDF'}")
    print("─" * 72)
    fetched_set = set(pdf_fetched)
    for i, s in enumerate(sources[:10], start=1):
        title = s.get("title", "")[:41]
        authors = s.get("authors", [])
        author = (authors[0] if authors else "N/A")[:17]
        year = str(s.get("publication_year", ""))[:4]
        pdf_status = "OK" if s.get("handle") in fetched_set else "--"
        print(f"{i:<3} {title:<42} {author:<18} {year:<5} {pdf_status}")
    print("─" * 72)


def format_timing(result: dict) -> str:
    return (
        f"Retrieval: {result['retrieval_time_s']}s | "
        f"PDF: {result['pdf_time_s']}s | "
        f"LLM: {result['llm_time_s']}s"
    )


def run_chat_loop(collection, driver):
    print(BANNER)
    history = []  # list of (query, answer) tuples, last 3

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if query.lower() == "clear":
            history.clear()
            print("Conversation history cleared.")
            continue

        if query.lower() == "help":
            print(HELP_TEXT)
            continue

        # Build augmented query with last 3 turns of history
        if history:
            history_text = "\n".join(
                f"Previous Q: {h[0]}\nPrevious A (summary): {h[1][:300]}..."
                for h in history[-3:]
            )
            augmented_query = f"[Conversation context]\n{history_text}\n\nCurrent question: {query}"
        else:
            augmented_query = query

        print("\nAssistant: ", end="", flush=True)

        try:
            result = run_query(
                query=augmented_query,
                collection=collection,
                driver=driver,
                top_k_retrieve=10,
                top_k_pdf=5,
                stream_callback=lambda text: print(text, end="", flush=True),
            )
        except Exception as e:
            print(f"\n[Error] {e}")
            continue

        print()  # newline after streamed response
        print_sources_table(result["sources"], result["pdf_fetched"])
        print(f"[{format_timing(result)} | "
              f"PDFs fetched: {len(result['pdf_fetched'])}/{len(result['pdf_fetched']) + len(result['pdf_failed'])}]")

        history.append((query, result["answer"]))
        if len(history) > 3:
            history.pop(0)


def main():
    collection, driver = initialize_clients()
    run_chat_loop(collection, driver)
    if driver:
        driver.close()


if __name__ == "__main__":
    main()
