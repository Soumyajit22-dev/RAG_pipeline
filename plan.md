# Hybrid RAG System — IIT Delhi Thesis Repository

## Context
The OAI harvesting step is complete: `IITD_output.json` contains 6,493 thesis records, all with `pdf_url` fields (direct download links) and populated metadata. Abstracts are universally empty, so all semantic signal comes from title + keywords + authors. The goal is to build a hybrid retrieval system (ChromaDB + Neo4j) that locates relevant theses from a user query, then fetches and OCRs the actual PDFs on-demand to give the LLM grounded, citable answers.

**LLM:** Local Ollama (no API key required) — default model `mistral`, configurable via `.env`  
**User preferences:** Neo4j already running, CLI chat interface, top-5 PDFs per query.

---

## Pipeline Overview

```
User Query (chat.py)
       |
       v
rag_pipeline.py (Orchestrator)
       |
   [Step 1] retrieval.py
       |-- ChromaDB: semantic vector search (cosine similarity, all-MiniLM-L6-v2)
       |-- Neo4j: keyword/author/advisor graph traversal (Cypher)
       |-- RRF Fusion: merge + re-rank both result sets
       |
   [Step 2] pdf_processor.py (top 5 fused results)
       |-- Cache check: PDF_output/{thesis_no}_{handle}.pdf
       |-- If miss: HTTP GET pdf_url → save to cache
       |-- PyMuPDF text extraction + chunk scoring
       |
   [Step 3] Ollama LLM (local, no API key)
       |-- POST http://localhost:11434/api/generate
       |-- Model: mistral (or any Ollama model)
       |-- Streaming response with source citations
       v
Answer + pdf_url proofs (chat.py)
```

---

## File Structure

```
RAG_final/
├── IITD_output.json      (existing — 6,493 thesis records)
├── PDF_output/           (existing — PDF cache, grows on demand)
├── app.py                (existing — OAI harvester, untouched)
├── requirements.txt      (new)
├── .env                  (new — credentials, not committed)
├── ingest.py             (new — one-time load into ChromaDB + Neo4j)
├── pdf_processor.py      (new — on-demand PDF download + PyMuPDF extraction)
├── retrieval.py          (new — hybrid search + RRF fusion)
├── rag_pipeline.py       (new — orchestrator: retrieval → PDF → Ollama LLM)
└── chat.py               (new — CLI chat loop)
```

---

## Step 1: `requirements.txt`

```
chromadb>=0.5.0
neo4j>=5.19.0
pymupdf>=1.24.0
sentence-transformers>=3.0.0
requests>=2.31.0
python-dotenv>=1.0.0
tqdm>=4.66.0
```

No `anthropic` package needed — Ollama is called via plain HTTP.

---

## Step 2: `.env`

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=mistral
CHROMA_PERSIST_DIR=./chroma_db
```

`CHROMA_PERSIST_DIR` is auto-created on first run. No external accounts needed.

---

## Step 3: `ingest.py` — One-time data ingestion

**Purpose:** Load all 6,493 records into ChromaDB (vectors) and Neo4j (graph). Idempotent — safe to re-run.

**Key functions:**
- `load_records(json_path)` — load + validate IITD_output.json
- `build_embedding_text(record)` — `"{title}. Keywords: {kws}. Author: {authors}. Advisor: {advisors}. Year: {year}"` (abstracts always empty, excluded)
- `build_chroma_metadata(record)` — scalar metadata dict; lists JSON-serialized to strings (ChromaDB limitation)
- `setup_chromadb(persist_dir)` — create/get collection `"iitd_theses"` with `SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")`, cosine distance
- `ingest_chromadb(collection, records, batch_size=100)` — `collection.upsert()` with tqdm
- `setup_neo4j(uri, user, password)` — connect + verify connectivity
- `create_neo4j_constraints(driver)` — uniqueness constraints + fulltext index `thesis_title` on Thesis.title
- `ingest_neo4j(driver, records, batch_size=200)` — UNWIND MERGE batches (3 separate calls per batch to avoid cross-products)

**Neo4j schema:**
```
(:Thesis {handle, oai_id, title, landing_page_url, pdf_url, publication_year, thesis_no})
(:Author {name})
(:Advisor {name})
(:Keyword {text})
(:Thesis)-[:AUTHORED_BY]->(:Author)
(:Thesis)-[:ADVISED_BY]->(:Advisor)
(:Thesis)-[:HAS_KEYWORD]->(:Keyword)
```

**Core UNWIND Cypher (3 separate session.run() calls per batch):**
```cypher
-- Call 1: Thesis nodes
UNWIND $batch AS row
MERGE (t:Thesis {handle: row.handle})
SET t.oai_id = row.oai_id, t.title = row.title, t.pdf_url = row.pdf_url,
    t.landing_page_url = row.landing_page_url,
    t.publication_year = row.publication_year, t.thesis_no = row.thesis_no

-- Call 2: Authors
UNWIND $batch AS row
MATCH (t:Thesis {handle: row.handle})
UNWIND row.authors AS name
WITH t, name WHERE name IS NOT NULL AND name <> ''
MERGE (a:Author {name: name})
MERGE (t)-[:AUTHORED_BY]->(a)

-- Call 3: Keywords (same pattern used for Advisors too)
UNWIND $batch AS row
MATCH (t:Thesis {handle: row.handle})
UNWIND row.keywords AS kw
WITH t, kw WHERE kw IS NOT NULL AND kw <> ''
MERGE (k:Keyword {text: kw})
MERGE (t)-[:HAS_KEYWORD]->(k)
```

---

## Step 4: `pdf_processor.py` — On-demand PDF fetching

**Purpose:** Download PDFs on-demand, cache in `PDF_output/`, extract text with PyMuPDF, return top chunks relevant to the query.

**Key functions:**
- `get_cache_path(record)` → `PDF_output/TH6719_123456789_4578.pdf`; fallback to URL UUID hash
- `download_pdf(pdf_url, cache_path)` — streaming GET, 30s timeout, 3 retries with exponential backoff; verify `%PDF-` magic bytes; 50MB size guard
- `extract_text_pymupdf(pdf_path)` — `fitz.open()` → `page.get_text("text")` per page; graceful handling of encrypted/scanned PDFs
- `chunk_text(text, chunk_size=1000, overlap=200)` — splits on `\n\n` paragraph boundaries preferentially
- `score_chunks_by_query(chunks, query, top_n=5)` — keyword overlap TF scoring (no extra embedding needed)
- `get_pdf_text(record, query, top_chunks=5)` → `{success, cached, pdf_path, text_chunks, total_pages, total_chars, error}`

---

## Step 5: `retrieval.py` — Hybrid search

**Purpose:** Combine ChromaDB semantic search + Neo4j graph traversal via Reciprocal Rank Fusion.

**Key functions:**
- `semantic_search(collection, query, top_k=20)` — ChromaDB `.query()`, deserializes JSON-string metadata back to lists
- `extract_query_entities(query)` → `{keywords, years, person_hints, has_person_signal}` via regex (no LLM call)
- `graph_search(driver, entities, top_k=20)` — up to 4 Cypher queries:
  1. Fulltext index on `thesis_title`
  2. Keyword node traversal
  3. Year-filtered keyword search
  4. Author/advisor name search (if detected)
- `reciprocal_rank_fusion(semantic, graph, k=60, sem_weight=0.6, graph_weight=0.4)` — weights flip to 0.3/0.7 for entity-focused queries
- `hybrid_search(collection, driver, query, top_k=10)` — orchestrates above; degrades gracefully to semantic-only if Neo4j is offline

**RRF formula:**
```
fused_score(d) = 0.6 / (60 + semantic_rank(d))
               + 0.4 / (60 + graph_rank(d))
```

---

## Step 6: `rag_pipeline.py` — Orchestrator + Ollama LLM

**Purpose:** End-to-end: query → hybrid search → PDF extraction → Ollama → answer with citations.

**Ollama call (streaming):**
```python
resp = requests.post(
    OLLAMA_URL,                          # http://localhost:11434/api/generate
    json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
    stream=True, timeout=300
)
for line in resp.iter_lines():
    chunk = json.loads(line)
    token = chunk.get("response", "")    # stream token-by-token to stdout
    if chunk.get("done"): break
```

**Key functions:**
- `format_metadata_context(results)` — structured block for top-10 results
- `format_pdf_context(pdf_results, results)` — top-5 PDF chunks, truncated to ~8,000 chars total
- `build_prompt(query, metadata_ctx, pdf_ctx)` — system prompt + XML-delimited contexts + user question (single string for Ollama)
- `call_llm(prompt, stream_callback)` — POST to Ollama, stream tokens
- `run_query(query, collection, driver, top_k_retrieve=10, top_k_pdf=5)`:
  1. `hybrid_search()` → 10 results
  2. `ThreadPoolExecutor(max_workers=3)` — fetch top-5 PDFs concurrently (45s timeout)
  3. Format prompt → call Ollama → stream answer
  4. Return `{answer, sources, pdf_fetched, pdf_failed, retrieval_time_s, pdf_time_s, llm_time_s}`

**System prompt:** Research assistant for IIT Delhi theses. Cite every source with title, author, year, and `pdf_url`. Never fabricate. End with a "Sources" section.

---

## Step 7: `chat.py` — CLI interface

**Purpose:** Interactive REPL with conversation history.

- `initialize_clients()` — connect ChromaDB + Neo4j + verify Ollama is reachable
- `run_chat_loop(collection, driver)`:
  - Commands: `quit`/`exit`, `clear`, `help`
  - Streams LLM output token-by-token to stdout
  - Prints sources table after each answer
  - Maintains last-3-turn history prepended to each new query

---

## Important Data Notes

- `abstract` is always `""` in all 6,493 records — never embed it
- `landing_page_url` uses internal IP `10.17.50.146:4000` — not publicly accessible; `pdf_url` is the public citation link
- 12 records have empty `pdf_url` — skipped gracefully in `pdf_processor.py`
- ~2,000 records have no `thesis_no` — cache path falls back to handle-based filename

---

## Storage Locations

| What | Path | Size |
|------|------|------|
| ChromaDB vectors | `RAG_final/chroma_db/` | ~500 MB |
| Downloaded PDFs | `RAG_final/PDF_output/` | grows ~5 MB/PDF |
| MiniLM model | `~/.cache/huggingface/hub/` | ~80 MB |
| Ollama models | `~/.ollama/models/` | ~4 GB (mistral) |
| venv packages | `RAG_final/venv/` | ~3–5 GB |
| Neo4j graph | Neo4j data directory | ~50–100 MB |

---

## How to Run

```bash
# 1. Install dependencies
source venv/bin/activate
pip install -r requirements.txt

# 2. Pull Ollama model (one-time)
ollama pull mistral

# 3. Create .env from .env.example and fill in Neo4j password

# 4. Run ingestion (~10–20 min for embeddings)
python ingest.py

# 5. Start chatting
python chat.py
```

**Example queries:**
- `What are recent theses on reinforcement learning?`
- `Find theses supervised by Bhawani Sankar Panda`
- `Theses on solar energy from 2022 and 2023`
- `Who has worked on influence maximization in networks?`
