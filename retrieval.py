import json
import os
import re
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "what", "which", "who", "how", "when", "where",
    "that", "this", "these", "those", "about", "find", "theses", "thesis",
    "research", "study", "work", "using", "based", "show", "paper",
}

PERSON_SIGNALS = re.compile(
    r"\b(advisor|adviser|supervised by|supervisor|prof|professor|dr|author|by)\b",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def get_chroma_collection(persist_dir: str = None):
    path = persist_dir or CHROMA_PERSIST_DIR
    client = chromadb.PersistentClient(path=path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    try:
        collection = client.get_collection(name="iitd_theses", embedding_function=ef)
    except Exception:
        raise RuntimeError(
            "ChromaDB collection 'iitd_theses' not found. Run `python ingest.py` first."
        )
    return collection


def _deserialize_metadata(meta: dict) -> dict:
    for field in ("authors", "advisors", "keywords"):
        val = meta.get(field, "[]")
        if isinstance(val, str):
            try:
                meta[field] = json.loads(val)
            except Exception:
                meta[field] = []
    return meta


def semantic_search(collection, query: str, top_k: int = 20) -> list:
    results = collection.query(query_texts=[query], n_results=min(top_k, collection.count()))
    docs = []
    ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for rank, (doc_id, meta, dist) in enumerate(zip(ids, metadatas, distances), start=1):
        meta = _deserialize_metadata(dict(meta))
        meta["chroma_rank"] = rank
        meta["chroma_distance"] = dist
        docs.append(meta)
    return docs


def extract_query_entities(query: str) -> dict:
    years = [int(y) for y in YEAR_PATTERN.findall(query)]

    # Detect person name hints after signal words
    person_hints = []
    for match in PERSON_SIGNAL_NAME.finditer(query):
        name = match.group(1).strip().strip(",;")
        if name:
            person_hints.append(name)

    has_person_signal = bool(PERSON_SIGNALS.search(query))

    # Extract keyword terms
    cleaned = re.sub(r"\b(advisor|adviser|supervised by|supervisor|prof|professor|dr|author|by|from|after|between|and)\b", " ", query, flags=re.IGNORECASE)
    cleaned = re.sub(YEAR_PATTERN, " ", cleaned)
    tokens = [t.lower() for t in re.findall(r"\b[a-zA-Z]{3,}\b", cleaned) if t.lower() not in STOPWORDS]

    return {
        "raw_query": query,
        "keywords": tokens,
        "years": years,
        "person_hints": person_hints,
        "has_person_signal": has_person_signal,
    }


# Match pattern like "advisor Bhawani Sankar" or "by Prof. John Smith"
PERSON_SIGNAL_NAME = re.compile(
    r"(?:advisor|adviser|supervised by|supervisor|prof\.?\s+|professor\s+|dr\.?\s+|author\s+|by\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
    re.IGNORECASE,
)


def graph_search(driver, entities: dict, top_k: int = 20) -> list:
    results_by_handle = {}
    rank = 1

    def add_result(record, score_boost=0):
        nonlocal rank
        handle = record.get("handle", "")
        if not handle:
            return
        if handle not in results_by_handle:
            results_by_handle[handle] = {
                "handle": handle,
                "oai_id": record.get("oai_id", ""),
                "title": record.get("title", ""),
                "pdf_url": record.get("pdf_url", ""),
                "landing_page_url": record.get("landing_page_url", ""),
                "authors": record.get("authors", []) if isinstance(record.get("authors"), list) else [],
                "advisors": record.get("advisors", []) if isinstance(record.get("advisors"), list) else [],
                "keywords": record.get("keywords", []) if isinstance(record.get("keywords"), list) else [],
                "publication_year": record.get("publication_year", 0),
                "thesis_no": record.get("thesis_no", ""),
                "graph_rank": rank,
                "graph_score": score_boost,
            }
            rank += 1
        else:
            results_by_handle[handle]["graph_score"] += score_boost

    with driver.session() as session:
        # Query 1: Fulltext search on thesis title
        raw_query = entities["raw_query"]
        try:
            ft_results = session.run(
                """
                CALL db.index.fulltext.queryNodes('thesis_title', $q)
                YIELD node, score
                RETURN node.handle AS handle, node.title AS title,
                       node.pdf_url AS pdf_url, node.landing_page_url AS landing_page_url,
                       node.publication_year AS publication_year, node.thesis_no AS thesis_no,
                       node.oai_id AS oai_id, score
                LIMIT $limit
                """,
                q=raw_query,
                limit=top_k,
            ).data()
            for r in ft_results:
                add_result(r, score_boost=r.get("score", 1))
        except Exception:
            pass  # fulltext index may not exist yet

        # Query 2: Keyword node traversal
        keywords = entities.get("keywords", [])
        if keywords:
            kw_results = session.run(
                """
                MATCH (k:Keyword)<-[:HAS_KEYWORD]-(t:Thesis)
                WHERE any(kw IN $kws WHERE toLower(k.text) CONTAINS kw)
                WITH t, count(k) AS matches
                ORDER BY matches DESC
                LIMIT $limit
                RETURN t.handle AS handle, t.title AS title,
                       t.pdf_url AS pdf_url, t.landing_page_url AS landing_page_url,
                       t.publication_year AS publication_year, t.thesis_no AS thesis_no,
                       t.oai_id AS oai_id, matches
                """,
                kws=keywords,
                limit=top_k,
            ).data()
            for r in kw_results:
                add_result(r, score_boost=r.get("matches", 1))

        # Query 3: Year filter if years detected
        years = entities.get("years", [])
        if years and keywords:
            year_min, year_max = min(years), max(years)
            yr_results = session.run(
                """
                MATCH (k:Keyword)<-[:HAS_KEYWORD]-(t:Thesis)
                WHERE t.publication_year >= $y_min AND t.publication_year <= $y_max
                  AND any(kw IN $kws WHERE toLower(k.text) CONTAINS kw)
                WITH t, count(k) AS matches
                ORDER BY matches DESC
                LIMIT $limit
                RETURN t.handle AS handle, t.title AS title,
                       t.pdf_url AS pdf_url, t.landing_page_url AS landing_page_url,
                       t.publication_year AS publication_year, t.thesis_no AS thesis_no,
                       t.oai_id AS oai_id, matches
                """,
                y_min=year_min,
                y_max=year_max,
                kws=keywords,
                limit=top_k,
            ).data()
            for r in yr_results:
                add_result(r, score_boost=r.get("matches", 1) * 1.5)

        # Query 4: Person name search (author or advisor)
        person_hints = entities.get("person_hints", [])
        if person_hints:
            for name in person_hints:
                name_lower = name.lower()
                person_results = session.run(
                    """
                    MATCH (t:Thesis)-[:AUTHORED_BY|ADVISED_BY]->(p)
                    WHERE toLower(p.name) CONTAINS $name
                    RETURN t.handle AS handle, t.title AS title,
                           t.pdf_url AS pdf_url, t.landing_page_url AS landing_page_url,
                           t.publication_year AS publication_year, t.thesis_no AS thesis_no,
                           t.oai_id AS oai_id
                    LIMIT $limit
                    """,
                    name=name_lower,
                    limit=top_k,
                ).data()
                for r in person_results:
                    add_result(r, score_boost=5)  # strong boost for exact entity match

    # Convert to ranked list
    graph_list = sorted(results_by_handle.values(), key=lambda x: x["graph_score"], reverse=True)
    for i, item in enumerate(graph_list, start=1):
        item["graph_rank"] = i
    return graph_list[:top_k]


def reciprocal_rank_fusion(
    semantic_results: list,
    graph_results: list,
    k: int = 60,
    sem_weight: float = 0.6,
    graph_weight: float = 0.4,
) -> list:
    scores = {}
    all_docs = {}

    for doc in semantic_results:
        handle = doc["handle"]
        rank = doc["chroma_rank"]
        scores[handle] = scores.get(handle, 0.0) + sem_weight / (k + rank)
        all_docs[handle] = dict(doc)
        all_docs[handle]["in_semantic"] = True
        all_docs[handle]["semantic_rank"] = rank

    for doc in graph_results:
        handle = doc["handle"]
        rank = doc["graph_rank"]
        scores[handle] = scores.get(handle, 0.0) + graph_weight / (k + rank)
        if handle not in all_docs:
            all_docs[handle] = dict(doc)
            all_docs[handle]["in_semantic"] = False
            all_docs[handle]["semantic_rank"] = None
        all_docs[handle]["in_graph"] = True
        all_docs[handle]["graph_rank"] = rank

    for handle in all_docs:
        all_docs[handle].setdefault("in_graph", False)
        all_docs[handle].setdefault("graph_rank", None)
        all_docs[handle]["fused_score"] = scores[handle]

    return sorted(all_docs.values(), key=lambda x: x["fused_score"], reverse=True)


def hybrid_search(
    collection,
    driver,
    query: str,
    top_k: int = 10,
) -> list:
    sem_results = semantic_search(collection, query, top_k=20)
    entities = extract_query_entities(query)

    # Flip weights if the query is entity-focused (person name detected)
    sem_w, graph_w = 0.6, 0.4
    if entities["has_person_signal"] and entities["person_hints"]:
        sem_w, graph_w = 0.3, 0.7

    graph_results = []
    if driver is not None:
        try:
            graph_results = graph_search(driver, entities, top_k=20)
        except Exception as e:
            print(f"[Warning] Neo4j graph search failed: {e} — using semantic only")

    fused = reciprocal_rank_fusion(sem_results, graph_results, sem_weight=sem_w, graph_weight=graph_w)
    return fused[:top_k]
