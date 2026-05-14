import json
import os
import sys
from dotenv import load_dotenv
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
JSON_PATH = os.path.join(os.path.dirname(__file__), "IITD_output.json")


def load_records(json_path: str) -> list:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    valid = [r for r in data if r.get("title") and r.get("handle")]
    print(f"Loaded {len(valid)} valid records (skipped {len(data) - len(valid)} without title/handle)")
    return valid


def build_embedding_text(record: dict) -> str:
    parts = [record.get("title", "").strip()]
    keywords = [k for k in record.get("keywords", []) if k]
    if keywords:
        parts.append("Keywords: " + ", ".join(keywords))
    authors = [a for a in record.get("authors", []) if a]
    if authors:
        parts.append("Author: " + ", ".join(authors))
    advisors = [v for v in record.get("advisors", []) if v and v != "XXX"]
    if advisors:
        parts.append("Advisor: " + ", ".join(advisors))
    year = record.get("publication_year")
    if year:
        parts.append(f"Year: {year}")
    return ". ".join(parts)


def build_chroma_metadata(record: dict) -> dict:
    return {
        "oai_id": record.get("oai_id", ""),
        "handle": record.get("handle", ""),
        "landing_page_url": record.get("landing_page_url", ""),
        "pdf_url": record.get("pdf_url", ""),
        "title": record.get("title", ""),
        "authors": json.dumps(record.get("authors", [])),
        "advisors": json.dumps([v for v in record.get("advisors", []) if v and v != "XXX"]),
        "keywords": json.dumps(record.get("keywords", [])),
        "publication_year": record.get("publication_year") or 0,
        "thesis_no": record.get("thesis_no", ""),
    }


def setup_chromadb(persist_dir: str):
    client = chromadb.PersistentClient(path=persist_dir)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name="iitd_theses",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def ingest_chromadb(collection, records: list, batch_size: int = 100) -> None:
    print(f"\nIngesting {len(records)} records into ChromaDB...")
    for i in tqdm(range(0, len(records), batch_size), desc="ChromaDB"):
        batch = records[i : i + batch_size]
        ids = [r["handle"].replace("/", "_") for r in batch]
        documents = [build_embedding_text(r) for r in batch]
        metadatas = [build_chroma_metadata(r) for r in batch]
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"ChromaDB: {collection.count()} documents total")


def setup_neo4j(uri: str, user: str, password: str):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
    except ServiceUnavailable as e:
        print(f"ERROR: Cannot connect to Neo4j at {uri}")
        print("Make sure Neo4j is running and credentials in .env are correct.")
        raise
    return driver


def create_neo4j_constraints(driver) -> None:
    queries = [
        "CREATE CONSTRAINT thesis_handle IF NOT EXISTS FOR (t:Thesis) REQUIRE t.handle IS UNIQUE",
        "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
        "CREATE CONSTRAINT advisor_name IF NOT EXISTS FOR (v:Advisor) REQUIRE v.name IS UNIQUE",
        "CREATE CONSTRAINT keyword_text IF NOT EXISTS FOR (k:Keyword) REQUIRE k.text IS UNIQUE",
        "CREATE INDEX thesis_year IF NOT EXISTS FOR (t:Thesis) ON (t.publication_year)",
        "CREATE INDEX thesis_no_idx IF NOT EXISTS FOR (t:Thesis) ON (t.thesis_no)",
    ]
    with driver.session() as session:
        for q in queries:
            session.run(q)

    # Fulltext index (separate — different syntax)
    with driver.session() as session:
        existing = session.run(
            "SHOW FULLTEXT INDEXES WHERE name = 'thesis_title'"
        ).data()
        if not existing:
            session.run(
                "CREATE FULLTEXT INDEX thesis_title FOR (t:Thesis) ON EACH [t.title]"
            )
    print("Neo4j constraints and indexes created")


def _ingest_batch(session, batch: list) -> None:
    # Call 1: Thesis nodes
    session.run(
        """
        UNWIND $batch AS row
        MERGE (t:Thesis {handle: row.handle})
        SET t.oai_id = row.oai_id,
            t.title = row.title,
            t.pdf_url = row.pdf_url,
            t.landing_page_url = row.landing_page_url,
            t.publication_year = row.publication_year,
            t.thesis_no = row.thesis_no
        """,
        batch=batch,
    )

    # Call 2: Authors
    session.run(
        """
        UNWIND $batch AS row
        MATCH (t:Thesis {handle: row.handle})
        UNWIND row.authors AS name
        WITH t, name WHERE name IS NOT NULL AND name <> ''
        MERGE (a:Author {name: name})
        MERGE (t)-[:AUTHORED_BY]->(a)
        """,
        batch=batch,
    )

    # Call 3: Advisors
    session.run(
        """
        UNWIND $batch AS row
        MATCH (t:Thesis {handle: row.handle})
        UNWIND row.advisors AS name
        WITH t, name WHERE name IS NOT NULL AND name <> '' AND name <> 'XXX'
        MERGE (v:Advisor {name: name})
        MERGE (t)-[:ADVISED_BY]->(v)
        """,
        batch=batch,
    )

    # Call 4: Keywords
    session.run(
        """
        UNWIND $batch AS row
        MATCH (t:Thesis {handle: row.handle})
        UNWIND row.keywords AS kw
        WITH t, kw WHERE kw IS NOT NULL AND kw <> ''
        MERGE (k:Keyword {text: kw})
        MERGE (t)-[:HAS_KEYWORD]->(k)
        """,
        batch=batch,
    )


def ingest_neo4j(driver, records: list, batch_size: int = 200) -> None:
    print(f"\nIngesting {len(records)} records into Neo4j...")
    for i in tqdm(range(0, len(records), batch_size), desc="Neo4j"):
        batch = records[i : i + batch_size]
        # Prepare serializable batch (no nested lists in Cypher params that clash)
        neo_batch = [
            {
                "handle": r.get("handle", ""),
                "oai_id": r.get("oai_id", ""),
                "title": r.get("title", ""),
                "pdf_url": r.get("pdf_url", ""),
                "landing_page_url": r.get("landing_page_url", ""),
                "publication_year": r.get("publication_year") or 0,
                "thesis_no": r.get("thesis_no", ""),
                "authors": [a for a in r.get("authors", []) if a],
                "advisors": [v for v in r.get("advisors", []) if v and v != "XXX"],
                "keywords": [k for k in r.get("keywords", []) if k],
            }
            for r in batch
        ]
        with driver.session() as session:
            _ingest_batch(session, neo_batch)

    with driver.session() as session:
        count = session.run("MATCH (t:Thesis) RETURN count(t) AS n").single()["n"]
        print(f"Neo4j: {count} Thesis nodes total")


def main():
    print("=== IIT Delhi RAG — Data Ingestion ===\n")

    records = load_records(JSON_PATH)

    # ChromaDB
    collection = setup_chromadb(CHROMA_PERSIST_DIR)
    ingest_chromadb(collection, records)

    # Neo4j
    driver = setup_neo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    create_neo4j_constraints(driver)
    ingest_neo4j(driver, records)
    driver.close()

    print("\nIngestion complete. Run `python chat.py` to start querying.")


if __name__ == "__main__":
    main()
