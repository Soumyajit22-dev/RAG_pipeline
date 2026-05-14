Ok my plan was

IIT Delhi XOAI Harvesting
            ↓
Metadata + PDF URL Extraction
            ↓
Selective PDF Downloading
            ↓
PDF Text Extraction
            ↓
Semantic Chunking
            ↓
Knowledge Graph Construction
            ↓
Embeddings + Vector DB
            ↓
Hybrid Retrieval
            ↓
LLM Conversational Layer

so as we have the Metadata and extracted PDF URL .

Lets save this data in a vector db like chroma db, and knowledge graph like neo4j and then we can use hybrid serch on them, so that when user ask some query it will locate the particular data , so that the metadata and pdf url user and LLM both can know .
After locating or understnading which metadata(s) it should return, it will explore the pdf through the pdf url link and will perform ocr using pip install pymupdf. After that that data will be given to LLM system for hybrid serch or output return,
to give the user proper results along with the proof(urls)

use IITD_output.json
the data will be there. When you are going through the located metadatas for their pdf url, if you want you can keep the pdfs in PDF_output directory
@./prompt.md
@./app.py
