import sys
import argparse
import uvicorn
from src.utils.helpers import config, logger

def run_server():
    """Starts the FastAPI Web and API server."""
    app_cfg = config.get("app", {})
    host = app_cfg.get("host", "127.0.0.1")
    port = app_cfg.get("port", 8000)
    log_level = app_cfg.get("log_level", "info").lower()
    
    logger.info(f"Starting API Server on http://{host}:{port}")
    uvicorn.run("src.api.routes:app", host=host, port=port, log_level=log_level, reload=True)

def run_cli_query(query: str):
    """Executes a single RAG query via command line."""
    logger.info("Executing offline CLI query...")
    # Delay imports to avoid initializing clients unless CLI is active
    from src.embeddings.embedder import Embedder
    from src.vectordb.vector_store import VectorStore
    from src.retrieval.retriever import Retriever
    from src.llm.llm_client import LLMClient
    from src.prompts.prompt_templates import DEFAULT_QA_SYSTEM_PROMPT, DEFAULT_QA_USER_TEMPLATE, assemble_prompt_context
    
    embedder = Embedder()
    vector_store = VectorStore()
    retriever = Retriever(embedder, vector_store)
    llm_client = LLMClient()
    
    retrieved = retriever.retrieve(query)
    if not retrieved:
        print("\n[RAG] No document context retrieved. Wrote query using default LLM context.")
        
    context_str = assemble_prompt_context(retrieved)
    user_prompt = DEFAULT_QA_USER_TEMPLATE.format(context=context_str, query=query)
    
    answer = llm_client.generate_response(
        system_prompt=DEFAULT_QA_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        retrieved_chunks=retrieved
    )
    
    print("\n" + "="*40)
    print(f"QUERY: {query}")
    print("="*40)
    print(f"ANSWER:\n{answer}")
    print("="*40)
    if retrieved:
        print("CITATIONS:")
        for idx, r in enumerate(retrieved):
            fname = r["metadata"].get("filename", "unknown")
            page = f" (Page {r['metadata']['page']})" if 'page' in r['metadata'] else ""
            print(f"[{idx+1}] {fname}{page} - score: {r['score']:.2f}")

def run_cli_ingest(file_path: str):
    """Ingests a document file via command line."""
    logger.info(f"Executing offline CLI ingestion for: {file_path}")
    from src.ingestion.loader import load_document
    from src.chunking.chunker import RecursiveCharacterTextSplitter
    from src.embeddings.embedder import Embedder
    from src.vectordb.vector_store import VectorStore
    
    documents = load_document(file_path)
    if not documents:
        print("[Error] Failed to load or parse document.")
        return
        
    splitter = RecursiveCharacterTextSplitter()
    chunks = splitter.split_documents(documents)
    if not chunks:
        print("[Error] Failed to split document into chunks.")
        return
        
    embedder = Embedder()
    vector_store = VectorStore()
    
    texts = [c["content"] for c in chunks]
    embeddings = embedder.embed_documents(texts)
    vector_store.add_documents(chunks, embeddings)
    
    print(f"\n[Success] Ingested '{file_path}'. Added {len(chunks)} chunks to vector database.")

def main():
    parser = argparse.ArgumentParser(description="Modular RAG CLI & Server Runner")
    parser.add_argument("--query", type=str, help="Run a query against the RAG system directly from CLI")
    parser.add_argument("--ingest", type=str, help="Ingest a document file directly from CLI")
    parser.add_argument("--server", action="store_true", help="Start the FastAPI web server (default)")
    
    args = parser.parse_args()
    
    if args.query:
        run_cli_query(args.query)
    elif args.ingest:
        run_cli_ingest(args.ingest)
    else:
        run_server()

if __name__ == "__main__":
    main()
