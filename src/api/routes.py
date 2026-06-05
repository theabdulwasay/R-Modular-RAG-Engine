import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.utils.helpers import logger, BASE_DIR
from src.ingestion.loader import load_document
from src.chunking.chunker import RecursiveCharacterTextSplitter
from src.embeddings.embedder import Embedder
from src.vectordb.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.llm.llm_client import LLMClient
from src.prompts.prompt_templates import (
    DEFAULT_QA_SYSTEM_PROMPT,
    DEFAULT_QA_USER_TEMPLATE,
    assemble_prompt_context
)

# Initialize FastAPI App
app = FastAPI(title="Modular RAG API")

# Configure CORS for browser-based dashboard and API clients
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost,http://127.0.0.1").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allowed_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Pipeline components
try:
    embedder = Embedder()
    vector_store = VectorStore()
    retriever = Retriever(embedder, vector_store)
    llm_client = LLMClient()
except Exception as e:
    logger.error(f"Error initializing RAG components: {e}")
    raise e

class QueryRequest(BaseModel):
    query: str
    top_k: int = 8


@app.get("/api/status")
async def api_status():
    """Returns current config and provider status for troubleshooting and UI display."""
    try:
        emb_provider = getattr(embedder, "provider", None)
        llm_provider = getattr(llm_client, "provider", None)
        vect_provider = config.get("vectordb", {}).get("provider")
        gemini_key = bool(get_env_var("GEMINI_API_KEY"))
        openai_key = bool(get_env_var("OPENAI_API_KEY"))

        return {
            "app": config.get("app", {}),
            "providers": {
                "embeddings_configured": config.get("embeddings", {}),
                "embeddings_active": emb_provider,
                "llm_active": llm_provider,
                "vectordb_configured": vect_provider,
                "keys": {"gemini": gemini_key, "openai": openai_key}
            }
        }
    except Exception as e:
        logger.error(f"Failed to produce status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Basic liveness endpoint for orchestrators."""
    return JSONResponse(status_code=200, content={"status": "ok"})


@app.get("/ready")
async def readiness_check():
    """Readiness endpoint to verify app dependencies are initialized."""
    issues = []
    # Check that core components were initialized
    try:
        _ = embedder is not None
    except Exception:
        issues.append("embedder")
    try:
        _ = vector_store is not None
    except Exception:
        issues.append("vector_store")
    try:
        _ = retriever is not None
    except Exception:
        issues.append("retriever")

    if issues:
        return JSONResponse(status_code=503, content={"ready": False, "issues": issues})
    return JSONResponse(status_code=200, content={"ready": True})

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the Single-Page Dashboard HTML."""
    template_path = Path(__file__).parent / "templates" / "index.html"
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html template not found")
        
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        logger.error(f"Failed to read index.html: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error loading frontend template")

@app.get("/api/documents")
async def list_documents():
    """Endpoint to list all documents currently ingested in the database."""
    try:
        docs = vector_store.get_all_documents()
        return docs
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear")
async def clear_database():
    """Endpoint to wipe the vector database clean."""
    try:
        vector_store.clear()
        return {"status": "success", "message": "Vector store wiped clean."}
    except Exception as e:
        logger.error(f"Failed to clear database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Uploads, chunks, embeds, and indexes a document file into the vector store."""
    # Ensure temporary upload directory exists
    temp_dir = BASE_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    file_path = temp_dir / file.filename
    try:
        # Save uploaded file temporarily to disk for parsing
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Load/parse document text
        documents = load_document(str(file_path))
        if not documents:
            raise HTTPException(status_code=400, detail="No readable content found in file.")
            
        # 2. Chunk document text
        splitter = RecursiveCharacterTextSplitter()
        chunks = splitter.split_documents(documents)
        if not chunks:
            raise HTTPException(status_code=500, detail="Text splitting resulted in zero chunks.")
            
        # 3. Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks of {file.filename}...")
        texts_to_embed = [c["content"] for c in chunks]
        embeddings = embedder.embed_documents(texts_to_embed)
        
        # 4. Save to Vector DB
        logger.info(f"Saving {len(chunks)} chunks to vector store...")
        vector_store.add_documents(chunks, embeddings)
        
        return {"status": "success", "filename": file.filename, "chunk_count": len(chunks)}
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        # Clean up the temporary file
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                logger.error(f"Failed to delete temp file {file_path}: {e}")

@app.post("/api/query")
async def query_pipeline(request: QueryRequest):
    """Retrieves relevant document context matching a query, formats prompts, and queries LLM."""
    try:
        query = request.query
        top_k = request.top_k
        
        # 1. Retrieve top context chunks matching query
        retrieved = retriever.retrieve(query, top_k=top_k)
        
        # 2. Compile context
        context_str = assemble_prompt_context(retrieved)
        
        # 3. Build user QA prompt
        user_prompt = DEFAULT_QA_USER_TEMPLATE.format(context=context_str, query=query)
        
        # 4. Generate answer
        answer = llm_client.generate_response(
            system_prompt=DEFAULT_QA_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            retrieved_chunks=retrieved
        )
        
        # 5. Format return payload
        # Ensure embedding arrays aren't sent to keep payload clean
        clean_retrieved = []
        for r in retrieved:
            clean_retrieved.append({
                "content": r["content"],
                "metadata": r["metadata"],
                "score": r["score"]
            })
            
        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": clean_retrieved
        }
    except Exception as e:
        logger.error(f"Query pipeline processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
