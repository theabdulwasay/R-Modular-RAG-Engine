from typing import List, Dict, Any
from src.embeddings.embedder import Embedder
from src.vectordb.vector_store import VectorStore
from src.utils.helpers import logger

class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store
        logger.info("Initialized Retriever.")

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Retrieves top K chunks matching the query string."""
        if not query or not query.strip():
            return []
            
        logger.info(f"Retrieving matching chunks for query: '{query}'")
        try:
            # Generate embedding for the query
            query_embedding = self.embedder.embed_text(query)
            
            # Query the vector database
            results = self.vector_store.query(query_embedding, top_k=top_k)
            logger.info(f"Retrieved {len(results)} chunks.")
            return results
        except Exception as e:
            logger.error(f"Retrieval process failed: {e}")
            return []
