import json
import uuid
import math
from pathlib import Path
from typing import List, Dict, Any
from src.utils.helpers import config, logger, BASE_DIR

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("chromadb package not available. VectorStore will fall back to JSON-persisted in-memory database.")

class VectorStore:
    def __init__(self, provider: str = None, persist_dir: str = None, collection_name: str = None):
        vdb_cfg = config.get("vectordb", {})
        self.provider = provider or vdb_cfg.get("provider", "chromadb").lower()
        self.collection_name = collection_name or vdb_cfg.get("collection_name", "rag_documents")
        
        # Determine persistence path
        raw_persist_dir = persist_dir or vdb_cfg.get("persist_directory", "chroma_db")
        self.persist_path = BASE_DIR / raw_persist_dir
        self.persist_path.mkdir(exist_ok=True)

        self.use_fallback = not CHROMA_AVAILABLE or self.provider == "fallback"
        
        # ChromaDB setup
        self._chroma_client = None
        self._chroma_collection = None
        
        # Fallback setup
        self.fallback_file = self.persist_path / "fallback_store.json"
        self._fallback_docs = []
        
        if self.use_fallback:
            logger.info(f"Using JSON-persisted Fallback Vector Store at {self.fallback_file}")
            self._load_fallback_db()
        else:
            logger.info(f"Using ChromaDB Vector Store at {self.persist_path}")
            self._init_chroma()

    def _init_chroma(self):
        try:
            self._chroma_client = chromadb.PersistentClient(path=str(self.persist_path))
            # Create or get collection. We set cosine space for similarity search
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}. Switching to fallback database.")
            self.use_fallback = True
            self._load_fallback_db()

    def _load_fallback_db(self):
        if self.fallback_file.exists():
            try:
                with open(self.fallback_file, "r", encoding="utf-8") as f:
                    self._fallback_docs = json.load(f)
                logger.info(f"Loaded {len(self._fallback_docs)} chunks from fallback database.")
            except Exception as e:
                logger.error(f"Error loading fallback database: {e}. Initializing empty.")
                self._fallback_docs = []
        else:
            self._fallback_docs = []

    def _save_fallback_db(self):
        try:
            with open(self.fallback_file, "w", encoding="utf-8") as f:
                json.dump(self._fallback_docs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save fallback database: {e}")

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Adds documents (content + metadata) and their corresponding embeddings to the vector store."""
        if not documents or not embeddings:
            return

        if len(documents) != len(embeddings):
            raise ValueError("The number of documents must match the number of embeddings.")

        if self.use_fallback:
            for doc, emb in zip(documents, embeddings):
                # Ensure metadata values are basic serializable types
                clean_meta = {}
                for k, v in doc["metadata"].items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_meta[k] = v
                    else:
                        clean_meta[k] = str(v)
                
                # Check for duplicates or generate unique ID
                doc_id = str(uuid.uuid4())
                self._fallback_docs.append({
                    "id": doc_id,
                    "content": doc["content"],
                    "metadata": clean_meta,
                    "embedding": emb
                })
            self._save_fallback_db()
            logger.info(f"Added {len(documents)} document chunks to Fallback DB.")
        else:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
            contents = [doc["content"] for doc in documents]
            
            # Clean metadatas for ChromaDB (no nested lists/dicts allowed)
            metadatas = []
            for doc in documents:
                meta = {}
                for k, v in doc["metadata"].items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                    else:
                        meta[k] = str(v)
                metadatas.append(meta)

            self._chroma_collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=contents
            )
            logger.info(f"Added {len(documents)} document chunks to ChromaDB collection '{self.collection_name}'.")

    def query(self, query_embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        """Queries the vector store for the top K closest documents to the query embedding."""
        if self.use_fallback:
            if not self._fallback_docs:
                return []

            q_norm = math.sqrt(sum(x*x for x in query_embedding))
            if q_norm == 0:
                return []

            results = []
            for doc in self._fallback_docs:
                doc_emb = doc["embedding"]
                doc_norm = math.sqrt(sum(x*x for x in doc_emb))
                
                if doc_norm == 0:
                    similarity = 0.0
                else:
                    dot_product = sum(x*y for x, y in zip(query_embedding, doc_emb))
                    similarity = dot_product / (q_norm * doc_norm)
                
                results.append({
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": similarity
                })
            
            # Sort by similarity score descending
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        else:
            try:
                results = self._chroma_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )
                
                formatted_results = []
                if results and "documents" in results and results["documents"]:
                    # Chroma returns list of lists since we passed query_embeddings as a list
                    docs = results["documents"][0]
                    metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
                    distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)
                    
                    for doc_text, meta, dist in zip(docs, metadatas, distances):
                        # Convert distance to similarity score: cosine distance is 1 - cosine similarity
                        # Therefore similarity = 1 - distance
                        score = max(0.0, min(1.0, 1.0 - dist))
                        formatted_results.append({
                            "content": doc_text,
                            "metadata": meta,
                            "score": score
                        })
                return formatted_results
            except Exception as e:
                logger.error(f"ChromaDB query failed: {e}. Falling back to in-memory matching if documents exist.")
                return []

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Returns metadata for all documents stored in the database."""
        if self.use_fallback:
            # Group by source filename
            files = {}
            for doc in self._fallback_docs:
                fname = doc["metadata"].get("filename", "unknown")
                source = doc["metadata"].get("source", "unknown")
                ftype = doc["metadata"].get("type", "unknown")
                if fname not in files:
                    files[fname] = {
                        "filename": fname,
                        "source": source,
                        "type": ftype,
                        "chunk_count": 0
                    }
                files[fname]["chunk_count"] += 1
            return list(files.values())
        else:
            try:
                # Get all documents in the collection
                all_data = self._chroma_collection.get()
                files = {}
                if all_data and "metadatas" in all_data and all_data["metadatas"]:
                    for meta in all_data["metadatas"]:
                        fname = meta.get("filename", "unknown")
                        source = meta.get("source", "unknown")
                        ftype = meta.get("type", "unknown")
                        if fname not in files:
                            files[fname] = {
                                "filename": fname,
                                "source": source,
                                "type": ftype,
                                "chunk_count": 0
                            }
                        files[fname]["chunk_count"] += 1
                return list(files.values())
            except Exception as e:
                logger.error(f"Failed to fetch document list from ChromaDB: {e}")
                return []

    def clear(self):
        """Clears all entries from the vector store."""
        if self.use_fallback:
            self._fallback_docs = []
            if self.fallback_file.exists():
                try:
                    self.fallback_file.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete fallback DB file: {e}")
            logger.info("Fallback database cleared.")
        else:
            try:
                # Delete collection and recreate it to clear all documents
                self._chroma_client.delete_collection(self.collection_name)
                self._chroma_collection = self._chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("ChromaDB database cleared.")
            except Exception as e:
                logger.error(f"Failed to clear ChromaDB: {e}")
