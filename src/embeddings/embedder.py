import math
import hashlib
from typing import List
from src.utils.helpers import config, logger, get_env_var

class Embedder:
    def __init__(self, provider: str = None, model: str = None):
        emb_cfg = config.get("embeddings", {})
        self.provider = provider or emb_cfg.get("provider", "gemini").lower()
        self.model = model or emb_cfg.get("model", "text-embedding-004")
        
        # Initialize clients lazily
        self._gemini_client = None
        self._openai_client = None
        
        # If user selected Gemini but key missing, prefer OpenAI if available
        if self.provider == "gemini":
            if not get_env_var("GEMINI_API_KEY") and get_env_var("OPENAI_API_KEY"):
                logger.warning("GEMINI_API_KEY not found; switching embedder provider to openai (fallback preference).")
                self.provider = "openai"

        logger.info(f"Initialized Embedder with provider: {self.provider}, model: {self.model}")

    def _get_gemini_client(self):
        if self._gemini_client is None:
            api_key = get_env_var("GEMINI_API_KEY")
            if not api_key:
                logger.warning("GEMINI_API_KEY env variable is missing! Falling back to offline embeddings.")
                self.provider = "fallback"
                return None
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Gemini Client: {e}. Falling back to offline embeddings.")
                self.provider = "fallback"
        return self._gemini_client

    def _get_openai_client(self):
        if self._openai_client is None:
            api_key = get_env_var("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY env variable is missing! Falling back to offline embeddings.")
                self.provider = "fallback"
                return None
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=api_key)
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI Client: {e}. Falling back to offline embeddings.")
                self.provider = "fallback"
        return self._openai_client

    def embed_text(self, text: str) -> List[float]:
        """Generates embeddings for a single text block."""
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of text blocks."""
        if not texts:
            return []

        active_provider = self.provider

        # Route based on provider
        if active_provider == "gemini":
            client = self._get_gemini_client()
            if client:
                try:
                    # google-genai Client API: client.models.embed_content
                    response = client.models.embed_content(
                        model=self.model,
                        contents=texts
                    )
                    # For list output, retrieve vectors
                    if hasattr(response, 'embeddings'):
                        return [emb.values for emb in response.embeddings]
                    elif hasattr(response, 'embedding'):
                        return [response.embedding.values]
                except Exception as e:
                    logger.error(f"Gemini embedding generation failed: {e}. Falling back to transient offline embeddings.")
                    active_provider = "fallback"

        if active_provider == "openai":
            client = self._get_openai_client()
            if client:
                try:
                    response = client.embeddings.create(
                        model=self.model,
                        input=texts
                    )
                    return [data.embedding for data in response.data]
                except Exception as e:
                    logger.error(f"OpenAI embedding generation failed: {e}. Falling back to transient offline embeddings.")
                    active_provider = "fallback"

        # Fallback Offline Feature Hashing Embeddings
        # Uses standard dimension of 1536 to match OpenAI models, or 768 for Gemini.
        dimension = 768 if "gemini" in self.model or self.model == "text-embedding-004" else 1536
        return [self._generate_hashing_embedding(text, dimension) for text in texts]

    def _generate_hashing_embedding(self, text: str, dimension: int) -> List[float]:
        """Generates a deterministic vector representation of text via hashing trick."""
        vector = [0.0] * dimension
        words = text.lower().split()
        if not words:
            return vector
            
        for word in words:
            # Hash words deterministically
            h = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            index = h % dimension
            vector[index] += 1.0
            
        # L2 Normalization
        norm = math.sqrt(sum(x*x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]
            
        return vector
