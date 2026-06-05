from pydantic import BaseModel, Field, model_validator
from typing import Optional, List


class AppConfig(BaseModel):
    name: str = Field("Modular RAG System")
    host: str = Field("127.0.0.1")
    port: int = Field(8000)
    log_level: str = Field("INFO")


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(500, ge=128)
    chunk_overlap: int = Field(50, ge=0)


class VectorDBConfig(BaseModel):
    provider: str = Field("chromadb")
    persist_directory: str = Field("chroma_db")
    collection_name: str = Field("rag_documents")


class EmbeddingsConfig(BaseModel):
    provider: str = Field("gemini")
    model: str = Field("text-embedding-004")


class LLMConfig(BaseModel):
    provider: str = Field("gemini")
    model: str = Field("gemini-2.5-flash")
    temperature: float = Field(0.2, ge=0.0, le=1.0)
    max_output_tokens: int = Field(1024, ge=1)


class FullConfig(BaseModel):
    app: AppConfig = AppConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    vectordb: VectorDBConfig = VectorDBConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    llm: LLMConfig = LLMConfig()

    @model_validator(mode="after")
    def validate_providers(self):
        if self.embeddings and self.embeddings.provider not in ("gemini", "openai", "fallback"):
            raise ValueError("embeddings.provider must be one of: gemini, openai, fallback")
        if self.llm and self.llm.provider not in ("gemini", "openai", "fallback"):
            raise ValueError("llm.provider must be one of: gemini, openai, fallback")
        return self


def validate_config(raw_cfg: dict) -> FullConfig:
    """Validate and return a strongly-typed configuration object.

    Raises `pydantic.ValidationError` on invalid configuration values.
    """
    return FullConfig(**raw_cfg)
