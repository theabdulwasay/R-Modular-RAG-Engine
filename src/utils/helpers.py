import os
import yaml
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from dotenv import load_dotenv
from src.utils.config import validate_config
from pydantic import ValidationError

# Base Directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Default Config in case config.yaml is missing
DEFAULT_CONFIG = {
    "app": {"name": "Modular RAG System", "host": "127.0.0.1", "port": 8000, "log_level": "INFO"},
    "chunking": {"chunk_size": 500, "chunk_overlap": 50},
    "vectordb": {"provider": "chromadb", "persist_directory": "chroma_db", "collection_name": "rag_documents"},
    "embeddings": {"provider": "gemini", "model": "text-embedding-004"},
    "llm": {"provider": "gemini", "model": "gemini-2.5-flash", "temperature": 0.2, "max_output_tokens": 1024}
}

def load_config() -> dict:
    """Loads configuration from config.yaml, falling back to defaults."""
    config_path = BASE_DIR / "config.yaml"
    if not config_path.exists():
        return DEFAULT_CONFIG
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or DEFAULT_CONFIG
            try:
                # Validate and coerce types via pydantic models
                validated = validate_config(raw)
                # Return plain dict for backward compatibility
                return validated.dict()
            except ValidationError as ve:
                print(f"Config validation error: {ve}. Using default configuration.")
                return DEFAULT_CONFIG
    except Exception as e:
        print(f"Error loading config.yaml: {e}. Using default configurations.")
        return DEFAULT_CONFIG

# Load config once
config = load_config()

def setup_logger(name: str = "rag_app") -> logging.Logger:
    """Sets up a logger that outputs to both a file and standard output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured
    log_level_str = config.get("app", {}).get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    # Create logs directory if not exists
    logs_dir = BASE_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Use a simple ISO time formatter; keep message concise for file and console
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Rotating file handler (daily rolls, keep 14 days)
    file_handler = TimedRotatingFileHandler(
        filename=str(logs_dir / "app.log"), when="D", interval=1, backupCount=14, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

# Shared logger instance
logger = setup_logger()

def get_env_var(key: str, default: str = None) -> str:
    """Gets environment variable, with fallback."""
    return os.getenv(key, default)


def enforce_required_keys(strict: bool = False):
    """Checks that required API keys are present for configured providers.

    If `strict` is True, the function will raise SystemExit on missing keys.
    Otherwise it logs warnings and allows fallbacks.
    """
    emb_provider = config.get("embeddings", {}).get("provider", "fallback")
    llm_provider = config.get("llm", {}).get("provider", "fallback")

    missing = []
    if emb_provider == "gemini" and not get_env_var("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY (embeddings)")
    if emb_provider == "openai" and not get_env_var("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY (embeddings)")

    if llm_provider == "gemini" and not get_env_var("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY (llm)")
    if llm_provider == "openai" and not get_env_var("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY (llm)")

    if missing:
        if strict:
            # Fail fast without printing secrets
            raise SystemExit(f"Missing required environment keys: {', '.join(missing)}")
        else:
            logger.warning(f"Missing optional environment keys (will fallback if possible): {', '.join(missing)}")


# Enforce required keys only if STRICT_KEYS=true in environment (default: false)
try:
    STRICT = os.getenv("STRICT_KEYS", "false").lower() in ("1", "true", "yes")
    enforce_required_keys(strict=STRICT)
except SystemExit:
    # Re-raise to stop startup when strict mode is requested
    raise
except Exception as e:
    # Log and continue — do not expose secrets
    print(f"Secret check failed: {e}")
