import os
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader
from src.utils.helpers import logger

def load_text(file_path: Path) -> List[Dict[str, Any]]:
    """Loads text from a plain text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return [{
            "content": content,
            "metadata": {
                "source": str(file_path),
                "filename": file_path.name,
                "type": "txt"
            }
        }]
    except Exception as e:
        logger.error(f"Failed to load text file {file_path}: {e}")
        return []

def load_markdown(file_path: Path) -> List[Dict[str, Any]]:
    """Loads text from a Markdown file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return [{
            "content": content,
            "metadata": {
                "source": str(file_path),
                "filename": file_path.name,
                "type": "md"
            }
        }]
    except Exception as e:
        logger.error(f"Failed to load markdown file {file_path}: {e}")
        return []

def load_pdf(file_path: Path) -> List[Dict[str, Any]]:
    """Loads pages from a PDF file."""
    documents = []
    try:
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                documents.append({
                    "content": text.strip(),
                    "metadata": {
                        "source": str(file_path),
                        "filename": file_path.name,
                        "page": i + 1,
                        "type": "pdf"
                    }
                })
        logger.info(f"Loaded {len(documents)} pages from PDF: {file_path.name}")
        return documents
    except Exception as e:
        logger.error(f"Failed to load PDF file {file_path}: {e}")
        return []

def load_document(file_path: str) -> List[Dict[str, Any]]:
    """Routes the document path to the appropriate loader based on extension."""
    path = Path(file_path)
    if not path.exists():
        logger.error(f"File not found: {file_path}")
        return []
        
    ext = path.suffix.lower()
    logger.info(f"Ingesting file: {path.name} (type: {ext})")
    
    if ext == ".txt":
        return load_text(path)
    elif ext in [".md", ".markdown"]:
        return load_markdown(path)
    elif ext == ".pdf":
        return load_pdf(path)
    else:
        # Fallback to general text reading if we can
        try:
            return load_text(path)
        except Exception as e:
            logger.error(f"Unsupported file format and fallback failed for {file_path}: {e}")
            return []
