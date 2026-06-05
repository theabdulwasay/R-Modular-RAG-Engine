from typing import List, Dict, Any
from src.utils.helpers import config, logger

class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None, separators: List[str] = None):
        chunking_cfg = config.get("chunking", {})
        self.chunk_size = chunk_size or chunking_cfg.get("chunk_size", 500)
        self.chunk_overlap = chunk_overlap or chunking_cfg.get("chunk_overlap", 50)
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Splits raw text into a list of chunks within chunk_size constraints."""
        if not text:
            return []
        return self._split_text_recursive(text, self.separators)

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            # Force split by chunk_size if no separators left
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i:i + self.chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
            return chunks

        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split text by current separator
        splits = text.split(separator)
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # If a split is larger than chunk_size, split recursively with remaining separators
            if len(split) > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                sub_chunks = self._split_text_recursive(split, remaining_separators)
                chunks.extend(sub_chunks)
            else:
                sep_len = len(separator) if current_chunk else 0
                if len(current_chunk) + sep_len + len(split) <= self.chunk_size:
                    current_chunk += (separator if current_chunk else "") + split
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = split
                    
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return self._merge_splits(chunks)

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merges small splits into chunks closer to chunk_size and applies overlap."""
        merged_chunks = []
        current_doc = []
        total_len = 0
        
        for split in splits:
            split_len = len(split)
            # Check if adding the split exceeds chunk size
            if total_len + split_len + (len(current_doc) - 1) > self.chunk_size:
                if current_doc:
                    merged_chunks.append(" ".join(current_doc))
                
                # Form overlapping prefix
                overlap_doc = []
                overlap_len = 0
                for doc_item in reversed(current_doc):
                    if overlap_len + len(doc_item) <= self.chunk_overlap:
                        overlap_doc.insert(0, doc_item)
                        overlap_len += len(doc_item)
                    else:
                        break
                current_doc = overlap_doc
                total_len = overlap_len
                
            current_doc.append(split)
            total_len += split_len
            
        if current_doc:
            merged_chunks.append(" ".join(current_doc))
            
        return merged_chunks

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Takes a list of documents, chunks their content, and copies metadata."""
        chunked_docs = []
        for doc in documents:
            text = doc["content"]
            metadata = doc["metadata"]
            chunks = self.split_text(text)
            
            for idx, chunk_text in enumerate(chunks):
                chunk_meta = metadata.copy()
                chunk_meta["chunk_index"] = idx
                chunked_docs.append({
                    "content": chunk_text,
                    "metadata": chunk_meta
                })
        logger.info(f"Split {len(documents)} document sections into {len(chunked_docs)} chunks.")
        return chunked_docs
