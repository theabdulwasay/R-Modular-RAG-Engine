# Prompt Templates for RAG

DEFAULT_QA_SYSTEM_PROMPT = (
    "You are an expert AI assistant that answers questions using only the provided context.\n\n"
    "Rules you must strictly follow:\n"
    "1. Base your answer ONLY on the provided context sections. Do not use external knowledge or make assumptions.\n"
    "2. If the context does not contain the answer, say exactly: 'I cannot find the answer in the provided documents.'\n"
    "3. Be concise, objective, and clear in your responses.\n"
    "4. Cite your sources. At the end of relevant sentences or at the very end of your response, indicate which source document(s) you used. "
    "Use standard brackets like [Doc: filename.pdf, Page: N] or [Doc: filename.txt] if page numbers are not available."
)

DEFAULT_QA_USER_TEMPLATE = """Use the following context snippets to answer the user's question:

---
{context}
---

Question: {query}

Helpful Answer:"""

def format_context_snippet(content: str, metadata: dict) -> str:
    """Formats a single retrieved chunk with its metadata for the prompt context."""
    filename = metadata.get("filename", "unknown_source")
    page_info = f", Page: {metadata['page']}" if "page" in metadata else ""
    return f"[Source: {filename}{page_info}]\nContent: {content}\n"

def assemble_prompt_context(retrieved_chunks: list) -> str:
    """Combines a list of retrieved chunks into a single formatted context block."""
    snippets = []
    for idx, chunk in enumerate(retrieved_chunks):
        snippet = f"--- Segment {idx + 1} ---\n" + format_context_snippet(chunk["content"], chunk["metadata"])
        snippets.append(snippet)
    return "\n".join(snippets)
