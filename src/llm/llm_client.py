from typing import List, Dict, Any
from src.utils.helpers import config, logger, get_env_var

class LLMClient:
    def __init__(self, provider: str = None, model: str = None, temperature: float = None, max_tokens: int = None):
        llm_cfg = config.get("llm", {})
        self.provider = provider or llm_cfg.get("provider", "gemini").lower()
        self.model = model or llm_cfg.get("model", "gemini-2.5-flash")
        self.temperature = temperature if temperature is not None else llm_cfg.get("temperature", 0.2)
        self.max_tokens = max_tokens if max_tokens is not None else llm_cfg.get("max_output_tokens", 1024)
        
        # Initialize clients lazily
        self._gemini_client = None
        self._openai_client = None
        
        logger.info(f"Initialized LLMClient with provider: {self.provider}, model: {self.model}")

    def _get_gemini_client(self):
        if self._gemini_client is None:
            api_key = get_env_var("GEMINI_API_KEY")
            if not api_key:
                logger.warning("GEMINI_API_KEY env variable is missing!")
                return None
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}.")
        return self._gemini_client

    def _get_openai_client(self):
        if self._openai_client is None:
            api_key = get_env_var("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY env variable is missing!")
                return None
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=api_key)
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}.")
        return self._openai_client

    def generate_response(self, system_prompt: str, user_prompt: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Generates an answer using the LLM, with fallback to offline synthesis if required."""
        active_provider = self.provider
        
        # Determine active provider
        if active_provider == "gemini":
            client = self._get_gemini_client()
            if client:
                try:
                    from google.genai import types
                    # Call generate_content in google-genai SDK
                    response = client.models.generate_content(
                        model=self.model,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=self.temperature,
                            max_output_tokens=self.max_tokens
                        )
                    )
                    if response.text:
                        return response.text
                except Exception as e:
                    logger.error(f"Gemini generation failed: {e}. Falling back to transient offline synthesis.")
                    active_provider = "fallback"
            else:
                active_provider = "fallback"

        if active_provider == "openai":
            client = self._get_openai_client()
            if client:
                try:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    logger.error(f"OpenAI generation failed: {e}. Falling back to transient offline synthesis.")
                    active_provider = "fallback"
            else:
                active_provider = "fallback"

        # Fallback Offline Extractive Synthesizer
        return self._offline_synthesize(user_prompt, retrieved_chunks)

    def _offline_synthesize(self, user_prompt: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Synthesizes an answer offline by compiling relevant quotes and details directly from retrieved snippets."""
        if not retrieved_chunks:
            return "I cannot find the answer in the provided documents."
            
        logger.info("Synthesizing response offline (no API key/connection).")
        
        lines = [
            "⚠️ **[OFFLINE SYNTHESIS MODE - NO API KEY SPECIFIED OR KEY INVALID]**",
            "I have retrieved relevant details from your uploaded documents to answer your question:",
            ""
        ]
        
        for idx, chunk in enumerate(retrieved_chunks):
            filename = chunk["metadata"].get("filename", "unknown")
            page_str = f", Page {chunk['metadata']['page']}" if "page" in chunk["metadata"] else ""
            score_percent = int(chunk["score"] * 100)
            
            lines.append(f"**Source: {filename}{page_str} (Match Confidence: {score_percent}%)**")
            lines.append(f"> {chunk['content']}")
            lines.append("")
            
        lines.append("---")
        lines.append("*Please configure a valid API key in `.env` to enable full LLM summarization.*")
        
        return "\n".join(lines)
