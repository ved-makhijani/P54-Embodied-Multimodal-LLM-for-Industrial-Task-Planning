"""
ollama_backend.py
-----------------
Qwen (or any Ollama model) backend.
Set in .env:
    LLM_BACKEND=ollama
    OLLAMA_MODEL=qwen2.5:7b
    OLLAMA_BASE_URL=http://ollama:11434   (or http://localhost:11434)
"""
import os
import logging

logger = logging.getLogger(__name__)

def build_llm():
    """Return a LangChain-compatible Ollama LLM instance."""
    from langchain_ollama import ChatOllama

    model    = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    logger.info(f"[Ollama backend] model={model}, base_url={base_url}")

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.0,
        format="json",          # enforce JSON output
        timeout=120,            # local models can be slower
    )