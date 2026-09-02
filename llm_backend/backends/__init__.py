"""
backends/__init__.py
--------------------
Factory that returns the correct LLM based on the LLM_BACKEND env var.

Usage:
    from llm_backend.backends import get_llm
    llm = get_llm()                    # reads LLM_BACKEND from environment

    from llm_backend.backends import get_llm_by_name
    llm = get_llm_by_name("openai")    # explicit model name (used by eval)
"""

import os

SUPPORTED_BACKENDS = ("openai", "gemini", "deepseek", "ollama")


def get_llm_by_name(backend: str):
    """
    Return an initialised LangChain LLM for the given backend name.

    Args:
        backend: One of "openai", "gemini", "deepseek", "ollama"

    Returns:
        A LangChain-compatible LLM instance.

    Raises:
        ValueError: If the backend name is not recognised.
    """
    backend = backend.lower().strip()

    if backend == "openai":
        from .openai_backend import build_llm
    elif backend == "gemini":
        from .gemini_backend import build_llm
    elif backend == "deepseek":
        from .deepseek_backend import build_llm
    elif backend == "ollama":
        from .ollama_backend import build_llm
    else:
        raise ValueError(
            f"Unsupported backend '{backend}'. "
            f"Choose one of: {SUPPORTED_BACKENDS}"
        )

    return build_llm()


def get_llm():
    """
    Return an initialised LangChain LLM based on the LLM_BACKEND env var.

    LLM_BACKEND=openai       -> openai_backend.build_llm()      (default)
    LLM_BACKEND=gemini       -> gemini_backend.build_llm()
    LLM_BACKEND=deepseek     -> deepseek_backend.build_llm()
    LLM_BACKEND=ollama       -> ollama_backend.build_llm() (fully local)

    Raises:
        ValueError: If LLM_BACKEND is set to an unsupported value.
    """
    backend = os.getenv("LLM_BACKEND", "ollama")
    return get_llm_by_name(backend)
