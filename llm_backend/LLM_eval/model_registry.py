"""
model_registry.py
-----------------
Unified registry for all supported LLM models used by the evaluation system.

Instead of building LLMs from scratch, this module delegates to the existing
backends/ factory so there is a single source of truth for LLM construction.

Supported models (configured via .env):
    OPENAI_API_KEY   + OPENAI_MODEL    -> GPT-4o (default: gpt-4o)
    GEMINI_API_KEY   + GEMINI_MODEL    -> Gemini (default: gemini-1.5-pro)
    DEEPSEEK_API_KEY + DEEPSEEK_MODEL  -> DeepSeek (default: deepseek-chat)
"""

import os
import logging
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import PydanticOutputParser

try:
    from llm_backend.schema import ParsedInstruction
except ImportError:
    from llm_backend.schema import ParsedInstruction
# from llm_backend.prompts import build_system_prompt    # up to llm_backend
try:
    from llm_backend.prompts import build_system_prompt    # up to llm_backend
except ImportError:
    from llm_backend.prompts import build_system_prompt

load_dotenv()
logger = logging.getLogger(__name__)

ModelName = Literal["openai", "gemini", "deepseek","ollama"]

# ── All supported models ───────────────────────────────────────────────────────
AVAILABLE_MODELS: list[ModelName] = ["openai", "gemini", "deepseek","ollama"]

# ── Friendly display names for reports ────────────────────────────────────────
MODEL_DISPLAY_NAMES: dict[ModelName, str] = {
    "openai":   f"GPT-4o ({os.getenv('OPENAI_MODEL', 'gpt-4o')})",
    "gemini":   f"Gemini ({os.getenv('GEMINI_MODEL', 'gemini-1.5-pro')})",
    "deepseek": f"DeepSeek ({os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')})",
    "ollama":   f"Qwen ({os.getenv('OLLAMA_MODEL', 'qwen2.5:7b')} — Local)",

}

# ── Shared prompt + output parser ─────────────────────────────────────────────
_output_parser = PydanticOutputParser(pydantic_object=ParsedInstruction)
_system_prompt = build_system_prompt(
    format_instructions=_output_parser.get_format_instructions()
)

_prompt_template = ChatPromptTemplate.from_messages([
    # Use template_format="mustache" to avoid LangChain misinterpreting
    # the JSON braces inside the system prompt as f-string placeholders.
    SystemMessagePromptTemplate.from_template(_system_prompt, template_format="mustache"),
    HumanMessagePromptTemplate.from_template("Instruction: {instruction}"),
])

# ── Backend builder map ───────────────────────────────────────────────────────
# Each entry lazily imports from the existing backends/ package so that:
#   1. There is no duplicated LLM construction logic.
#   2. Optional dependencies (langchain-google-genai, langchain-deepseek)
#      are only imported when that backend is actually requested.

def _build_openai():
    from llm_backend.backends.openai_backend import build_llm
    return build_llm()

def _build_gemini():
    from llm_backend.backends.gemini_backend import build_llm
    return build_llm()

def _build_deepseek():
    from llm_backend.backends.deepseek_backend import build_llm
    return build_llm()

_BUILDERS = {
    "openai":   _build_openai,
    "gemini":   _build_gemini,
    "deepseek": _build_deepseek,
}

# ── API key map (for checking availability without building the LLM) ──────────
_KEY_MAP = {
    "openai":   "OPENAI_API_KEY",
    "gemini":   "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def get_chain(model_name: ModelName):
    """
    Build and return a LangChain chain for the requested model.
    The chain accepts {"instruction": str} and returns a ParsedInstruction.

    Args:
        model_name: One of "openai", "gemini", "deepseek"

    Returns:
        A runnable LangChain chain.

    Raises:
        ValueError:        If model_name is not recognised.
        EnvironmentError:  If the required API key is missing from .env.
        ImportError:       If a required package is not installed.
    """
    if model_name not in _BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {AVAILABLE_MODELS}"
        )
    logger.debug(f"Building chain for model: {model_name}")
    llm = _BUILDERS[model_name]()
    return _prompt_template | llm | _output_parser


def get_available_models() -> list[ModelName]:
    """
    Return only the models whose API keys are present in the environment.
    Safe to call without raising — models with missing keys are silently skipped.
    """
    return [m for m in AVAILABLE_MODELS if os.getenv(_KEY_MAP[m])]
