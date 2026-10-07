"""
hello_world.py
--------------
Basic connection test for the llm_backend module.
Run this first to confirm your chosen backend is working
before running the full parser.

python -m llm_backend.hello_world

Usage:
    python hello_world.py
    LLM_BACKEND=gemini       python hello_world.py
    LLM_BACKEND=deepseek     python hello_world.py
    LLM_BACKEND=huggingface  python hello_world.py
"""

import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()


def main():
    backend = os.getenv("LLM_BACKEND", "openai").lower().strip()
    print(f"Testing backend: {backend.upper()}")
    print("-" * 40)

    # -- Model info display ----------------------------------------------------
    if backend == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
    elif backend == "gemini":
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    elif backend == "deepseek":
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    elif backend == "huggingface":
        model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    else:
        model = "unknown"

    print(f"Model    : {model}")

    # -- Load the backend ------------------------------------------------------
    if backend == "huggingface":
        print("Loading local model into memory...")
        print("(This may take 30-60 seconds on first run while weights are loaded.)")
        print("(Subsequent runs will be faster as the model is already cached.)\n")
    else:
        print("Sending test message...\n")

    try:
        from .backends import get_llm
        llm = get_llm()
    except EnvironmentError as e:
        print(f"  Setup error: {e}")
        print("  Check your .env file and make sure the API key is set.")
        sys.exit(1)
    except ImportError as e:
        print(f"  Missing dependency: {e}")
        print("  Run: pip install langchain-huggingface transformers accelerate")
        sys.exit(1)

    # -- Send test message -----------------------------------------------------
    if backend == "huggingface":
        print("Model loaded. Sending test message...\n")

    try:
        response = llm.invoke([
            HumanMessage(content=(
                "You are a robot assistant. "
                "Reply with exactly one sentence confirming you are ready."
            ))
        ])

        # HuggingFace models sometimes include the prompt in the response --
        # extract just the last sentence if needed.
        content = response.content.strip()
        if "robot assistant" in content.lower():
            # Model echoed the prompt -- take only the last non-empty line
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            content = lines[-1] if lines else content

        print(f"Response : {content}")
        print(f"\n  {backend.upper()} backend working. You're ready to use the LLM module.")

    except Exception as e:
        print(f"  Connection failed: {e}")
        if backend in ("openai", "gemini", "deepseek"):
            print("  Check your API key and internet connection.")
        else:
            print("  Check that your model downloaded correctly and you have enough RAM.")
        sys.exit(1)

if __name__ == "__main__":
    main()