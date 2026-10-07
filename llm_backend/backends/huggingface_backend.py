"""
backends/huggingface_backend.py
-------------------------------
Runs a local LLM entirely on your machine using HuggingFace Transformers
and LangChain's HuggingFacePipeline. No API key or internet connection
required after the model is downloaded.

Install dependencies:
    pip install langchain-huggingface transformers accelerate bitsandbytes

Required env vars (in .env):
    HF_MODEL        -- HuggingFace model repo ID (see recommended models below)
    HF_DEVICE       -- device to run on: "auto", "cpu", "cuda", "mps" (default: auto)
    HF_MAX_TOKENS   -- max new tokens to generate (default: 512)
    HF_LOAD_IN_4BIT -- "true" to enable 4-bit quantization, reduces VRAM (default: false)
    HF_TOKEN        -- only needed for gated models (Llama, Gemma)
    
--------------------------------------  ------------------------------------
RECOMMENDED MODELS  (all work natively with HuggingFace Transformers)
--------------------------------------------------------------------------

-- DeepSeek R1 Distill (reasoning, open-weight) --------------------------
    deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B   1.1 GB   2 GB RAM  testing only
    deepseek-ai/DeepSeek-R1-Distill-Qwen-7B     4.7 GB   8 GB RAM  minimum viable
    deepseek-ai/DeepSeek-R1-Distill-Qwen-14B    9.0 GB  16 GB RAM  recommended
    deepseek-ai/DeepSeek-R1-Distill-Qwen-32B   20.0 GB  32 GB RAM  best quality
    deepseek-ai/DeepSeek-R1-Distill-Llama-8B    4.9 GB   8 GB RAM  Llama-based alt

-- Qwen 3 (Alibaba, strong instruction following) ------------------------
    Qwen/Qwen3-0.6B          0.4 GB   2 GB RAM  ultra lightweight
    Qwen/Qwen3-1.7B          1.1 GB   3 GB RAM  testing
    Qwen/Qwen3-4B            2.6 GB   6 GB RAM  good balance
    Qwen/Qwen3-8B            5.2 GB   8 GB RAM  recommended for laptops
    Qwen/Qwen3-14B           9.5 GB  16 GB RAM  high quality
    Qwen/Qwen3-32B          20.0 GB  32 GB RAM  near-frontier quality

-- Qwen 2.5 (stable, widely tested) -------------------------------------
    Qwen/Qwen2.5-1.5B-Instruct   1.0 GB   2 GB RAM
    Qwen/Qwen2.5-7B-Instruct     4.7 GB   8 GB RAM  good default
    Qwen/Qwen2.5-14B-Instruct    9.0 GB  16 GB RAM
    Qwen/Qwen2.5-32B-Instruct   20.0 GB  32 GB RAM

-- Llama 3 (Meta, widely supported) -------------------------------------
    meta-llama/Llama-3.2-1B-Instruct    0.7 GB   2 GB RAM  ultra fast
    meta-llama/Llama-3.2-3B-Instruct    2.0 GB   4 GB RAM  good for testing
    meta-llama/Llama-3.1-8B-Instruct    4.9 GB   8 GB RAM  recommended
    meta-llama/Llama-3.1-70B-Instruct  40.0 GB  80 GB RAM  large setup only
    NOTE: Llama models require accepting Meta's license on HuggingFace first.
          Visit huggingface.co/meta-llama and click "Agree and access repository".
          Then set HF_TOKEN in your .env.

-- Mistral (strong, efficient) -------------------------------------------
    mistralai/Mistral-7B-Instruct-v0.3   4.1 GB   8 GB RAM  solid default
    mistralai/Mistral-Nemo-Instruct       7.0 GB  12 GB RAM  12B, very capable
    mistralai/Mixtral-8x7B-Instruct      26.0 GB  48 GB RAM  MoE, high quality

-- Gemma 3 (Google, open-weight) -----------------------------------------
    google/gemma-3-1b-it    0.7 GB   2 GB RAM  tiny, fast
    google/gemma-3-4b-it    2.5 GB   6 GB RAM  good balance
    google/gemma-3-12b-it   7.5 GB  14 GB RAM  recommended
    google/gemma-3-27b-it  17.0 GB  32 GB RAM  high quality
    NOTE: Gemma models require accepting Google's license on HuggingFace.

-- Microsoft Phi-4 (small but capable) -----------------------------------
    microsoft/Phi-4-mini-instruct   2.3 GB   5 GB RAM  punches above its weight
    microsoft/phi-4                 9.1 GB  16 GB RAM  strong reasoning

--------------------------------------------------------------------------
QUICK GUIDE: which model to pick?
    Low RAM  (< 8 GB):  Qwen/Qwen3-4B  or  Qwen/Qwen2.5-7B-Instruct
    8 GB RAM:           Qwen/Qwen3-8B  or  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    16 GB RAM:          Qwen/Qwen3-14B or  deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
    32 GB RAM:          Qwen/Qwen3-32B or  mistralai/Mistral-Nemo-Instruct
--------------------------------------------------------------------------

First run:
    Model weights are downloaded from HuggingFace Hub and cached locally
    in ~/.cache/huggingface/. Subsequent runs load from cache with no internet.

Privacy:
    Once cached, all inference is 100% local. No data leaves your machine.
    Satisfies the Australian Privacy Principles requirement for sensitive data.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Default -- good balance of quality and RAM for most laptops
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def build_llm():
    """
    Return a LangChain-compatible ChatHuggingFace instance running locally.

    Wraps HuggingFacePipeline in ChatHuggingFace so it exposes the same
    chat interface (SystemMessage / HumanMessage) as ChatOpenAI,
    ChatDeepSeek, and ChatGoogleGenerativeAI -- keeping custom_LLM_parser.py
    fully backend-agnostic.
    """
    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
    except ImportError as e:
        raise ImportError(
            f"Missing dependency for HuggingFace backend: {e}\n"
        ) from e

    model_id     = os.getenv("HF_MODEL", DEFAULT_MODEL)
    device       = os.getenv("HF_DEVICE", "auto")
    max_tokens   = int(os.getenv("HF_MAX_TOKENS", "512"))
    load_in_4bit = os.getenv("HF_LOAD_IN_4BIT", "false").lower() == "true"
    hf_token     = os.getenv("HF_TOKEN", None)  # required for gated models (Llama, Gemma)

    logger.info(f"[HuggingFace backend] model={model_id}")
    logger.info(f"[HuggingFace backend] device={device}, max_new_tokens={max_tokens}, 4bit={load_in_4bit}")
    logger.info("[HuggingFace backend] Loading model -- first run may take a while...")

    # -- Quantization config (optional, reduces VRAM by ~50-75%) --------------
    quantization_config = None
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("[HuggingFace backend] 4-bit quantization enabled")
        except ImportError:
            logger.warning(
                "[HuggingFace backend] bitsandbytes not installed -- "
                "falling back to full precision. "
                "Install with: pip install bitsandbytes"
            )

    # -- Tokenizer -------------------------------------------------------------
    tokenizer_kwargs = {"trust_remote_code": True}
    if hf_token:
        tokenizer_kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)

    # -- Model -----------------------------------------------------------------
    # Use float16 on GPU (faster, less VRAM), float32 on CPU (float16 not
    # supported on most CPUs and causes warnings or errors).
    is_cpu_only = device == "cpu" or (
        device == "auto" and not torch.cuda.is_available()
        and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    )
    dtype = torch.float32 if is_cpu_only else torch.float16

    if is_cpu_only:
        logger.info("[HuggingFace backend] No GPU detected -- running on CPU with float32")
        logger.info("[HuggingFace backend] For faster inference use a smaller model (1.5B or 4B)")

    model_kwargs = {
        "pretrained_model_name_or_path": model_id,
        "device_map": device,
        "trust_remote_code": True,
    }
    if hf_token:
        model_kwargs["token"] = hf_token
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    # -- HuggingFace pipeline --------------------------------------------------
    hf_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,
        do_sample=False,         # deterministic (equivalent to temperature=0)
        return_full_text=False,  # return only the generated text, not the prompt
        pad_token_id=tokenizer.eos_token_id,
    )

    # -- LangChain wrapper -----------------------------------------------------
    lc_pipeline = HuggingFacePipeline(pipeline=hf_pipeline)
    chat_model  = ChatHuggingFace(llm=lc_pipeline)

    logger.info(f"[HuggingFace backend] Model loaded: {model_id}")
    return chat_model