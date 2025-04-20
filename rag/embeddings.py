"""
Embedding functions for the RAG chatbot.
"""

import logging
from config.settings import DEFAULT_EMBEDDING_MODEL
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)

def get_embedding_function(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """
    Return a LangChain Embeddings instance for the requested model.
    Handles:
      • Nomic‑AI `nomic-embed-text-*`
      • Sentence‑Transformers  (e.g. all‑mpnet‑base‑v2)
      • Anything else Hugging Face supports
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        # ---------- Nomic‑AI models -------------------------------------------------
        if model_name.startswith("nomic-ai/nomic-embed-text"):
            # ✓ correct repo‑id, ✓ trust_remote_code so ST loads Nomic’s custom class
            model_kwargs = {
                "device": "mps",              # "cuda" / "cpu" depending on your box
                "trust_remote_code": True,
            }
            encode_kwargs = {
                "normalize_embeddings": True,  # for cosine similarity
                "batch_size": 64,              # tune to your VRAM / RAM
            }
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                cache_folder="./model_cache",
            )

        # ---------- Typical Sentence‑Transformers models ---------------------------
        elif "all-mpnet-base-v2" in model_name:
            model_kwargs = {"device": "cpu"}   # or "cuda"
            encode_kwargs = {
                "normalize_embeddings": True,
                "batch_size": 16,
            }
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                cache_folder="./model_cache",
            )

        # ---------- Fallback -------------------------------------------------------
        else:
            embeddings = HuggingFaceEmbeddings(model_name=model_name)

        logger.info("Embedding function using %s created successfully", model_name)
        return embeddings

    except Exception as e:
        logger.error("Error creating embedding function: %s", e)
        raise
