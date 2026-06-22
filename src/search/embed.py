"""Embedding generation using sentence-transformers."""

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

_model = None


def get_model(model_name: str = "BAAI/bge-small-en-v1.5"):
    """Get or initialize the embedding model.

    Args:
        model_name: Name of the sentence-transformers model.

    Returns:
        SentenceTransformer model instance.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")
    return _model


def generate_embedding(text: str, model_name: str = "BAAI/bge-small-en-v1.5") -> Optional[np.ndarray]:
    """Generate embedding for a single text.

    Args:
        text: Input text to embed.
        model_name: Name of the embedding model.

    Returns:
        Embedding vector as numpy array or None if failed.
    """
    try:
        model = get_model(model_name)
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None


def generate_embeddings_batch(
    texts: list[str],
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 32
) -> Optional[np.ndarray]:
    """Generate embeddings for a batch of texts.

    Args:
        texts: List of input texts to embed.
        model_name: Name of the embedding model.
        batch_size: Number of texts to process at once.

    Returns:
        Array of embedding vectors or None if failed.
    """
    try:
        model = get_model(model_name)
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True
        )
        logger.info(f"Generated embeddings for {len(texts)} texts")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        return None
