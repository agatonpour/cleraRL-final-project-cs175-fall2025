"""
OpenAI Embedding Provider Implementation
Handles text-to-vector conversion using OpenAI's embedding API
"""

import os
import logging
from typing import List
from openai import OpenAI
from clera_agents.reinforcement_learning import IEmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(IEmbeddingProvider):
    """
    Concrete implementation using OpenAI's text-embedding-3-small model.
    This is a Strategy Pattern implementation for embedding generation.
    """
    
    MODEL_NAME = "text-embedding-3-small"
    EMBEDDING_DIM = 1536
    
    def __init__(self, api_key: str = None):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        
        Raises:
            ValueError: If no API key is provided or found in environment
        """
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAIEmbeddingProvider with model {self.MODEL_NAME}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            ValueError: If text is empty
            Exception: If OpenAI API call fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")
        
        try:
            # Truncate very long texts to avoid API limits
            # OpenAI embedding models have 8191 token limit
            if len(text) > 30000:  # Rough character limit
                logger.warning(f"Truncating long text from {len(text)} to 30000 characters")
                text = text[:30000]
            
            response = self.client.embeddings.create(
                model=self.MODEL_NAME,
                input=text
            )
            
            embedding = response.data[0].embedding
            
            # Validate embedding dimension
            if len(embedding) != self.EMBEDDING_DIM:
                raise ValueError(f"Expected {self.EMBEDDING_DIM} dimensions, got {len(embedding)}")
            
            logger.debug(f"Generated embedding for text of length {len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Return dimensionality of embeddings"""
        return self.EMBEDDING_DIM

