"""
Embedding Generation Utilities
"""

import os
from typing import List, Union
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class EmbeddingGenerator:
    """Generate embeddings using OpenAI's embedding models"""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize embedding generator
        
        Args:
            model: OpenAI embedding model to use
        """
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = model
        # text-embedding-3-small produces 1536 dimensions by default
        self.dimension = 1536
    
    def generate(self, text: str) -> List[float]:
        """
        Generate embedding for single text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector of dimension 1536
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        response = self.client.embeddings.create(
            model=self.model,
            input=text.strip(),
            dimensions=1536  # Explicitly set dimension to 1536
        )
        
        return response.data[0].embedding
    
    def generate_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once
            
        Returns:
            List of embedding vectors (each 1536 dimensions)
        """
        if not texts:
            return []
        
        # Clean texts
        clean_texts = [t.strip() for t in texts if t and t.strip()]
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(clean_texts), batch_size):
            batch = clean_texts[i:i + batch_size]
            
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=1536  # Explicitly set dimension to 1536
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score (0-1)
        """
        import numpy as np
        
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        return dot_product / (norm_v1 * norm_v2)
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension
