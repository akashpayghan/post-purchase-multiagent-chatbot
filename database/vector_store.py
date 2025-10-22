"""
Vector Store - Pinecone Async Wrapper with Connection Pool and Retry
"""

import os
import asyncio
import logging
from typing import List, Dict, Optional, Any
from pinecone import Pinecone, Index
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import AsyncOpenAI, APIError, APITimeoutError

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Async wrapper for Pinecone vector database operations with retries and connection pooling.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.api_key = None
        self.index_name = None
        self.dimension = 1536  # Should align with model embeddings
        self.metric = 'cosine'
        
        config = config or {}
        self.api_key = config.get('pinecone_api_key', os.getenv('PINECONE_API_KEY'))
        self.index_name = config.get('pinecone_index_name', os.getenv('PINECONE_INDEX_NAME', 'ecommerce-guardian'))
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY is required for VectorStore")
        
        if not self.index_name:
            raise ValueError("PINECONE_INDEX_NAME is required for VectorStore")
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)
        
        # Use connection pool for pinecone Index client
        self.index: Optional[Index] = None
        self._connection_lock = asyncio.Lock()
        
        logger.info(f"VectorStore initialized for index {self.index_name}")

    async def _init_index(self):
        """Initialize Pinecone index with connection pooling"""
        async with self._connection_lock:
            if self.index is None:
                self.index = self.pc.Index(self.index_name)
                logger.debug(f"Pinecone index {self.index_name} client initialized")
    
    @retry(
        retry=retry_if_exception_type((APIError, APITimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5)
    )
    async def upsert_vector(self, vector_id: str, text: str, metadata: Dict[str, Any]) -> None:
        """Upsert a single vector asynchronously"""
        await self._init_index()
        
        # Compute embedding
        embedding = await self.generate_embedding(text)
        
        vector = {
            'id': vector_id,
            'values': embedding,
            'metadata': metadata
        }
        
        # Use thread to call blocking sync method (pinecone python client is sync)
        await asyncio.to_thread(self.index.upsert, vectors=[vector])
        
        logger.debug(f"Upserted vector id={vector_id}")
    
    @retry(
        retry=retry_if_exception_type((APIError, APITimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5)
    )
    async def upsert_vectors_batch(self, vectors: List[Dict[str, Any]]) -> None:
        """Upsert multiple vectors asynchronously"""
        await self._init_index()
        
        # Generate embeddings asynchronously in parallel batches
        texts = [v['text'] for v in vectors]
        embeddings = await self._generate_batch_embeddings(texts)
        
        # Construct vectors with embeddings
        vectors_with_embeddings = []
        for orig, emb in zip(vectors, embeddings):
            vectors_with_embeddings.append({
                'id': orig['id'],
                'values': emb,
                'metadata': orig.get('metadata', {})
            })
        
        await asyncio.to_thread(self.index.upsert, vectors=vectors_with_embeddings)
        logger.info(f"Upserted batch of {len(vectors)} vectors")
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors to the query
        
        Args:
            query: Text query string
            top_k: Number of results to return
            filter_dict: Optional filter dict for metadata filtering
        
        Returns:
            List of matches with id, score, metadata
        """
        await self._init_index()
        
        embedding = await self.generate_embedding(query)
        
        query_args = {
            'vector': embedding,
            'top_k': top_k,
            'include_metadata': True
        }
        if filter_dict:
            query_args['filter'] = filter_dict
        
        response = await asyncio.to_thread(self.index.query, **query_args)
        
        matches = response.get('matches', [])
        logger.debug(f"Search query returned {len(matches)} matches")
        return matches or []
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector from text using OpenAI Async client
        
        Args:
            text: Text to embed
        
        Returns:
            List of floats representing embedding
        """
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY not found")
        
        client = AsyncOpenAI(api_key=openai_key)
        
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            dimensions=1536
        )
        embedding = response.data[0].embedding
        return embedding
    
    async def _generate_batch_embeddings(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of text inputs
            batch_size: Number of texts per batch
        
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY not found")
        client = AsyncOpenAI(api_key=openai_key)
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = await client.embeddings.create(
                model="text-embedding-3-small",
                input=batch,
                dimensions=1536
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    async def get_stats(self) -> Dict[str, Any]:
        """Fetch index stats asynchronously"""
        await self._init_index()
        
        stats = await asyncio.to_thread(self.index.describe_index_stats)
        
        return stats or {}
    
    async def delete_by_id(self, vector_id: str) -> None:
        """Delete vector by ID"""
        await self._init_index()
        await asyncio.to_thread(self.index.delete, ids=[vector_id])
        logger.info(f"Deleted vector with id={vector_id}")
