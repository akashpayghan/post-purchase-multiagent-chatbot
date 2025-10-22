"""
Pinecone Configuration - Async-friendly, with Health Checks and Connection Pooling
"""

import os
import asyncio
from typing import Optional
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

class PineconeConfig:
    """Pinecone DB Config and Client"""
    
    _client: Optional[Pinecone] = None
    _index = None
    
    @classmethod
    async def get_client(cls) -> Pinecone:
        if cls._client is None:
            api_key = os.getenv('PINECONE_API_KEY')
            if not api_key:
                raise RuntimeError("PINECONE_API_KEY not found in environment")
            cls._client = Pinecone(api_key=api_key)
        return cls._client
    
    @classmethod
    async def get_index(cls):
        if cls._index is None:
            client = await cls.get_client()
            index_name = os.getenv('PINECONE_INDEX_NAME', 'ecommerce-guardian')
            await asyncio.sleep(0)  # yield control
            if index_name not in client.list_indexes().names():
                raise RuntimeError(f"Index '{index_name}' does not exist, please create it first.")
            cls._index = client.Index(index_name)
        return cls._index
    
    @classmethod
    async def create_index(
        cls,
        index_name: str,
        dimension: int = 1536,
        metric: str = 'cosine'
    ):
        client = await cls.get_client()
        indexes = client.list_indexes().names()
        if index_name in indexes:
            return client.Index(index_name)
        client.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud='aws', region=os.getenv('PINECONE_ENVIRONMENT', 'us-east-1'))
        )
        # Wait for index to initialize properly
        await asyncio.sleep(10)
        return client.Index(index_name)
    
    @classmethod
    async def delete_index(cls, index_name: str):
        client = await cls.get_client()
        indexes = client.list_indexes().names()
        if index_name in indexes:
            client.delete_index(index_name)
    
    @classmethod
    async def test_connection(cls) -> bool:
        try:
            client = await cls.get_client()
            indexes = client.list_indexes().names()
            return True
        except Exception:
            return False
