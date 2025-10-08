"""
Pinecone Configuration and Setup
"""

import os
from pinecone import Pinecone, ServerlessSpec
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class PineconeConfig:
    """Pinecone vector database configuration"""
    
    _pc_instance: Optional[Pinecone] = None
    _index = None
    
    @classmethod
    def get_client(cls) -> Pinecone:
        """Get or create Pinecone client"""
        if cls._pc_instance is None:
            api_key = os.getenv('PINECONE_API_KEY')
            if not api_key:
                raise ValueError("PINECONE_API_KEY must be set in environment variables")
            
            cls._pc_instance = Pinecone(api_key=api_key)
        
        return cls._pc_instance
    
    @classmethod
    def get_index(cls):
        """Get Pinecone index"""
        if cls._index is None:
            pc = cls.get_client()
            index_name = os.getenv('PINECONE_INDEX_NAME', 'ecommerce-guardian')
            
            # Check if index exists
            if index_name not in pc.list_indexes().names():
                raise ValueError(f"Index '{index_name}' does not exist. Run setup_pinecone.py first.")
            
            cls._index = pc.Index(index_name)
        
        return cls._index
    
    @classmethod
    def create_index(cls, index_name: str, dimension: int = 1536, metric: str = 'cosine'):
        """Create new Pinecone index with correct dimension"""
        pc = cls.get_client()
        
        if index_name in pc.list_indexes().names():
            print(f"Index '{index_name}' already exists")
            return pc.Index(index_name)
        
        # Create index with correct dimension
        pc.create_index(
            name=index_name,
            dimension=dimension,  # Must match embedding model dimension
            metric=metric,
            spec=ServerlessSpec(
                cloud='aws',
                region=os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
            )
        )
        
        print(f"Created index '{index_name}' with dimension {dimension}")
        
        # Wait for index to be ready
        import time
        time.sleep(5)
        
        return pc.Index(index_name)
    
    @classmethod
    def delete_index(cls, index_name: str):
        """Delete Pinecone index"""
        pc = cls.get_client()
        if index_name in pc.list_indexes().names():
            pc.delete_index(index_name)
            print(f"Deleted index '{index_name}'")
        else:
            print(f"Index '{index_name}' does not exist")
    
    @classmethod
    def test_connection(cls) -> bool:
        """Test Pinecone connection"""
        try:
            pc = cls.get_client()
            indexes = pc.list_indexes().names()
            print(f"Connected to Pinecone. Available indexes: {indexes}")
            return True
        except Exception as e:
            print(f"Pinecone connection test failed: {e}")
            return False

# Convenience functions
def get_pinecone_client() -> Pinecone:
    """Get Pinecone client instance"""
    return PineconeConfig.get_client()

def get_pinecone_index():
    """Get Pinecone index instance"""
    return PineconeConfig.get_index()
