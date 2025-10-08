"""
Vector Store Operations
Handles embedding generation, storage, and retrieval
"""

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from .pinecone_config import get_pinecone_index
from dotenv import load_dotenv

load_dotenv()

class VectorStore:
    """Vector database operations for embeddings and semantic search"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.index = get_pinecone_index()
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimension = 1536
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def upsert_vector(self, vector_id: str, text: str, metadata: Dict[str, Any]):
        """
        Insert or update a single vector
        
        Args:
            vector_id: Unique identifier for the vector
            text: Text to embed and store
            metadata: Associated metadata
        """
        embedding = self.generate_embedding(text)
        
        self.index.upsert(vectors=[{
            'id': vector_id,
            'values': embedding,
            'metadata': metadata
        }])
    
    def upsert_vectors_batch(self, vectors: List[Dict[str, Any]]):
        """
        Insert or update multiple vectors in batch
        
        Args:
            vectors: List of dicts with 'id', 'text', and 'metadata'
        """
        # Generate embeddings
        texts = [v['text'] for v in vectors]
        embeddings = self.generate_embeddings_batch(texts)
        
        # Prepare vectors for upsert
        upsert_data = []
        for i, vector in enumerate(vectors):
            upsert_data.append({
                'id': vector['id'],
                'values': embeddings[i],
                'metadata': vector['metadata']
            })
        
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(upsert_data), batch_size):
            batch = upsert_data[i:i + batch_size]
            self.index.upsert(vectors=batch)
    
    def search(self, query: str, top_k: int = 5, 
               filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Semantic search for similar vectors
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filter_dict: Metadata filters
            
        Returns:
            List of matching results with metadata
        """
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Search
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Format results
        matches = []
        for match in results['matches']:
            matches.append({
                'id': match['id'],
                'score': match['score'],
                'metadata': match.get('metadata', {})
            })
        
        return matches
    
    def search_by_vector(self, vector: List[float], top_k: int = 5,
                        filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search using pre-computed embedding vector
        
        Args:
            vector: Embedding vector
            top_k: Number of results
            filter_dict: Metadata filters
            
        Returns:
            List of matching results
        """
        results = self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        matches = []
        for match in results['matches']:
            matches.append({
                'id': match['id'],
                'score': match['score'],
                'metadata': match.get('metadata', {})
            })
        
        return matches
    
    def get_by_id(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch vector by ID
        
        Args:
            vector_id: Vector identifier
            
        Returns:
            Vector data or None
        """
        try:
            result = self.index.fetch(ids=[vector_id])
            if vector_id in result['vectors']:
                return result['vectors'][vector_id]
            return None
        except Exception as e:
            print(f"Error fetching vector {vector_id}: {e}")
            return None
    
    def delete_by_id(self, vector_id: str):
        """Delete vector by ID"""
        self.index.delete(ids=[vector_id])
    
    def delete_by_filter(self, filter_dict: Dict[str, Any]):
        """Delete vectors matching filter"""
        self.index.delete(filter=filter_dict)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return self.index.describe_index_stats()
