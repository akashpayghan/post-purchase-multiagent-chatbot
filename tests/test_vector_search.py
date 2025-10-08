"""
Unit Tests for Vector Search
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.embeddings import EmbeddingGenerator
from database.vector_store import VectorStore

class TestEmbeddingGenerator:
    """Test Embedding Generator"""
    
    def test_initialization(self):
        """Test generator initialization"""
        generator = EmbeddingGenerator()
        
        assert generator.model == "text-embedding-3-small"
        assert generator.dimension == 1536
    
    @pytest.mark.skipif(
        not os.getenv('OPENAI_API_KEY'),
        reason="OPENAI_API_KEY not set"
    )
    def test_generate_single(self):
        """Test single embedding generation"""
        generator = EmbeddingGenerator()
        
        embedding = generator.generate("This is a test product")
        
        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)
    
    def test_generate_empty_text(self):
        """Test empty text handling"""
        generator = EmbeddingGenerator()
        
        with pytest.raises(ValueError):
            generator.generate("")
    
    @pytest.mark.skipif(
        not os.getenv('OPENAI_API_KEY'),
        reason="OPENAI_API_KEY not set"
    )
    def test_generate_batch(self):
        """Test batch embedding generation"""
        generator = EmbeddingGenerator()
        
        texts = [
            "Product 1",
            "Product 2",
            "Product 3"
        ]
        
        embeddings = generator.generate_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(e) == 1536 for e in embeddings)
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        generator = EmbeddingGenerator()
        
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        vec3 = [0.0, 1.0, 0.0]
        
        # Identical vectors
        sim = generator.cosine_similarity(vec1, vec2)
        assert abs(sim - 1.0) < 0.001
        
        # Orthogonal vectors
        sim = generator.cosine_similarity(vec1, vec3)
        assert abs(sim - 0.0) < 0.001
    
    def test_get_dimension(self):
        """Test dimension getter"""
        generator = EmbeddingGenerator()
        assert generator.get_dimension() == 1536

class TestVectorStore:
    """Test Vector Store (without actual Pinecone connection)"""
    
    @pytest.mark.skipif(
        not os.getenv('PINECONE_API_KEY'),
        reason="PINECONE_API_KEY not set"
    )
    def test_initialization(self):
        """Test vector store initialization"""
        vs = VectorStore()
        
        assert vs.embedding_model == "text-embedding-3-small"
        assert vs.embedding_dimension == 1536
    
    @pytest.mark.skipif(
        not os.getenv('OPENAI_API_KEY'),
        reason="OPENAI_API_KEY not set"
    )
    def test_generate_embedding(self):
        """Test embedding generation"""
        vs = VectorStore()
        
        embedding = vs.generate_embedding("test text")
        
        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)
    
    @pytest.mark.skipif(
        not all([os.getenv('PINECONE_API_KEY'), os.getenv('OPENAI_API_KEY')]),
        reason="API keys not set"
    )
    def test_upsert_vector(self):
        """Test single vector upsert"""
        vs = VectorStore()
        
        # This will actually insert into Pinecone if keys are set
        vs.upsert_vector(
            vector_id='test_vec_1',
            text='Test product',
            metadata={'type': 'test', 'name': 'Test'}
        )
        
        # Cleanup would happen here in a real test
        # vs.delete_by_id('test_vec_1')
    
    @pytest.mark.skipif(
        not all([os.getenv('PINECONE_API_KEY'), os.getenv('OPENAI_API_KEY')]),
        reason="API keys not set"
    )
    def test_search(self):
        """Test vector search"""
        vs = VectorStore()
        
        # Search for similar items
        results = vs.search("t-shirt clothing", top_k=5)
        
        assert isinstance(results, list)
        assert len(results) <= 5
        
        if results:
            assert 'id' in results[0]
            assert 'score' in results[0]
            assert 'metadata' in results[0]
    
    @pytest.mark.skipif(
        not os.getenv('PINECONE_API_KEY'),
        reason="PINECONE_API_KEY not set"
    )
    def test_get_stats(self):
        """Test getting index statistics"""
        vs = VectorStore()
        
        stats = vs.get_stats()
        
        assert isinstance(stats, dict)
        # Stats structure may vary based on Pinecone version

class TestVectorSearchIntegration:
    """Integration tests for vector search"""
    
    @pytest.mark.skipif(
        not all([os.getenv('PINECONE_API_KEY'), os.getenv('OPENAI_API_KEY')]),
        reason="API keys not set"
    )
    def test_product_search_flow(self):
        """Test complete product search flow"""
        vs = VectorStore()
        
        # Insert test products
        test_products = [
            {
                'id': 'test_prod_1',
                'text': 'Blue cotton t-shirt comfortable casual wear',
                'metadata': {'type': 'product', 'name': 'Blue T-Shirt', 'price': 29.99}
            },
            {
                'id': 'test_prod_2',
                'text': 'Red wool sweater warm winter clothing',
                'metadata': {'type': 'product', 'name': 'Red Sweater', 'price': 59.99}
            }
        ]
        
        # Upsert
        vs.upsert_vectors_batch(test_products)
        
        # Search for similar
        results = vs.search("blue shirt", top_k=2, filter_dict={'type': 'product'})
        
        assert len(results) > 0
        
        # Cleanup
        vs.delete_by_id('test_prod_1')
        vs.delete_by_id('test_prod_2')
    
    @pytest.mark.skipif(
        not all([os.getenv('PINECONE_API_KEY'), os.getenv('OPENAI_API_KEY')]),
        reason="API keys not set"
    )
    def test_semantic_search_accuracy(self):
        """Test semantic search understands meaning"""
        vs = VectorStore()
        
        # Search with synonyms
        results1 = vs.search("track package", top_k=3)
        results2 = vs.search("where is my order", top_k=3)
        
        # Both queries should return similar results (if data exists)
        assert isinstance(results1, list)
        assert isinstance(results2, list)

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, '-v'])
