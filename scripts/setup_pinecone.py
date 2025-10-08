"""
Initialize Pinecone Vector Database
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.pinecone_config import PineconeConfig

load_dotenv()

def setup_pinecone():
    """Initialize Pinecone index"""
    
    print("🚀 Starting Pinecone Setup...")
    
    # Get configuration
    index_name = os.getenv('PINECONE_INDEX_NAME', 'ecommerce-guardian')
    dimension = 1536  # text-embedding-3-small dimension
    metric = 'cosine'
    
    print(f"\n📝 Configuration:")
    print(f"   Index Name: {index_name}")
    print(f"   Dimension: {dimension}")
    print(f"   Metric: {metric}")
    print(f"   Model: text-embedding-3-small")
    
    try:
        # Test connection
        print("\n🔗 Testing Pinecone connection...")
        if not PineconeConfig.test_connection():
            print("❌ Connection failed. Check your PINECONE_API_KEY")
            return False
        
        print("✅ Connection successful!")
        
        # Check if index already exists
        pc = PineconeConfig.get_client()
        existing_indexes = pc.list_indexes().names()
        
        if index_name in existing_indexes:
            print(f"\n⚠️  Index '{index_name}' already exists!")
            
            # Get existing index info
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            
            # Check dimension
            print(f"\n📊 Existing Index Info:")
            print(f"   Total vectors: {stats.get('total_vector_count', 0)}")
            print(f"   Dimension: Check in Pinecone console")
            
            response = input(f"\n❓ Delete and recreate index '{index_name}'? (yes/no): ")
            
            if response.lower() == 'yes':
                print(f"\n🗑️  Deleting existing index '{index_name}'...")
                PineconeConfig.delete_index(index_name)
                print("✅ Index deleted")
                
                # Wait a moment for deletion to complete
                import time
                print("⏳ Waiting for deletion to complete...")
                time.sleep(5)
            else:
                print("\n⚠️  Keeping existing index. Make sure dimension is 1536!")
                return True
        
        # Create index
        print(f"\n📦 Creating index '{index_name}' with dimension {dimension}...")
        index = PineconeConfig.create_index(
            index_name=index_name,
            dimension=dimension,
            metric=metric
        )
        
        print(f"✅ Index '{index_name}' created successfully!")
        
        # Verify index
        print("\n🔍 Verifying index...")
        stats = index.describe_index_stats()
        print(f"✅ Index verified. Total vectors: {stats.get('total_vector_count', 0)}")
        
        print("\n✅ Pinecone setup complete!")
        print("\n💡 Next steps:")
        print("   1. Run: python scripts/generate_embeddings.py")
        print("   2. Run: python scripts/load_data.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = setup_pinecone()
    sys.exit(0 if success else 1)
