"""
Load Data into Pinecone Vector Database
"""

import os
import sys
import json
from glob import glob
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.vector_store import VectorStore

load_dotenv()

# Get the correct base path (parent directory of scripts/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_products():
    """Load product catalog"""
    print("\n📦 Loading products...")
    
    file_path = os.path.join(BASE_DIR, 'data', 'products', 'product_catalog.json')
    
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return 0
    
    with open(file_path, 'r') as f:
        products = json.load(f)
    
    vs = VectorStore()
    vectors = []
    
    for product in products:
        # Create searchable text
        text = f"{product['name']} {product['description']} {product['category']} {' '.join(product.get('tags', []))}"
        
        vectors.append({
            'id': f"product_{product['product_id']}",
            'text': text,
            'metadata': {
                'type': 'product',
                'product_id': product['product_id'],
                'name': product['name'],
                'category': product['category'],
                'price': product['price'],
                'description': product['description'][:500]  # Truncate for metadata limit
            }
        })
    
    # Batch upsert
    if vectors:
        vs.upsert_vectors_batch(vectors)
        print(f"✅ Loaded {len(vectors)} products")
    else:
        print("⚠️  No products to load")
    
    return len(vectors)

def load_policies():
    """Load policy documents"""
    print("\n📋 Loading policies...")
    
    policies_dir = os.path.join(BASE_DIR, 'data', 'policies', '*.json')
    policy_files = glob(policies_dir)
    
    # Exclude embedding files
    policy_files = [f for f in policy_files if 'embedding' not in f.lower()]
    
    if not policy_files:
        print(f"⚠️  No policy files found in {policies_dir}")
        return 0
    
    vs = VectorStore()
    vectors = []
    
    for idx, file_path in enumerate(policy_files):
        try:
            with open(file_path, 'r') as f:
                policy = json.load(f)
            
            # Handle if policy is a list (wrap it in a dict)
            if isinstance(policy, list):
                print(f"⚠️  {os.path.basename(file_path)} contains a list, wrapping it...")
                policy = {
                    'policy_name': os.path.basename(file_path).replace('.json', ''),
                    'items': policy
                }
            
            # Handle if policy is a dict
            policy_name = policy.get('policy_name', os.path.basename(file_path).replace('.json', ''))
            
            # Create searchable text from policy
            text = json.dumps(policy)
            
            # Truncate text if too long (Pinecone metadata has limits)
            if len(text) > 10000:
                text = text[:10000]
            
            vectors.append({
                'id': f"policy_{idx}_{os.path.basename(file_path).replace('.json', '')}",
                'text': text,
                'metadata': {
                    'type': 'policy',
                    'policy_name': policy_name,
                    'file': os.path.basename(file_path),
                    'content': text[:1000]  # Truncate for metadata
                }
            })
            
            print(f"   ✓ Loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"   ✗ Error loading {os.path.basename(file_path)}: {e}")
            continue
    
    if vectors:
        vs.upsert_vectors_batch(vectors)
        print(f"✅ Loaded {len(vectors)} policies")
    else:
        print("⚠️  No policies to load")
    
    return len(vectors)

def load_playbooks():
    """Load resolution playbooks"""
    print("\n📚 Loading playbooks...")
    
    file_path = os.path.join(BASE_DIR, 'data', 'playbooks', 'resolution_playbooks.json')
    
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return 0
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different data structures
        if isinstance(data, dict) and 'playbooks' in data:
            playbooks = data.get('playbooks', [])
        elif isinstance(data, list):
            playbooks = data
        else:
            print(f"⚠️  Unexpected data structure in {file_path}")
            return 0
        
        vs = VectorStore()
        vectors = []
        
        for playbook in playbooks:
            # Create searchable text
            text = f"{playbook['issue_type']} {playbook['severity']} {json.dumps(playbook['steps'])}"
            
            vectors.append({
                'id': f"playbook_{playbook['playbook_id']}",
                'text': text,
                'metadata': {
                    'type': 'playbook',
                    'playbook_id': playbook['playbook_id'],
                    'issue_type': playbook['issue_type'],
                    'severity': playbook['severity'],
                    'content': json.dumps(playbook)[:1000]
                }
            })
        
        if vectors:
            vs.upsert_vectors_batch(vectors)
            print(f"✅ Loaded {len(vectors)} playbooks")
        else:
            print("⚠️  No playbooks to load")
        
        return len(vectors)
        
    except Exception as e:
        print(f"❌ Error loading playbooks: {e}")
        return 0

def load_faqs():
    """Load FAQ knowledge base"""
    print("\n❓ Loading FAQs...")
    
    file_path = os.path.join(BASE_DIR, 'data', 'knowledge', 'faq.json')
    
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return 0
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different data structures
        if isinstance(data, dict) and 'faqs' in data:
            faqs = data.get('faqs', [])
        elif isinstance(data, list):
            faqs = data
        else:
            print(f"⚠️  Unexpected data structure in {file_path}")
            return 0
        
        vs = VectorStore()
        vectors = []
        
        for idx, faq in enumerate(faqs):
            # Create searchable text
            text = f"{faq['question']} {faq['answer']} {' '.join(faq.get('keywords', []))}"
            
            vectors.append({
                'id': f"faq_{idx}",
                'text': text,
                'metadata': {
                    'type': 'faq',
                    'question': faq['question'],
                    'answer': faq['answer'][:500],  # Truncate
                    'category': faq.get('category', 'general')
                }
            })
        
        if vectors:
            vs.upsert_vectors_batch(vectors)
            print(f"✅ Loaded {len(vectors)} FAQs")
        else:
            print("⚠️  No FAQs to load")
        
        return len(vectors)
        
    except Exception as e:
        print(f"❌ Error loading FAQs: {e}")
        return 0

def load_all_data():
    """Load all data into vector database"""
    
    print("🚀 Starting Data Load...")
    print("="*60)
    print(f"Base directory: {BASE_DIR}")
    print("="*60)
    
    total_loaded = 0
    
    try:
        # Load each data type
        total_loaded += load_products()
        total_loaded += load_policies()
        total_loaded += load_playbooks()
        total_loaded += load_faqs()
        
        # Get final stats
        print("\n" + "="*60)
        print("📊 Loading Statistics")
        print("="*60)
        
        vs = VectorStore()
        stats = vs.get_stats()
        
        print(f"\n✅ Total vectors loaded: {total_loaded}")
        print(f"📈 Index total: {stats.get('total_vector_count', 0)}")
        print(f"💾 Namespaces: {len(stats.get('namespaces', {}))}")
        
        print("\n✅ Data load complete!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = load_all_data()
    sys.exit(0 if success else 1)
