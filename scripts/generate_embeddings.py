"""
Generate Embeddings for All Documents
Pre-compute embeddings for faster retrieval
"""

import os
import sys
import json
from glob import glob
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.embeddings import EmbeddingGenerator

load_dotenv()

def generate_product_embeddings():
    """Generate embeddings for products"""
    print("\n📦 Generating product embeddings...")
    
    # FIXED: Add ../ to go to parent directory
    file_path = '../data/products/product_catalog.json'
    output_path = '../data/products/product_embeddings.json'
    
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return 0
    
    with open(file_path, 'r') as f:
        products = json.load(f)
    
    generator = EmbeddingGenerator()
    
    # Prepare texts
    texts = []
    for product in products:
        text = f"{product['name']} {product['description']} {' '.join(product.get('tags', []))}"
        texts.append(text)
    
    print(f"   Processing {len(texts)} products...")
    
    # Generate embeddings in batch
    embeddings = generator.generate_batch(texts, batch_size=100)
    
    # Create output structure
    output_data = []
    for product, embedding in zip(products, embeddings):
        output_data.append({
            'product_id': product['product_id'],
            'name': product['name'],
            'embedding': embedding,
            'dimension': len(embedding)
        })
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Generated {len(embeddings)} product embeddings")
    print(f"   Saved to: {output_path}")
    
    return len(embeddings)


def generate_faq_embeddings():
    """Generate embeddings for FAQs"""
    print("\n❓ Generating FAQ embeddings...")
    
    file_path = '../data/knowledge/faq.json'
    output_path = '../data/knowledge/faq_embeddings.json'
    
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return 0
    
    with open(file_path, 'r') as f:
        data = json.load(f)
        faqs = data.get('faqs', [])
    
    generator = EmbeddingGenerator()
    
    # Prepare texts
    texts = [f"{faq['question']} {faq['answer']}" for faq in faqs]
    
    print(f"   Processing {len(texts)} FAQs...")
    
    # Generate embeddings
    embeddings = generator.generate_batch(texts)
    
    # Create output
    output_data = []
    for faq, embedding in zip(faqs, embeddings):
        output_data.append({
            'question': faq['question'],
            'answer': faq['answer'],
            'category': faq.get('category', 'general'),
            'embedding': embedding,
            'dimension': len(embedding)
        })
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Generated {len(embeddings)} FAQ embeddings")
    print(f"   Saved to: {output_path}")
    
    return len(embeddings)


def generate_policy_embeddings():
    """Generate embeddings for policies"""
    print("\n📋 Generating policy embeddings...")
    
    policy_files = glob('../data/policies/*.json')
    
    if not policy_files:
        print("⚠️  No policy files found")
        return 0
    
    generator = EmbeddingGenerator()
    
    all_embeddings = []
    
    for file_path in policy_files:
        with open(file_path, 'r') as f:
            policy = json.load(f)
        
        # Convert policy to text
        text = json.dumps(policy)
        
        # Generate embedding
        embedding = generator.generate(text)
        
        all_embeddings.append({
            'file': os.path.basename(file_path),
            'policy_name': policy.get('policy_name', 'Unknown'),
            'embedding': embedding,
            'dimension': len(embedding)
        })
    
    # Save
    output_path = '../data/policies/policy_embeddings.json'
    with open(output_path, 'w') as f:
        json.dump(all_embeddings, f, indent=2)
    
    print(f"✅ Generated {len(all_embeddings)} policy embeddings")
    print(f"   Saved to: {output_path}")
    
    return len(all_embeddings)


def generate_playbook_embeddings():
    """Generate embeddings for playbooks"""
    print("\n📚 Generating playbook embeddings...")
    
    file_path = '../data/playbooks/resolution_playbooks.json'
    output_path = '../data/playbooks/playbook_embeddings.json'
    
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return 0
    
    with open(file_path, 'r') as f:
        data = json.load(f)
        playbooks = data.get('playbooks', [])
    
    generator = EmbeddingGenerator()
    
    # Prepare texts
    texts = []
    for playbook in playbooks:
        text = f"{playbook['issue_type']} {playbook['severity']} {json.dumps(playbook['steps'])}"
        texts.append(text)
    
    print(f"   Processing {len(texts)} playbooks...")
    
    # Generate embeddings
    embeddings = generator.generate_batch(texts)
    
    # Create output
    output_data = []
    for playbook, embedding in zip(playbooks, embeddings):
        output_data.append({
            'playbook_id': playbook['playbook_id'],
            'issue_type': playbook['issue_type'],
            'severity': playbook['severity'],
            'embedding': embedding,
            'dimension': len(embedding)
        })
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Generated {len(embeddings)} playbook embeddings")
    print(f"   Saved to: {output_path}")
    
    return len(embeddings)


def generate_all_embeddings():
    """Generate all embeddings"""
    
    print("🚀 Starting Embedding Generation...")
    print("="*60)
    
    total_generated = 0
    
    try:
        # Generate embeddings for each data type
        total_generated += generate_product_embeddings()
        total_generated += generate_faq_embeddings()
        total_generated += generate_policy_embeddings()
        total_generated += generate_playbook_embeddings()
        
        # Summary
        print("\n" + "="*60)
        print("📊 Generation Summary")
        print("="*60)
        print(f"\n✅ Total embeddings generated: {total_generated}")
        print(f"💾 Embedding model: text-embedding-3-small")
        print(f"📏 Dimension: 1536")
        
        print("\n✅ Embedding generation complete!")
        print("\n💡 Next step: Run 'python scripts/load_data.py' to load into Pinecone")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error generating embeddings: {e}")
        print("\n💡 Make sure OPENAI_API_KEY is set in .env")
        return False


if __name__ == "__main__":
    success = generate_all_embeddings()
    sys.exit(0 if success else 1)
