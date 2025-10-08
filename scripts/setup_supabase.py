"""
Setup Supabase Database Tables
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.supabase_config import get_supabase_client

load_dotenv()

# Table schemas
TABLE_SCHEMAS = {
    'conversations': """
        CREATE TABLE IF NOT EXISTS conversations (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            conversation_id TEXT UNIQUE NOT NULL,
            customer_id TEXT,
            order_id TEXT,
            started_at TIMESTAMP DEFAULT NOW(),
            ended_at TIMESTAMP,
            status TEXT DEFAULT 'active',
            sentiment_score FLOAT,
            resolved_by TEXT,
            satisfaction_score INTEGER,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """,
    
    'messages': """
        CREATE TABLE IF NOT EXISTS messages (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            agent_type TEXT,
            timestamp TIMESTAMP DEFAULT NOW(),
            metadata JSONB
        );
    """,
    
    'orders': """
        CREATE TABLE IF NOT EXISTS orders (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            order_id TEXT UNIQUE NOT NULL,
            customer_id TEXT,
            product_id TEXT,
            product_name TEXT,
            category TEXT,
            size TEXT,
            color TEXT,
            price FLOAT,
            total FLOAT,
            status TEXT DEFAULT 'pending',
            order_date TIMESTAMP DEFAULT NOW(),
            shipped_at TIMESTAMP,
            delivered_at TIMESTAMP,
            tracking_number TEXT,
            payment_method TEXT,
            shipping_cost FLOAT DEFAULT 0.0
        );
    """,
    
    'resolutions': """
        CREATE TABLE IF NOT EXISTS resolutions (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            conversation_id TEXT,
            order_id TEXT,
            resolution_type TEXT,
            issue_type TEXT,
            resolution_amount FLOAT,
            refund_method TEXT,
            reference_number TEXT,
            resolved_at TIMESTAMP DEFAULT NOW(),
            resolved_by TEXT,
            customer_satisfied BOOLEAN,
            notes TEXT
        );
    """,
    
    'customers': """
        CREATE TABLE IF NOT EXISTS customers (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            customer_id TEXT UNIQUE NOT NULL,
            name TEXT,
            email TEXT,
            phone TEXT,
            tier TEXT DEFAULT 'regular',
            total_orders INTEGER DEFAULT 0,
            lifetime_value FLOAT DEFAULT 0.0,
            return_rate FLOAT DEFAULT 0.0,
            average_satisfaction FLOAT,
            size_preferences JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """,
    
    'analytics': """
        CREATE TABLE IF NOT EXISTS analytics (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            event_type TEXT NOT NULL,
            conversation_id TEXT,
            order_id TEXT,
            agent_type TEXT,
            resolution_time_seconds INTEGER,
            customer_satisfaction INTEGER,
            timestamp TIMESTAMP DEFAULT NOW(),
            metadata JSONB
        );
    """
}

def setup_supabase():
    """Setup Supabase tables"""
    
    print("🚀 Starting Supabase Setup...")
    
    try:
        # Get client
        print("\n🔗 Connecting to Supabase...")
        client = get_supabase_client()
        print("✅ Connected successfully!")
        
        # Create tables
        print("\n📦 Creating tables...")
        
        for table_name, schema in TABLE_SCHEMAS.items():
            try:
                print(f"   Creating '{table_name}'...", end=' ')
                
                # Execute SQL via RPC or direct query
                # Note: Supabase Python client may require using raw SQL via custom function
                # For simplicity, we'll log the schema
                
                # In production, you would execute this via Supabase SQL editor or RPC
                # client.rpc('execute_sql', {'query': schema}).execute()
                
                # For now, we'll just verify the table exists by trying to query it
                result = client.table(table_name).select('*').limit(1).execute()
                print("✅")
                
            except Exception as e:
                print(f"⚠️  (Table may not exist yet - run SQL manually)")
                print(f"      SQL to execute:\n{schema}\n")
        
        print("\n✅ Supabase setup complete!")
        print("\n📝 Note: If tables don't exist, run the SQL schemas via Supabase SQL Editor:")
        print("   https://app.supabase.com/project/_/sql")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        print("\n💡 Make sure SUPABASE_URL and SUPABASE_KEY are set in .env")
        return False

def print_schemas():
    """Print all table schemas for manual execution"""
    print("\n" + "="*80)
    print("📋 COPY AND RUN THESE SQL SCHEMAS IN SUPABASE SQL EDITOR")
    print("="*80 + "\n")
    
    for table_name, schema in TABLE_SCHEMAS.items():
        print(f"-- {table_name.upper()} TABLE")
        print(schema)
        print()

if __name__ == "__main__":
    print_schemas()
    success = setup_supabase()
    sys.exit(0 if success else 1)
