"""
Supabase Configuration and Connection
"""

import os
from supabase import create_client, Client
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class SupabaseConfig:
    """Supabase database configuration and connection management"""
    
    _instance: Optional[Client] = None
    
    @classmethod
    def get_client(cls) -> Client:
        """Get or create Supabase client (Singleton pattern)"""
        if cls._instance is None:
            url = os.getenv('SUPABASE_URL')
            key = os.getenv('SUPABASE_KEY')
            
            if not url or not key:
                raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
            
            cls._instance = create_client(url, key)
        
        return cls._instance
    
    @classmethod
    def test_connection(cls) -> bool:
        """Test database connection"""
        try:
            client = cls.get_client()
            # Simple query to test connection
            response = client.table('conversations').select('*').limit(1).execute()
            return True
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False

# Convenience function
def get_supabase_client() -> Client:
    """Get Supabase client instance"""
    return SupabaseConfig.get_client()
