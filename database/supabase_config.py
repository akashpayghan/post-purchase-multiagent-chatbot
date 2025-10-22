"""
Supabase Async Client Setup with Connection Pooling
"""

import os
from supabase import create_client, Client

# Singleton instance and connection pool
_client: Client = None

def get_supabase_client() -> Client:
    """
    Get singleton Supabase client instance
    
    Note: supabase-py client does not currently support async;
    Consider running database calls in thread pool if needed.
    """
    global _client
    if _client is None:
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_KEY')
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")
        _client = create_client(url, key)
    return _client
