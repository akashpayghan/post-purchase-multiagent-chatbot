"""
Data Models for Supabase Tables
"""

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime

@dataclass
class Conversation:
    """Conversation record model"""
    id: Optional[str] = None
    customer_id: str = None
    order_id: Optional[str] = None
    started_at: str = None
    ended_at: Optional[str] = None
    status: str = 'active'  # active, resolved, escalated
    sentiment_score: Optional[float] = None
    resolved_by: Optional[str] = None  # ai_agent or human
    satisfaction_score: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion"""
        return {k: v for k, v in asdict(self).items() if v is not None}

@dataclass
class Message:
    """Individual message model"""
    id: Optional[str] = None
    conversation_id: str = None
    role: str = None  # user, assistant, system
    content: str = None
    timestamp: str = None
    agent_type: Optional[str] = None  # controller, monitor, visual, exchange, resolution
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in asdict(self).items() if v is not None}

@dataclass
class Order:
    """Order record model"""
    id: Optional[str] = None
    order_id: str = None
    customer_id: str = None
    product_id: str = None
    product_name: str = None
    category: str = None
    size: Optional[str] = None
    color: Optional[str] = None
    price: float = None
    total: float = None
    status: str = 'pending'  # pending, shipped, delivered, returned, refunded
    order_date: str = None
    shipped_at: Optional[str] = None
    delivered_at: Optional[str] = None
    tracking_number: Optional[str] = None
    payment_method: Optional[str] = None
    shipping_cost: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in asdict(self).items() if v is not None}

@dataclass
class Resolution:
    """Resolution record model"""
    id: Optional[str] = None
    conversation_id: str = None
    order_id: str = None
    resolution_type: str = None  # refund, exchange, replacement, compensation
    issue_type: str = None  # defect, delay, wrong_item, etc.
    resolution_amount: Optional[float] = None
    refund_method: Optional[str] = None
    reference_number: str = None
    resolved_at: str = None
    resolved_by: str = None  # agent_type or human name
    customer_satisfied: Optional[bool] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in asdict(self).items() if v is not None}

@dataclass
class Customer:
    """Customer record model"""
    id: Optional[str] = None
    customer_id: str = None
    name: str = None
    email: str = None
    phone: Optional[str] = None
    tier: str = 'regular'  # regular, premium, vip
    total_orders: int = 0
    lifetime_value: float = 0.0
    return_rate: float = 0.0
    average_satisfaction: Optional[float] = None
    size_preferences: Optional[Dict[str, str]] = None
    created_at: str = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in asdict(self).items() if v is not None}

@dataclass
class Analytics:
    """Analytics event model"""
    id: Optional[str] = None
    event_type: str = None  # conversation_started, issue_resolved, escalated, etc.
    conversation_id: Optional[str] = None
    order_id: Optional[str] = None
    agent_type: Optional[str] = None
    resolution_time_seconds: Optional[int] = None
    customer_satisfaction: Optional[int] = None
    timestamp: str = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in asdict(self).items() if v is not None}

# Database Operations Helper Class
class DatabaseOperations:
    """Helper class for common database operations"""
    
    def __init__(self, supabase_client):
        self.client = supabase_client
    
    def insert_conversation(self, conversation: Conversation) -> Dict[str, Any]:
        """Insert new conversation"""
        data = conversation.to_dict()
        if not data.get('started_at'):
            data['started_at'] = datetime.now().isoformat()
        
        response = self.client.table('conversations').insert(data).execute()
        return response.data[0] if response.data else None
    
    def insert_message(self, message: Message) -> Dict[str, Any]:
        """Insert new message"""
        data = message.to_dict()
        if not data.get('timestamp'):
            data['timestamp'] = datetime.now().isoformat()
        
        response = self.client.table('messages').insert(data).execute()
        return response.data[0] if response.data else None
    
    def insert_order(self, order: Order) -> Dict[str, Any]:
        """Insert new order"""
        data = order.to_dict()
        if not data.get('order_date'):
            data['order_date'] = datetime.now().isoformat()
        
        response = self.client.table('orders').insert(data).execute()
        return response.data[0] if response.data else None
    
    def insert_resolution(self, resolution: Resolution) -> Dict[str, Any]:
        """Insert resolution record"""
        data = resolution.to_dict()
        if not data.get('resolved_at'):
            data['resolved_at'] = datetime.now().isoformat()
        
        response = self.client.table('resolutions').insert(data).execute()
        return response.data[0] if response.data else None
    
    def insert_analytics(self, analytics: Analytics) -> Dict[str, Any]:
        """Insert analytics event"""
        data = analytics.to_dict()
        if not data.get('timestamp'):
            data['timestamp'] = datetime.now().isoformat()
        
        response = self.client.table('analytics').insert(data).execute()
        return response.data[0] if response.data else None
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation by ID"""
        response = self.client.table('conversations').select('*').eq('id', conversation_id).execute()
        return response.data[0] if response.data else None
    
    def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a conversation"""
        response = self.client.table('messages').select('*').eq('conversation_id', conversation_id).order('timestamp').execute()
        return response.data if response.data else []
    
    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order by ID"""
        response = self.client.table('orders').select('*').eq('order_id', order_id).execute()
        return response.data[0] if response.data else None
    
    def get_customer_orders(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get all orders for a customer"""
        response = self.client.table('orders').select('*').eq('customer_id', customer_id).order('order_date', desc=True).execute()
        return response.data if response.data else []
    
    def update_conversation_status(self, conversation_id: str, status: str) -> bool:
        """Update conversation status"""
        response = self.client.table('conversations').update({'status': status}).eq('id', conversation_id).execute()
        return len(response.data) > 0
    
    def update_order_status(self, order_id: str, status: str) -> bool:
        """Update order status"""
        response = self.client.table('orders').update({'status': status}).eq('order_id', order_id).execute()
        return len(response.data) > 0
    
    def get_analytics(self, event_type: Optional[str] = None, 
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get analytics events with optional filters"""
        query = self.client.table('analytics').select('*')
        
        if event_type:
            query = query.eq('event_type', event_type)
        if start_date:
            query = query.gte('timestamp', start_date)
        if end_date:
            query = query.lte('timestamp', end_date)
        
        response = query.order('timestamp', desc=True).execute()
        return response.data if response.data else []
