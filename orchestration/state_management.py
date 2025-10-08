"""
State Management for Multi-Agent System
Shared memory and conversation state
"""

from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
import json

class ConversationState(TypedDict):
    """Type definition for conversation state"""
    conversation_id: str
    customer_id: Optional[str]
    order_id: Optional[str]
    messages: List[Dict[str, str]]
    current_agent: str
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    next_action: Optional[str]

class StateManager:
    """Manage conversation state across agents"""
    
    def __init__(self):
        """Initialize state manager"""
        self.states: Dict[str, ConversationState] = {}
    
    def create_state(self, conversation_id: str, 
                    customer_id: Optional[str] = None,
                    order_id: Optional[str] = None) -> ConversationState:
        """
        Create new conversation state
        
        Args:
            conversation_id: Unique conversation identifier
            customer_id: Customer identifier
            order_id: Order identifier
            
        Returns:
            Initial conversation state
        """
        state: ConversationState = {
            'conversation_id': conversation_id,
            'customer_id': customer_id,
            'order_id': order_id,
            'messages': [],
            'current_agent': 'controller',
            'context': {
                'started_at': datetime.now().isoformat(),
                'turn_count': 0,
                'issues_detected': [],
                'sentiment': 'neutral'
            },
            'metadata': {},
            'next_action': None
        }
        
        self.states[conversation_id] = state
        return state
    
    def get_state(self, conversation_id: str) -> Optional[ConversationState]:
        """Get conversation state by ID"""
        return self.states.get(conversation_id)
    
    def update_state(self, conversation_id: str, updates: Dict[str, Any]) -> ConversationState:
        """
        Update conversation state
        
        Args:
            conversation_id: Conversation identifier
            updates: Dictionary of updates
            
        Returns:
            Updated state
        """
        if conversation_id not in self.states:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        state = self.states[conversation_id]
        
        # Update top-level fields
        for key, value in updates.items():
            if key in state:
                state[key] = value
        
        return state
    
    def add_message(self, conversation_id: str, role: str, 
                   content: str, agent_type: Optional[str] = None) -> ConversationState:
        """
        Add message to conversation
        
        Args:
            conversation_id: Conversation identifier
            role: Message role (user/assistant/system)
            content: Message content
            agent_type: Type of agent that generated the message
            
        Returns:
            Updated state
        """
        state = self.states.get(conversation_id)
        if not state:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        
        if agent_type:
            message['agent_type'] = agent_type
        
        state['messages'].append(message)
        state['context']['turn_count'] += 1
        
        return state
    
    def update_context(self, conversation_id: str, 
                      context_updates: Dict[str, Any]) -> ConversationState:
        """
        Update context within state
        
        Args:
            conversation_id: Conversation identifier
            context_updates: Context updates
            
        Returns:
            Updated state
        """
        state = self.states.get(conversation_id)
        if not state:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        state['context'].update(context_updates)
        return state
    
    def set_current_agent(self, conversation_id: str, agent_name: str) -> ConversationState:
        """Set current active agent"""
        return self.update_state(conversation_id, {'current_agent': agent_name})
    
    def set_next_action(self, conversation_id: str, action: str) -> ConversationState:
        """Set next action to be taken"""
        return self.update_state(conversation_id, {'next_action': action})
    
    def get_messages(self, conversation_id: str, 
                    last_n: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get conversation messages
        
        Args:
            conversation_id: Conversation identifier
            last_n: Number of recent messages to return
            
        Returns:
            List of messages
        """
        state = self.states.get(conversation_id)
        if not state:
            return []
        
        messages = state['messages']
        
        if last_n:
            return messages[-last_n:]
        
        return messages
    
    def clear_state(self, conversation_id: str):
        """Remove conversation state"""
        if conversation_id in self.states:
            del self.states[conversation_id]
    
    def export_state(self, conversation_id: str) -> str:
        """Export state as JSON string"""
        state = self.states.get(conversation_id)
        if not state:
            return "{}"
        return json.dumps(state, indent=2)
    
    def import_state(self, state_json: str) -> ConversationState:
        """Import state from JSON string"""
        state = json.loads(state_json)
        conversation_id = state['conversation_id']
        self.states[conversation_id] = state
        return state
