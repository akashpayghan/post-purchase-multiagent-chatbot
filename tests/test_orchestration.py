"""
Unit Tests for Orchestration System
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestration import AgentRouter, StateManager, ConversationState

class TestStateManager:
    """Test State Manager"""
    
    def test_create_state(self):
        """Test state creation"""
        manager = StateManager()
        
        state = manager.create_state(
            conversation_id='test_conv_1',
            customer_id='cust_123',
            order_id='ord_456'
        )
        
        assert state['conversation_id'] == 'test_conv_1'
        assert state['customer_id'] == 'cust_123'
        assert state['order_id'] == 'ord_456'
        assert state['messages'] == []
        assert state['current_agent'] == 'controller'
    
    def test_add_message(self):
        """Test adding messages"""
        manager = StateManager()
        state = manager.create_state('test_conv_1')
        
        manager.add_message('test_conv_1', 'user', 'Hello')
        manager.add_message('test_conv_1', 'assistant', 'Hi there!')
        
        messages = manager.get_messages('test_conv_1')
        
        assert len(messages) == 2
        assert messages[0]['role'] == 'user'
        assert messages[1]['role'] == 'assistant'
    
    def test_update_context(self):
        """Test context updates"""
        manager = StateManager()
        manager.create_state('test_conv_1')
        
        manager.update_context('test_conv_1', {
            'order_id': 'ORD123',
            'sentiment': 'positive'
        })
        
        state = manager.get_state('test_conv_1')
        
        assert state['context']['order_id'] == 'ORD123'
        assert state['context']['sentiment'] == 'positive'
    
    def test_set_current_agent(self):
        """Test agent switching"""
        manager = StateManager()
        manager.create_state('test_conv_1')
        
        manager.set_current_agent('test_conv_1', 'monitor')
        
        state = manager.get_state('test_conv_1')
        assert state['current_agent'] == 'monitor'
    
    def test_get_recent_messages(self):
        """Test getting recent messages"""
        manager = StateManager()
        manager.create_state('test_conv_1')
        
        for i in range(5):
            manager.add_message('test_conv_1', 'user', f'Message {i}')
        
        recent = manager.get_messages('test_conv_1', last_n=3)
        
        assert len(recent) == 3
        assert recent[-1]['content'] == 'Message 4'
    
    def test_export_import_state(self):
        """Test state serialization"""
        manager = StateManager()
        state = manager.create_state('test_conv_1', 'cust_123')
        manager.add_message('test_conv_1', 'user', 'Test message')
        
        # Export
        exported = manager.export_state('test_conv_1')
        
        assert 'test_conv_1' in exported
        assert 'cust_123' in exported
        
        # Import to new manager
        new_manager = StateManager()
        imported_state = new_manager.import_state(exported)
        
        assert imported_state['conversation_id'] == 'test_conv_1'
        assert len(imported_state['messages']) == 1

class TestAgentRouter:
    """Test Agent Router"""
    
    def test_initialization(self):
        """Test router initialization"""
        router = AgentRouter()
        
        assert 'controller' in router.agent_capabilities
        assert 'monitor' in router.agent_capabilities
        assert len(router.intent_keywords) > 0
    
    def test_detect_intent(self):
        """Test intent detection"""
        router = AgentRouter()
        
        # Order status intent
        intent = router._detect_intent("Where is my order?", {})
        assert intent == 'order_status'
        
        # Defect intent
        intent = router._detect_intent("The item is broken", {})
        assert intent == 'defect'
        
        # Exchange intent
        intent = router._detect_intent("I need a different size", {})
        assert intent in ['exchange', 'sizing']
        
        # Refund intent
        intent = router._detect_intent("I want a refund", {})
        assert intent == 'refund'
    
    def test_map_intent_to_agent(self):
        """Test intent to agent mapping"""
        router = AgentRouter()
        
        assert router._map_intent_to_agent('order_status', {}) == 'monitor'
        assert router._map_intent_to_agent('defect', {}) == 'visual'
        assert router._map_intent_to_agent('exchange', {}) == 'exchange'
        assert router._map_intent_to_agent('refund', {}) == 'resolution'
    
    def test_calculate_confidence(self):
        """Test confidence calculation"""
        router = AgentRouter()
        
        # Strong match
        confidence = router._calculate_confidence(
            "track my order status",
            'order_status',
            {'order_id': 'ORD123'}
        )
        assert confidence >= 0.8
        
        # Weak match
        confidence = router._calculate_confidence(
            "hello",
            'general_query',
            {}
        )
        assert confidence < 0.9
    
    def test_should_escalate(self):
        """Test escalation detection"""
        router = AgentRouter()
        
        # Should escalate - negative sentiment
        assert router.should_escalate({'sentiment': 'very_negative'}) == True
        
        # Should escalate - too many turns
        assert router.should_escalate({'turn_count': 15}) == True
        
        # Should escalate - legal keywords
        assert router.should_escalate({'legal_keywords_detected': True}) == True
        
        # Should not escalate
        assert router.should_escalate({'sentiment': 'neutral', 'turn_count': 3}) == False
    
    def test_route_method(self):
        """Test main routing method"""
        router = AgentRouter()
        
        result = router.route("Where is my package?", {'order_id': 'ORD123'})
        
        assert 'agent' in result
        assert 'intent' in result
        assert 'confidence' in result
        assert result['agent'] == 'monitor'
