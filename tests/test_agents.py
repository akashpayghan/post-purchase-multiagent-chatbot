"""
Unit Tests for Agent System
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import ControllerAgent, MonitorAgent, VisualAgent, ExchangeAgent, ResolutionAgent

# Test configuration
TEST_CONFIG = {
    'openai_api_key': 'test-key',
    'gemini_api_key': 'test-key',
    'pinecone_api_key': 'test-key',
    'pinecone_index_name': 'test-index',
    'controller_model': 'gpt-4o',
    'monitor_model': 'gpt-4o-mini',
    'visual_model': 'gemini-1.5-pro',
    'exchange_model': 'gpt-4o',
    'resolution_model': 'gpt-4o-mini'
}

class TestControllerAgent:
    """Test Controller Agent"""
    
    def test_initialization(self):
        """Test agent initialization"""
        agent = ControllerAgent(TEST_CONFIG)
        assert agent is not None
        assert agent.model == 'gpt-4o'
        assert agent.temperature == 0.5
    
    def test_check_escalation(self):
        """Test escalation detection"""
        agent = ControllerAgent(TEST_CONFIG)
        
        # Test legal keywords
        result = agent._check_escalation("I'm going to sue you", {})
        assert result['should_escalate'] == True
        assert result['escalate_to_tier'] == 4
        
        # Test high value order
        result = agent._check_escalation("test message", {'order_value': 600})
        assert result['should_escalate'] == True
        
        # Test normal message
        result = agent._check_escalation("where is my order", {})
        assert result['should_escalate'] == False
    
    def test_format_response(self):
        """Test response formatting with templates"""
        agent = ControllerAgent(TEST_CONFIG)
        
        # Test with valid template
        response = agent.format_response('GREET001', {
            'customer_name': 'John',
            'order_id': '12345'
        })
        
        assert 'John' in response or 'here to help' in response

class TestMonitorAgent:
    """Test Monitor Agent"""
    
    def test_initialization(self):
        """Test agent initialization"""
        agent = MonitorAgent(TEST_CONFIG)
        assert agent is not None
        assert agent.model == 'gpt-4o-mini'
    
    def test_simulate_tracking_data(self):
        """Test tracking data simulation"""
        agent = MonitorAgent(TEST_CONFIG)
        
        tracking = agent._simulate_tracking_data('ORD12345', {})
        
        assert tracking['order_id'] == 'ORD12345'
        assert 'tracking_number' in tracking
        assert 'current_status' in tracking
        assert 'carrier' in tracking
    
    def test_detect_tracking_issues(self):
        """Test issue detection"""
        agent = MonitorAgent(TEST_CONFIG)
        
        # Test delivery attempt
        tracking = {
            'current_status': 'delivery_attempted',
            'expected_delivery': '2025-10-01T00:00:00',
            'tracking_events': []
        }
        
        issues = agent._detect_tracking_issues(tracking)
        assert len(issues) > 0
        assert any(i['type'] == 'delivery_attempted' for i in issues)
    
    def test_generate_status_message(self):
        """Test status message generation"""
        agent = MonitorAgent(TEST_CONFIG)
        
        tracking = {
            'current_status': 'shipped',
            'carrier': 'USPS',
            'expected_delivery': '2025-10-10T00:00:00',
            'current_location': 'Local facility'
        }
        
        message = agent._generate_status_message(tracking, [])
        assert 'shipped' in message.lower() or 'package' in message.lower()

class TestVisualAgent:
    """Test Visual Agent"""
    
    def test_initialization(self):
        """Test agent initialization"""
        agent = VisualAgent(TEST_CONFIG)
        assert agent is not None
        assert agent.max_image_size_mb == 5
    
    def test_build_analysis_prompt(self):
        """Test prompt building"""
        agent = VisualAgent(TEST_CONFIG)
        
        prompt = agent._build_analysis_prompt('defect', {'name': 'Test Product'})
        
        assert 'defect' in prompt.lower()
        assert len(prompt) > 50
    
    def test_parse_analysis_response(self):
        """Test response parsing"""
        agent = VisualAgent(TEST_CONFIG)
        
        response = """
        DEFECT PRESENT: YES
        DESCRIPTION: Seam coming apart
        SEVERITY: major
        CONFIDENCE: 95%
        """
        
        result = agent._parse_analysis_response(response, 'defect')
        
        assert result['issue_confirmed'] == True
        assert result['severity'] == 'major'
    
    def test_recommend_action(self):
        """Test action recommendation"""
        agent = VisualAgent(TEST_CONFIG)
        
        analysis = {
            'issue_confirmed': True,
            'severity': 'major'
        }
        
        action = agent._recommend_action(analysis, 'defect')
        
        assert action['action'] in ['immediate_replacement', 'replacement']
        assert action['priority'] == 'high'

class TestExchangeAgent:
    """Test Exchange Agent"""
    
    def test_initialization(self):
        """Test agent initialization"""
        agent = ExchangeAgent(TEST_CONFIG)
        assert agent is not None
        assert agent.recommendation_count == 5
    
    def test_check_exchange_eligibility(self):
        """Test exchange eligibility"""
        agent = ExchangeAgent(TEST_CONFIG)
        
        # Valid order
        order = {
            'delivered_at': '2025-10-01T00:00:00',
            'exchange_count': 0,
            'category': 'Clothing'
        }
        
        result = agent._check_exchange_eligibility(order)
        assert result['eligible'] == True
    
    def test_size_exchange(self):
        """Test size exchange processing"""
        agent = ExchangeAgent(TEST_CONFIG)
        
        order = {
            'product_name': 'T-Shirt',
            'size': 'M'
        }
        
        result = agent._process_size_exchange(order, 'L')
        
        assert result['success'] == True
        assert result['exchange_type'] == 'size'
        assert 'L' in result['message']
    
    def test_size_up_down(self):
        """Test size conversion"""
        agent = ExchangeAgent(TEST_CONFIG)
        
        assert agent._size_up('M') == 'L'
        assert agent._size_down('L') == 'M'

class TestResolutionAgent:
    """Test Resolution Agent"""
    
    def test_initialization(self):
        """Test agent initialization"""
        agent = ResolutionAgent(TEST_CONFIG)
        assert agent is not None
        assert agent.auto_approve_limit == 50
    
    def test_check_refund_eligibility(self):
        """Test refund eligibility"""
        agent = ResolutionAgent(TEST_CONFIG)
        
        # Valid refund
        order = {
            'order_date': '2025-10-01T00:00:00',
            'category': 'Clothing',
            'status': 'delivered',
            'total': 50.0
        }
        
        result = agent._check_refund_eligibility(order, {'reason': 'changed_mind'})
        assert result['eligible'] == True
    
    def test_calculate_refund_amount(self):
        """Test refund calculation"""
        agent = ResolutionAgent(TEST_CONFIG)
        
        order = {
            'total': 100.0,
            'shipping_cost': 10.0
        }
        
        # Full refund
        amount = agent._calculate_refund_amount(order, 'full_refund', {})
        assert amount == 90.0  # Excluding shipping
        
        # Partial refund
        amount = agent._calculate_refund_amount(order, 'partial_refund', {'partial_percentage': 0.5})
        assert amount == 50.0
    
    def test_offer_compensation(self):
        """Test compensation offer"""
        agent = ResolutionAgent(TEST_CONFIG)
        
        order = {'total': 100.0, 'customer_tier': 'regular'}
        
        result = agent.offer_compensation('delay', order, 'high')
        
        assert result['success'] == True
        assert 'compensation_type' in result
    
    def test_generate_return_label(self):
        """Test return label generation"""
        agent = ResolutionAgent(TEST_CONFIG)
        
        order = {
            'order_id': 'ORD12345',
            'email': 'test@example.com',
            'return_reason': 'Defective item'
        }
        
        result = agent.generate_return_label(order)
        
        assert result['success'] == True
        assert 'tracking_number' in result
        assert result['is_free'] == True  # Defective item = free return

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, '-v'])
