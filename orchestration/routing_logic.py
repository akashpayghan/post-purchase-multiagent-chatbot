"""
Agent Routing Logic
Decision-making for agent selection and task routing
"""

from typing import Dict, Any, Optional, List

class AgentRouter:
    """Route requests to appropriate agents based on intent and context"""
    
    def __init__(self):
        """Initialize router with routing rules"""
        self.agent_capabilities = {
            'controller': ['general_query', 'routing', 'greeting', 'fallback'],
            'monitor': ['order_status', 'tracking', 'shipping_delay', 'delivery_update'],
            'visual': ['defect_check', 'wrong_item', 'quality_issue', 'image_verification'],
            'exchange': ['size_change', 'color_change', 'product_recommendation', 'alternative'],
            'resolution': ['refund', 'return', 'compensation', 'policy_question']
        }
        
        self.intent_keywords = {
            'order_status': ['track', 'status', 'where is', 'shipped', 'delivery'],
            'defect': ['defect', 'broken', 'damaged', 'wrong item', 'quality'],
            'exchange': ['exchange', 'different size', 'different color', 'swap'],
            'refund': ['refund', 'money back', 'return', 'cancel'],
            'sizing': ['size', 'fit', 'too small', 'too large', 'too big']
        }
    
    def route(self, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route user message to appropriate agent
        
        Args:
            user_message: User's message
            context: Conversation context
            
        Returns:
            Routing decision with agent and confidence
        """
        # Detect intent
        intent = self._detect_intent(user_message, context)
        
        # Map intent to agent
        agent = self._map_intent_to_agent(intent, context)
        
        # Calculate confidence
        confidence = self._calculate_confidence(user_message, intent, context)
        
        return {
            'agent': agent,
            'intent': intent,
            'confidence': confidence,
            'reasoning': self._generate_reasoning(intent, agent)
        }
    
    def _detect_intent(self, message: str, context: Dict[str, Any]) -> str:
        """Detect user intent from message"""
        message_lower = message.lower()
        
        # Check for image upload context
        if context.get('image_uploaded'):
            return 'visual_verification'
        
        # Score each intent
        intent_scores = {}
        
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Return highest scoring intent
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        
        # Default intent
        return 'general_query'
    
    def _map_intent_to_agent(self, intent: str, context: Dict[str, Any]) -> str:
        """Map detected intent to appropriate agent"""
        
        intent_agent_map = {
            'order_status': 'monitor',
            'tracking': 'monitor',
            'defect': 'visual',
            'visual_verification': 'visual',
            'exchange': 'exchange',
            'sizing': 'exchange',
            'refund': 'resolution',
            'return': 'resolution',
            'policy_question': 'resolution',
            'general_query': 'controller'
        }
        
        agent = intent_agent_map.get(intent, 'controller')
        
        # Context-based overrides
        if context.get('requires_approval'):
            agent = 'resolution'
        
        if context.get('escalate'):
            agent = 'human'
        
        return agent
    
    def _calculate_confidence(self, message: str, intent: str, 
                             context: Dict[str, Any]) -> float:
        """Calculate routing confidence score"""
        base_confidence = 0.7
        
        # Increase confidence if keywords strongly match
        message_lower = message.lower()
        keywords = self.intent_keywords.get(intent, [])
        
        match_count = sum(1 for keyword in keywords if keyword in message_lower)
        
        if match_count >= 2:
            base_confidence = 0.9
        elif match_count == 1:
            base_confidence = 0.75
        
        # Increase confidence if context supports the intent
        if context.get('order_id') and intent in ['order_status', 'refund', 'exchange']:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _generate_reasoning(self, intent: str, agent: str) -> str:
        """Generate human-readable reasoning for routing decision"""
        reasoning_map = {
            ('order_status', 'monitor'): 'User asking about order tracking/status',
            ('tracking', 'monitor'): 'User needs shipping/delivery information',
            ('defect', 'visual'): 'User reporting product defect requiring verification',
            ('exchange', 'exchange'): 'User wants to exchange product',
            ('sizing', 'exchange'): 'User has sizing concerns',
            ('refund', 'resolution'): 'User requesting refund',
            ('return', 'resolution'): 'User wants to return product',
            ('general_query', 'controller'): 'General question, controller can handle'
        }
        
        return reasoning_map.get((intent, agent), f'Routing {intent} to {agent}')
    
    def should_escalate(self, context: Dict[str, Any]) -> bool:
        """
        Determine if conversation should be escalated to human
        
        Args:
            context: Conversation context
            
        Returns:
            True if escalation needed
        """
        # Check escalation triggers
        if context.get('sentiment') == 'very_negative':
            return True
        
        if context.get('turn_count', 0) > 10:  # Too many turns without resolution
            return True
        
        if context.get('legal_keywords_detected'):
            return True
        
        if context.get('high_value_order'):
            return True
        
        return False
    
    def get_next_agent(self, current_agent: str, task_result: Dict[str, Any]) -> str:
        """
        Determine next agent based on current agent's result
        
        Args:
            current_agent: Current agent name
            task_result: Result from current agent's task
            
        Returns:
            Next agent name
        """
        # Check if task completed
        if task_result.get('completed'):
            return 'end'
        
        # Agent-specific transitions
        transitions = {
            'monitor': {
                'delay_detected': 'resolution',
                'issue_found': 'resolution',
                'default': 'controller'
            },
            'visual': {
                'defect_confirmed': 'resolution',
                'exchange_needed': 'exchange',
                'default': 'controller'
            },
            'exchange': {
                'out_of_stock': 'resolution',
                'exchange_processed': 'end',
                'default': 'controller'
            },
            'resolution': {
                'refund_processed': 'end',
                'default': 'controller'
            }
        }
        
        agent_transitions = transitions.get(current_agent, {})
        
        # Check for specific conditions
        for condition, next_agent in agent_transitions.items():
            if condition in task_result and task_result[condition]:
                return next_agent
        
        # Default transition
        return agent_transitions.get('default', 'controller')
