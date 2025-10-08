"""
Controller Agent - Main Orchestrator
Routes requests to specialized agents and manages conversation flow
"""

import os
import json
from typing import Dict, List, Any, Optional
from openai import OpenAI
from datetime import datetime

class ControllerAgent:
    """
    Main orchestrator that routes customer requests to specialized agents.
    Manages conversation state and coordinates multi-agent workflows.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Controller Agent
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.client = OpenAI(api_key=config.get('openai_api_key'))
        self.model = config.get('controller_model', 'gpt-4o')
        self.temperature = config.get('controller_temperature', 0.5)
        self.max_tokens = config.get('max_tokens', 1500)
        
        # Load templates and escalation rules
        self.templates = self._load_templates()
        self.escalation_rules = self._load_escalation_rules()
        
        # Available agents
        self.available_agents = [
            'monitor_agent',
            'visual_agent',
            'exchange_agent',
            'resolution_agent'
        ]
        
    def _load_templates(self) -> Dict:
        """Load conversation templates"""
        try:
            with open('data/templates/conversation_templates.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _load_escalation_rules(self) -> Dict:
        """Load escalation rules"""
        try:
            with open('data/playbooks/escalation_rules.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def route_request(self, user_message: str, conversation_history: List[Dict], 
                     context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main routing function - determines which agent should handle the request
        
        Args:
            user_message: Current user message
            conversation_history: Previous conversation messages
            context: Additional context (order info, customer data, etc.)
            
        Returns:
            Dictionary with routing decision and agent assignment
        """
        
        # Check for escalation triggers first
        escalation_check = self._check_escalation(user_message, context)
        if escalation_check['should_escalate']:
            return escalation_check
        
        # Prepare routing prompt
        routing_prompt = self._build_routing_prompt(
            user_message, 
            conversation_history, 
            context
        )
        
        # Call OpenAI to determine routing
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": routing_prompt}
                ],
                functions=self._get_routing_functions(),
                function_call="auto"
            )
            
            # Extract function call
            message = response.choices[0].message
            
            if message.function_call:
                function_name = message.function_call.name
                function_args = json.loads(message.function_call.arguments)
                
                return {
                    'agent': function_name,
                    'parameters': function_args,
                    'reasoning': function_args.get('reasoning', ''),
                    'confidence': function_args.get('confidence', 0.8)
                }
            else:
                # Default to direct response
                return {
                    'agent': 'controller_agent',
                    'response': message.content,
                    'parameters': {}
                }
                
        except Exception as e:
            print(f"Error in routing: {e}")
            return {
                'agent': 'controller_agent',
                'error': str(e),
                'fallback': True
            }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the controller"""
        return """You are the Controller Agent for an e-commerce post-purchase support system.

Your job is to analyze customer requests and route them to the appropriate specialist agent:

1. **monitor_agent**: Order tracking, delivery status, shipping delays, package location
2. **visual_agent**: Product quality issues, defects, wrong items (requires image upload)
3. **exchange_agent**: Size changes, color swaps, product recommendations, alternatives
4. **resolution_agent**: Refunds, returns, compensation, policy questions

Analyze the user's intent and select the BEST agent to handle their request.
If you can answer directly (simple questions), respond without calling an agent.

Be intelligent about routing:
- Multiple issues? Route to the most critical agent first
- Unclear request? Ask clarifying questions before routing
- Simple FAQ? Answer directly using your knowledge
"""
    
    def _build_routing_prompt(self, user_message: str, 
                             conversation_history: List[Dict],
                             context: Dict[str, Any]) -> str:
        """Build the routing prompt with context"""
        
        prompt_parts = [
            "=== CUSTOMER REQUEST ===",
            f"Message: {user_message}",
            "",
            "=== CONTEXT ===",
        ]
        
        if context.get('order_id'):
            prompt_parts.append(f"Order ID: {context['order_id']}")
        if context.get('product_name'):
            prompt_parts.append(f"Product: {context['product_name']}")
        if context.get('customer_tier'):
            prompt_parts.append(f"Customer Tier: {context['customer_tier']}")
        if context.get('order_status'):
            prompt_parts.append(f"Order Status: {context['order_status']}")
        
        if conversation_history:
            prompt_parts.append("\n=== RECENT CONVERSATION ===")
            for msg in conversation_history[-3:]:  # Last 3 messages
                prompt_parts.append(f"{msg['role']}: {msg['content']}")
        
        prompt_parts.append("\n=== TASK ===")
        prompt_parts.append("Determine which agent should handle this request, or respond directly if appropriate.")
        
        return "\n".join(prompt_parts)
    
    def _get_routing_functions(self) -> List[Dict]:
        """Define functions for routing decisions"""
        return [
            {
                "name": "monitor_agent",
                "description": "Routes to Monitor Agent for order tracking, shipping status, delivery updates, and package location inquiries",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Why this agent was selected"
                        },
                        "order_id": {
                            "type": "string",
                            "description": "Order ID to track"
                        },
                        "query_type": {
                            "type": "string",
                            "enum": ["status", "location", "delay", "delivery_date"],
                            "description": "Type of tracking query"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence in routing decision (0-1)"
                        }
                    },
                    "required": ["reasoning", "query_type"]
                }
            },
            {
                "name": "visual_agent",
                "description": "Routes to Visual Agent for product defects, wrong items, quality issues that require image analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Why this agent was selected"
                        },
                        "issue_type": {
                            "type": "string",
                            "enum": ["defect", "wrong_item", "damage", "color_mismatch", "quality_issue"],
                            "description": "Type of visual verification needed"
                        },
                        "request_image": {
                            "type": "boolean",
                            "description": "Whether to request image upload"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence in routing decision (0-1)"
                        }
                    },
                    "required": ["reasoning", "issue_type", "request_image"]
                }
            },
            {
                "name": "exchange_agent",
                "description": "Routes to Exchange Agent for size changes, color swaps, product alternatives, and recommendations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Why this agent was selected"
                        },
                        "exchange_type": {
                            "type": "string",
                            "enum": ["size", "color", "style", "similar_product"],
                            "description": "Type of exchange requested"
                        },
                        "current_product_id": {
                            "type": "string",
                            "description": "ID of current product"
                        },
                        "preference": {
                            "type": "string",
                            "description": "Customer's stated preference for new item"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence in routing decision (0-1)"
                        }
                    },
                    "required": ["reasoning", "exchange_type"]
                }
            },
            {
                "name": "resolution_agent",
                "description": "Routes to Resolution Agent for refunds, returns, compensation, policy questions, and final resolution",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Why this agent was selected"
                        },
                        "resolution_type": {
                            "type": "string",
                            "enum": ["refund", "return", "compensation", "policy_question", "cancellation"],
                            "description": "Type of resolution needed"
                        },
                        "urgency": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "Urgency level of the resolution"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence in routing decision (0-1)"
                        }
                    },
                    "required": ["reasoning", "resolution_type"]
                }
            }
        ]
    
    def _check_escalation(self, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if the request should be escalated to human agent
        
        Args:
            user_message: User's message
            context: Conversation context
            
        Returns:
            Dict with escalation decision
        """
        escalation_triggers = self.escalation_rules.get('automatic_escalation_triggers', [])
        
        user_message_lower = user_message.lower()
        
        for trigger in escalation_triggers:
            # Check keyword-based triggers
            if 'keywords' in trigger.get('condition', {}):
                keywords = trigger['condition']['keywords']
                if any(keyword in user_message_lower for keyword in keywords):
                    return {
                        'should_escalate': True,
                        'escalate_to_tier': trigger['escalate_to_tier'],
                        'reason': trigger['reason'],
                        'trigger_id': trigger['trigger_id'],
                        'agent': 'human_escalation'
                    }
            
            # Check value-based triggers
            if 'order_value_exceeds' in trigger.get('condition', {}):
                order_value = context.get('order_value', 0)
                if order_value > trigger['condition']['order_value_exceeds']:
                    return {
                        'should_escalate': True,
                        'escalate_to_tier': trigger['escalate_to_tier'],
                        'reason': trigger['reason'],
                        'trigger_id': trigger['trigger_id'],
                        'agent': 'human_escalation'
                    }
            
            # Check customer tier triggers
            if 'customer_tier' in trigger.get('condition', {}):
                customer_tier = context.get('customer_tier', 'regular')
                if customer_tier in trigger['condition']['customer_tier']:
                    # For VIP, offer choice rather than auto-escalate
                    if not trigger.get('auto_escalate', True):
                        return {
                            'should_escalate': False,
                            'offer_escalation': True,
                            'message': trigger.get('message', '')
                        }
        
        return {'should_escalate': False}
    
    def generate_response(self, message: str, context: Dict[str, Any] = None) -> str:
        """
        Generate direct response for simple queries that don't need specialist agents
        
        Args:
            message: User message
            context: Optional context
            
        Returns:
            Generated response string
        """
        try:
            messages = [
                {"role": "system", "content": self._get_response_system_prompt()},
                {"role": "user", "content": message}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=messages
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"I apologize, but I encountered an error. Please try again or let me connect you with a specialist."
    
    def _get_response_system_prompt(self) -> str:
        """System prompt for direct responses"""
        return """You are a helpful e-commerce customer support AI assistant.

Provide clear, friendly, and concise responses to customer questions.
Be empathetic and solution-focused.
If the question is complex or requires specific actions, suggest routing to a specialist.

Keep responses under 3 sentences for simple questions.
"""
    
    def format_response(self, template_id: str, variables: Dict[str, str]) -> str:
        """
        Format a response using a template
        
        Args:
            template_id: Template identifier
            variables: Variables to fill in template
            
        Returns:
            Formatted message string
        """
        templates = self.templates.get('templates', {})
        
        # Search through all template categories
        for category in templates.values():
            if isinstance(category, list):
                for template in category:
                    if template.get('template_id') == template_id:
                        message = template['message']
                        # Replace variables
                        for key, value in variables.items():
                            message = message.replace(f"{{{key}}}", str(value))
                        return message
        
        return "I'm here to help! How can I assist you today?"
    
    def log_interaction(self, interaction_data: Dict[str, Any]):
        """Log interaction for analytics and learning"""
        interaction_data['timestamp'] = datetime.now().isoformat()
        # In production, save to database
        print(f"[LOG] {json.dumps(interaction_data, indent=2)}")
