"""
Controller Agent - Main Orchestrator (Production-Ready with Async)
Routes requests to specialized agents and manages conversation flow
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import AsyncOpenAI, APIError, APITimeoutError

# Configure structured logging
logger = logging.getLogger(__name__)

class ControllerAgent:
    """
    Main orchestrator that routes customer requests to specialized agents.
    Production-ready with async, retry logic, and health monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Controller Agent
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.client = AsyncOpenAI(
            api_key=config.get('openai_api_key'),
            timeout=config.get('timeout', 30.0),
            max_retries=0  # We handle retries ourselves
        )
        self.model = config.get('controller_model', 'gpt-4o')
        self.temperature = config.get('controller_temperature', 0.5)
        self.max_tokens = config.get('max_tokens', 1500)
        self.request_timeout = config.get('request_timeout', 25.0)
        
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
        
        # Health status
        self._healthy = True
        self._last_health_check = None
        
        logger.info(f"ControllerAgent initialized with model={self.model}")
    
    def _load_templates(self) -> Dict:
        """Load conversation templates"""
        try:
            with open('data/templates/conversation_templates.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Conversation templates file not found, using defaults")
            return {}
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            return {}
    
    def _load_escalation_rules(self) -> Dict:
        """Load escalation rules"""
        try:
            with open('data/playbooks/escalation_rules.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Escalation rules file not found, using defaults")
            return {}
        except Exception as e:
            logger.error(f"Error loading escalation rules: {e}")
            return {}
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APIError, APITimeoutError, asyncio.TimeoutError)),
        reraise=True
    )
    async def route_request(
        self, 
        user_message: str, 
        conversation_history: List[Dict], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main routing function - determines which agent should handle the request
        
        Args:
            user_message: Current user message
            conversation_history: Previous conversation messages
            context: Additional context (order info, customer data, etc.)
            
        Returns:
            Dictionary with routing decision and agent assignment
        """
        start_time = datetime.now()
        
        try:
            # Check for escalation triggers first
            escalation_check = self._check_escalation(user_message, context)
            if escalation_check['should_escalate']:
                logger.info(f"Escalation triggered: {escalation_check['reason']}")
                return escalation_check
            
            # Prepare routing prompt with context summarization
            routing_prompt = self._build_routing_prompt(
                user_message, 
                conversation_history[-5:],  # Only last 5 messages
                context
            )
            
            # Call OpenAI with timeout
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        messages=[
                            {"role": "system", "content": self._get_system_prompt()},
                            {"role": "user", "content": routing_prompt}
                        ],
                        functions=self._get_routing_functions(),
                        function_call="auto"
                    ),
                    timeout=self.request_timeout
                )
            except asyncio.TimeoutError:
                logger.error(f"Routing request timed out after {self.request_timeout}s")
                raise
            
            # Extract function call
            message = response.choices[0].message
            
            if message.function_call:
                function_name = message.function_call.name
                function_args = json.loads(message.function_call.arguments)
                
                # Log routing decision
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(f"Routing decision: {function_name}, confidence: {function_args.get('confidence', 0)}, elapsed: {elapsed:.2f}s")
                
                return {
                    'agent': function_name,
                    'parameters': function_args,
                    'reasoning': function_args.get('reasoning', ''),
                    'confidence': function_args.get('confidence', 0.8),
                    'latency_ms': elapsed * 1000
                }
            else:
                # Default to direct response
                return {
                    'agent': 'controller_agent',
                    'response': message.content,
                    'parameters': {},
                    'latency_ms': (datetime.now() - start_time).total_seconds() * 1000
                }
                
        except Exception as e:
            logger.error(f"Error in routing: {type(e).__name__}: {e}", exc_info=True)
            self._healthy = False
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
- Simple FAQ? Answer directly using your knowledge"""
    
    def _build_routing_prompt(
        self, 
        user_message: str, 
        conversation_history: List[Dict],
        context: Dict[str, Any]
    ) -> str:
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
            for msg in conversation_history:
                prompt_parts.append(f"{msg['role']}: {msg['content'][:200]}")  # Truncate long messages
        
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
        """Check if the request should be escalated to human agent"""
        
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
        
        return {'should_escalate': False}
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((APIError, APITimeoutError))
    )
    async def generate_response(self, message: str, context: Dict[str, Any] = None) -> str:
        """
        Generate direct response for simple queries
        
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
            
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=messages
                ),
                timeout=self.request_timeout
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error. Please try again or let me connect you with a specialist."
    
    def _get_response_system_prompt(self) -> str:
        """System prompt for direct responses"""
        return """You are a helpful e-commerce customer support AI assistant.

Provide clear, friendly, and concise responses to customer questions.
Be empathetic and solution-focused.
If the question is complex or requires specific actions, suggest routing to a specialist.

Keep responses under 3 sentences for simple questions."""
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the controller agent
        
        Returns:
            Health status dictionary
        """
        try:
            # Test OpenAI API connectivity
            start_time = datetime.now()
            
            test_response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                ),
                timeout=10.0
            )
            
            latency = (datetime.now() - start_time).total_seconds()
            
            self._healthy = True
            self._last_health_check = datetime.now()
            
            return {
                'status': 'healthy',
                'service': 'controller_agent',
                'openai_status': 'connected',
                'latency_ms': latency * 1000,
                'last_check': self._last_health_check.isoformat()
            }
            
        except Exception as e:
            self._healthy = False
            logger.error(f"Health check failed: {e}")
            
            return {
                'status': 'unhealthy',
                'service': 'controller_agent',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
    
    def is_healthy(self) -> bool:
        """Quick health status check"""
        return self._healthy
    
    def format_response(self, template_id: str, variables: Dict[str, str]) -> str:
        """Format a response using a template"""
        templates = self.templates.get('templates', {})
        
        for category in templates.values():
            if isinstance(category, list):
                for template in category:
                    if template.get('template_id') == template_id:
                        message = template['message']
                        for key, value in variables.items():
                            message = message.replace(f"{{{key}}}", str(value))
                        return message
        
        return "I'm here to help! How can I assist you today?"
