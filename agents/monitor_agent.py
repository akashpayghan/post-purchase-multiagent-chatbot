"""
Monitor Agent - Order Tracking & Proactive Issue Detection (Production-Ready)
Handles order status, tracking, delivery monitoring, and proactive alerts
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import AsyncOpenAI, APIError, APITimeoutError
import random
import json

logger = logging.getLogger(__name__)

class MonitorAgent:
    """
    Monitor Agent for order tracking, shipping status, and proactive issue detection.
    Production-ready with async operations, retry logic, and health monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Monitor Agent"""
        self.client = AsyncOpenAI(
            api_key=config.get('openai_api_key'),
            timeout=config.get('timeout', 30.0),
            max_retries=0
        )
        self.model = config.get('monitor_model', 'gpt-4o-mini')
        self.temperature = config.get('monitor_temperature', 0.3)
        self.request_timeout = config.get('request_timeout', 20.0)
        
        # Load proactive triggers
        self.proactive_triggers = self._load_proactive_triggers()
        
        # Health status
        self._healthy = True
        self._last_health_check = None
        
        logger.info(f"MonitorAgent initialized with model={self.model}")
    
    def _load_proactive_triggers(self) -> Dict:
        """Load proactive monitoring triggers"""
        try:
            with open('data/playbooks/proactive_triggers.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Proactive triggers file not found, using defaults")
            return {}
        except Exception as e:
            logger.error(f"Error loading proactive triggers: {e}")
            return {}
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APIError, APITimeoutError, asyncio.TimeoutError))
    )
    async def check_order_status(
        self, 
        order_id: str, 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Check order status and detect issues proactively
        
        Args:
            order_id: Order identifier
            context: Additional context
            
        Returns:
            Order status with tracking info and detected issues
        """
        start_time = datetime.now()
        context = context or {}
        
        try:
            # Simulate fetching tracking data (replace with real API in production)
            tracking_data = await self._fetch_tracking_data(order_id, context)
            
            # Detect issues proactively
            issues = await self._detect_tracking_issues(tracking_data)
            
            # Generate customer-friendly status message
            status_message = await self._generate_status_message(tracking_data, issues)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Order status checked: {order_id}, issues: {len(issues)}, elapsed: {elapsed:.2f}s")
            
            return {
                'success': True,
                'order_id': order_id,
                'tracking_data': tracking_data,
                'issues_detected': issues,
                'status_message': status_message,
                'latency_ms': elapsed * 1000
            }
            
        except Exception as e:
            logger.error(f"Error checking order status for {order_id}: {e}", exc_info=True)
            self._healthy = False
            
            return {
                'success': False,
                'order_id': order_id,
                'error': str(e),
                'status_message': f"I'm having trouble accessing the tracking information for order #{order_id}. Please try again in a moment."
            }
    
    async def _fetch_tracking_data(self, order_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch tracking data from carrier/shipping system
        
        In production, replace with actual API calls to:
        - Shopify/WooCommerce order API
        - Carrier APIs (USPS, FedEx, UPS, DHL)
        - Internal warehouse management system
        """
        # Simulate API delay
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # Simulate tracking data (replace with real API)
        return self._simulate_tracking_data(order_id, context)
    
    def _simulate_tracking_data(self, order_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate simulated tracking data for demo purposes"""
        
        statuses = ['pending', 'processing', 'shipped', 'in_transit', 'out_for_delivery', 
                   'delivered', 'delayed', 'delivery_attempted']
        
        current_status = random.choice(statuses[2:7])  # More likely to be shipped/in transit
        
        order_date = datetime.now() - timedelta(days=random.randint(1, 5))
        shipped_date = order_date + timedelta(days=1) if current_status != 'pending' else None
        expected_delivery = datetime.now() + timedelta(days=random.randint(1, 3))
        
        carriers = ['USPS', 'FedEx', 'UPS', 'DHL']
        carrier = random.choice(carriers)
        
        tracking_number = f"{carrier[:3].upper()}{random.randint(1000000000000, 9999999999999)}"
        
        # Generate tracking events
        events = []
        if current_status in ['shipped', 'in_transit', 'out_for_delivery', 'delivered']:
            events.append({
                'timestamp': (order_date + timedelta(days=1)).isoformat(),
                'status': 'Picked up by carrier',
                'location': 'Origin Facility'
            })
            events.append({
                'timestamp': (order_date + timedelta(days=2)).isoformat(),
                'status': 'In transit',
                'location': 'Sorting Facility'
            })
            if current_status in ['out_for_delivery', 'delivered']:
                events.append({
                    'timestamp': datetime.now().isoformat(),
                    'status': 'At local facility',
                    'location': 'Local Distribution Center'
                })
        
        return {
            'order_id': order_id,
            'current_status': current_status,
            'carrier': carrier,
            'tracking_number': tracking_number,
            'order_date': order_date.isoformat(),
            'shipped_at': shipped_date.isoformat() if shipped_date else None,
            'expected_delivery': expected_delivery.isoformat(),
            'current_location': 'Local Distribution Center' if current_status == 'in_transit' else None,
            'tracking_events': events,
            'delivery_instructions': context.get('delivery_instructions'),
            'customer_email': context.get('customer_email')
        }
    
    async def _detect_tracking_issues(self, tracking_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Proactively detect issues with order/delivery
        
        Args:
            tracking_data: Tracking information
            
        Returns:
            List of detected issues with severity and recommended actions
        """
        issues = []
        current_status = tracking_data.get('current_status')
        
        # Check for delivery delays
        if current_status == 'delayed':
            issues.append({
                'type': 'delivery_delay',
                'severity': 'medium',
                'message': 'Your order is experiencing a delay.',
                'action': 'offer_compensation',
                'compensation_amount': 10.00
            })
        
        # Check for delivery attempts
        if current_status == 'delivery_attempted':
            issues.append({
                'type': 'delivery_attempted',
                'severity': 'high',
                'message': 'Delivery was attempted but unsuccessful.',
                'action': 'provide_redelivery_options',
                'requires_customer_action': True
            })
        
        # Check for stuck packages
        expected_delivery = datetime.fromisoformat(tracking_data.get('expected_delivery'))
        if datetime.now() > expected_delivery and current_status not in ['delivered', 'out_for_delivery']:
            issues.append({
                'type': 'package_stuck',
                'severity': 'high',
                'message': 'Your package appears to be delayed beyond the expected delivery date.',
                'action': 'proactive_investigation',
                'days_overdue': (datetime.now() - expected_delivery).days
            })
        
        # Check for missing tracking updates
        events = tracking_data.get('tracking_events', [])
        if events:
            last_event_time = datetime.fromisoformat(events[-1]['timestamp'])
            hours_since_update = (datetime.now() - last_event_time).total_seconds() / 3600
            
            if hours_since_update > 48 and current_status not in ['delivered']:
                issues.append({
                    'type': 'no_tracking_updates',
                    'severity': 'medium',
                    'message': 'No tracking updates in 48+ hours.',
                    'action': 'carrier_inquiry',
                    'hours_since_update': int(hours_since_update)
                })
        
        return issues
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5)
    )
    async def _generate_status_message(
        self, 
        tracking_data: Dict[str, Any], 
        issues: List[Dict[str, Any]]
    ) -> str:
        """
        Generate customer-friendly status message using LLM
        
        Args:
            tracking_data: Tracking information
            issues: Detected issues
            
        Returns:
            Formatted status message
        """
        try:
            # Build prompt
            prompt = self._build_status_prompt(tracking_data, issues)
            
            # Generate message
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=500,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ]
                ),
                timeout=self.request_timeout
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating status message: {e}")
            # Fallback to template-based message
            return self._fallback_status_message(tracking_data, issues)
    
    def _get_system_prompt(self) -> str:
        """System prompt for status message generation"""
        return """You are a helpful e-commerce order tracking assistant.

Generate clear, friendly, and informative order status messages.

Guidelines:
- Start with the current status (e.g., "Your order is in transit!")
- Include tracking number and carrier
- Provide expected delivery date
- If there are issues, acknowledge them with empathy and provide solutions
- Keep the tone positive and reassuring
- Use emojis sparingly (📦, 🚚, ✅)
- Include next steps or actions the customer should take
- Format with clear sections and bullet points for easy reading"""
    
    def _build_status_prompt(self, tracking_data: Dict[str, Any], issues: List[Dict[str, Any]]) -> str:
        """Build prompt for status message generation"""
        
        prompt_parts = [
            "=== ORDER TRACKING DATA ===",
            f"Order ID: {tracking_data.get('order_id')}",
            f"Status: {tracking_data.get('current_status')}",
            f"Carrier: {tracking_data.get('carrier')}",
            f"Tracking Number: {tracking_data.get('tracking_number')}",
            f"Expected Delivery: {tracking_data.get('expected_delivery')}",
        ]
        
        if tracking_data.get('current_location'):
            prompt_parts.append(f"Current Location: {tracking_data['current_location']}")
        
        if issues:
            prompt_parts.append("\n=== ISSUES DETECTED ===")
            for issue in issues:
                prompt_parts.append(f"- {issue['type']}: {issue['message']} (Severity: {issue['severity']})")
        
        prompt_parts.append("\n=== TASK ===")
        prompt_parts.append("Generate a customer-friendly order status message that:")
        prompt_parts.append("1. Clearly communicates the current status")
        prompt_parts.append("2. Addresses any issues with empathy")
        prompt_parts.append("3. Provides actionable next steps")
        prompt_parts.append("4. Maintains a positive, helpful tone")
        
        return "\n".join(prompt_parts)
    
    def _fallback_status_message(self, tracking_data: Dict[str, Any], issues: List[Dict[str, Any]]) -> str:
        """Generate fallback status message without LLM"""
        
        order_id = tracking_data.get('order_id')
        status = tracking_data.get('current_status', 'unknown')
        carrier = tracking_data.get('carrier', 'carrier')
        tracking_number = tracking_data.get('tracking_number', 'N/A')
        expected_delivery = tracking_data.get('expected_delivery', 'soon')
        
        status_messages = {
            'pending': '⏳ Your order is being prepared',
            'processing': '📦 Your order is being processed',
            'shipped': '🚚 Your order has been shipped',
            'in_transit': '🚚 Your order is on the way',
            'out_for_delivery': '📬 Your order is out for delivery today',
            'delivered': '✅ Your order has been delivered',
            'delayed': '⚠️ Your order is experiencing a delay'
        }
        
        message = f"{status_messages.get(status, 'Your order status')}\n\n"
        message += f"**Order:** #{order_id}\n"
        message += f"**Carrier:** {carrier}\n"
        message += f"**Tracking:** {tracking_number}\n"
        message += f"**Expected Delivery:** {expected_delivery}\n"
        
        if issues:
            message += f"\n⚠️ **Note:** {issues[0]['message']}\n"
        
        return message
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            start_time = datetime.now()
            
            # Test OpenAI API
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
                'service': 'monitor_agent',
                'openai_status': 'connected',
                'latency_ms': latency * 1000,
                'last_check': self._last_health_check.isoformat()
            }
            
        except Exception as e:
            self._healthy = False
            logger.error(f"Health check failed: {e}")
            
            return {
                'status': 'unhealthy',
                'service': 'monitor_agent',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
    
    def is_healthy(self) -> bool:
        """Quick health status check"""
        return self._healthy
