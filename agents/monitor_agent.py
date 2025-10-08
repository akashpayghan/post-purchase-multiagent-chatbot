"""
Monitor Agent - Order Tracking & Issue Detection
Handles order status, shipping tracking, delivery updates, and proactive issue detection
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from openai import OpenAI

class MonitorAgent:
    """
    Specialist agent for order monitoring, tracking, and proactive issue detection.
    Simulates tracking systems and detects potential problems before customers complain.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Monitor Agent
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.client = OpenAI(api_key=config.get('openai_api_key'))
        self.model = config.get('monitor_model', 'gpt-4o-mini')
        self.temperature = config.get('monitor_temperature', 0.3)
        
        # Load policies and triggers
        self.shipping_policy = self._load_shipping_policy()
        self.proactive_triggers = self._load_proactive_triggers()
        
    def _load_shipping_policy(self) -> Dict:
        """Load shipping policy"""
        try:
            with open('data/policies/shipping_policy.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _load_proactive_triggers(self) -> Dict:
        """Load proactive trigger rules"""
        try:
            with open('data/playbooks/proactive_triggers.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def check_order_status(self, order_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check order status and provide detailed tracking information
        
        Args:
            order_id: Order identifier
            context: Additional context (customer info, etc.)
            
        Returns:
            Dictionary with order status and tracking details
        """
        # In production, this would query real tracking API
        # For demo, simulate tracking data
        
        tracking_data = self._simulate_tracking_data(order_id, context)
        
        # Analyze for issues
        issues_detected = self._detect_tracking_issues(tracking_data)
        
        # Generate human-readable status
        status_message = self._generate_status_message(tracking_data, issues_detected)
        
        return {
            'order_id': order_id,
            'tracking_data': tracking_data,
            'issues_detected': issues_detected,
            'status_message': status_message,
            'proactive_actions': self._suggest_proactive_actions(issues_detected)
        }
    
    def _simulate_tracking_data(self, order_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Simulate tracking data (replace with real API in production)
        
        Args:
            order_id: Order ID
            context: Optional context
            
        Returns:
            Simulated tracking data
        """
        # Demo data - in production, call real carrier API
        import random
        
        statuses = [
            'order_placed',
            'processing',
            'shipped',
            'in_transit',
            'out_for_delivery',
            'delivered',
            'delivery_attempted',
            'delayed'
        ]
        
        current_status = context.get('order_status', random.choice(statuses[:4]))
        
        shipped_date = datetime.now() - timedelta(days=random.randint(1, 5))
        expected_delivery = shipped_date + timedelta(days=random.randint(5, 7))
        
        return {
            'order_id': order_id,
            'current_status': current_status,
            'tracking_number': f'TRK{order_id[-6:]}',
            'carrier': random.choice(['USPS', 'UPS', 'FedEx']),
            'shipped_date': shipped_date.isoformat(),
            'expected_delivery': expected_delivery.isoformat(),
            'current_location': random.choice([
                'Distribution Center - Chicago, IL',
                'In Transit to Local Facility',
                'Local Distribution Center',
                'Out for Delivery'
            ]),
            'tracking_events': [
                {
                    'timestamp': (shipped_date + timedelta(hours=2)).isoformat(),
                    'location': 'Warehouse',
                    'status': 'Shipment picked up'
                },
                {
                    'timestamp': (shipped_date + timedelta(days=1)).isoformat(),
                    'location': 'Distribution Hub',
                    'status': 'Arrived at facility'
                },
                {
                    'timestamp': (shipped_date + timedelta(days=2)).isoformat(),
                    'location': 'In Transit',
                    'status': 'Package in transit'
                }
            ],
            'delays': random.choice([None, {'reason': 'Weather delay', 'estimated_delay_days': 2}])
        }
    
    def _detect_tracking_issues(self, tracking_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect potential issues from tracking data
        
        Args:
            tracking_data: Tracking information
            
        Returns:
            List of detected issues
        """
        issues = []
        
        # Check for delays
        expected_delivery = datetime.fromisoformat(tracking_data['expected_delivery'])
        today = datetime.now()
        
        if expected_delivery < today and tracking_data['current_status'] != 'delivered':
            issues.append({
                'type': 'delivery_delay',
                'severity': 'high',
                'description': f"Package is {(today - expected_delivery).days} days past expected delivery",
                'action_required': True
            })
        
        # Check for stuck package
        last_update = tracking_data['tracking_events'][-1] if tracking_data['tracking_events'] else None
        if last_update:
            last_update_time = datetime.fromisoformat(last_update['timestamp'])
            hours_since_update = (today - last_update_time).total_seconds() / 3600
            
            if hours_since_update > 48:  # No update in 48 hours
                issues.append({
                    'type': 'tracking_stale',
                    'severity': 'medium',
                    'description': f"No tracking update in {int(hours_since_update)} hours",
                    'action_required': True
                })
        
        # Check for delivery attempts
        if tracking_data['current_status'] == 'delivery_attempted':
            issues.append({
                'type': 'delivery_attempted',
                'severity': 'medium',
                'description': "Delivery was attempted but customer was not available",
                'action_required': True
            })
        
        # Check for weather delays
        if tracking_data.get('delays'):
            issues.append({
                'type': 'weather_delay',
                'severity': 'low',
                'description': tracking_data['delays']['reason'],
                'action_required': False
            })
        
        return issues
    
    def _generate_status_message(self, tracking_data: Dict[str, Any], 
                                 issues: List[Dict[str, Any]]) -> str:
        """
        Generate human-readable status message
        
        Args:
            tracking_data: Tracking information
            issues: Detected issues
            
        Returns:
            Status message string
        """
        status = tracking_data['current_status']
        carrier = tracking_data['carrier']
        expected_delivery = datetime.fromisoformat(tracking_data['expected_delivery']).strftime('%B %d')
        location = tracking_data['current_location']
        
        # Build message based on status
        if status == 'delivered':
            message = f"✅ Great news! Your package was delivered."
        elif status == 'out_for_delivery':
            message = f"📦 Your package is out for delivery today with {carrier}!"
        elif status == 'in_transit':
            message = f"📦 Your package is in transit. Current location: {location}. Expected delivery: {expected_delivery}."
        elif status == 'shipped':
            message = f"📦 Your package has shipped via {carrier}. Expected delivery: {expected_delivery}."
        elif status == 'processing':
            message = f"⏳ Your order is being processed and will ship soon."
        else:
            message = f"📦 Order status: {status.replace('_', ' ').title()}"
        
        # Add issue warnings
        if issues:
            high_severity_issues = [i for i in issues if i['severity'] == 'high']
            if high_severity_issues:
                message += f"\n\n⚠️ {high_severity_issues[0]['description']}"
        
        return message
    
    def _suggest_proactive_actions(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Suggest proactive actions based on detected issues
        
        Args:
            issues: List of detected issues
            
        Returns:
            List of suggested actions
        """
        actions = []
        
        for issue in issues:
            if issue['type'] == 'delivery_delay':
                actions.append({
                    'action': 'offer_compensation',
                    'type': 'discount_code',
                    'value': '15% off next order',
                    'message': "I'd like to offer you 15% off your next order for this delay."
                })
                actions.append({
                    'action': 'offer_refund',
                    'type': 'shipping_refund',
                    'message': "I can also refund your shipping cost immediately."
                })
            
            elif issue['type'] == 'delivery_attempted':
                actions.append({
                    'action': 'reschedule_delivery',
                    'message': "Would you like me to reschedule delivery or have it held at a pickup location?"
                })
            
            elif issue['type'] == 'tracking_stale':
                actions.append({
                    'action': 'investigate',
                    'message': "Let me file a tracking investigation with the carrier to locate your package."
                })
        
        return actions
    
    def evaluate_proactive_trigger(self, order_data: Dict[str, Any], 
                                  customer_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate if proactive outreach should be triggered
        
        Args:
            order_data: Order information
            customer_data: Customer information
            
        Returns:
            Trigger data if should reach out, None otherwise
        """
        triggers = self.proactive_triggers.get('triggers', [])
        
        for trigger in triggers:
            if self._evaluate_trigger_conditions(trigger, order_data, customer_data):
                return {
                    'trigger_id': trigger['trigger_id'],
                    'trigger_name': trigger['name'],
                    'message': self._format_trigger_message(
                        trigger['message_template'],
                        order_data,
                        customer_data
                    ),
                    'priority': trigger['priority']
                }
        
        return None
    
    def _evaluate_trigger_conditions(self, trigger: Dict[str, Any],
                                    order_data: Dict[str, Any],
                                    customer_data: Dict[str, Any]) -> bool:
        """
        Evaluate if trigger conditions are met
        
        Args:
            trigger: Trigger configuration
            order_data: Order data
            customer_data: Customer data
            
        Returns:
            True if conditions met, False otherwise
        """
        conditions = trigger.get('conditions', {})
        
        # Time-based triggers
        if 'hours_after_delivery' in conditions:
            if order_data.get('status') == 'delivered':
                delivered_time = datetime.fromisoformat(order_data.get('delivered_at', datetime.now().isoformat()))
                hours_elapsed = (datetime.now() - delivered_time).total_seconds() / 3600
                if hours_elapsed >= conditions['hours_after_delivery']:
                    return True
        
        # Event-based triggers
        if 'expected_delivery_missed' in conditions:
            expected_delivery = datetime.fromisoformat(order_data.get('expected_delivery', datetime.now().isoformat()))
            if datetime.now() > expected_delivery and order_data.get('status') != 'delivered':
                delay_days = conditions.get('delay_days', 0)
                if (datetime.now() - expected_delivery).days >= delay_days:
                    return True
        
        # Pattern-based triggers
        if 'product_return_rate' in conditions:
            product_return_rate = order_data.get('product_return_rate', 0)
            threshold = float(conditions['product_return_rate'].replace('>', '').replace('%', ''))
            if product_return_rate > threshold:
                return True
        
        return False
    
    def _format_trigger_message(self, template: str, 
                               order_data: Dict[str, Any],
                               customer_data: Dict[str, Any]) -> str:
        """
        Format trigger message with actual data
        
        Args:
            template: Message template
            order_data: Order data
            customer_data: Customer data
            
        Returns:
            Formatted message
        """
        message = template
        
        # Replace placeholders
        replacements = {
            '{customer_name}': customer_data.get('name', 'there'),
            '{product_name}': order_data.get('product_name', 'order'),
            '{tracking_status}': order_data.get('tracking_status', 'in transit'),
            '{ordered_size}': order_data.get('size', 'ordered size')
        }
        
        for placeholder, value in replacements.items():
            message = message.replace(placeholder, value)
        
        return message
    
    def get_shipping_policy_info(self, query: str) -> str:
        """
        Answer shipping policy questions using RAG
        
        Args:
            query: User's question about shipping
            
        Returns:
            Answer based on shipping policy
        """
        # In production, use vector search on shipping_policy
        # For now, use OpenAI with policy as context
        
        policy_context = json.dumps(self.shipping_policy, indent=2)
        
        prompt = f"""Based on the following shipping policy, answer the customer's question clearly and concisely:

SHIPPING POLICY:
{policy_context}

CUSTOMER QUESTION:
{query}

Provide a helpful answer in 2-3 sentences."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": "You are a helpful shipping policy expert."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return "I apologize, but I'm having trouble accessing our shipping policy information right now. Would you like me to connect you with a specialist?"
