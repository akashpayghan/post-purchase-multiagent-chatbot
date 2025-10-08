"""
Resolution Agent - Refund/Exchange Processing
Handles refunds, returns, compensation, and final resolution of customer issues
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from openai import OpenAI

class ResolutionAgent:
    """
    Specialist agent for final resolution of customer issues.
    Handles refunds, returns, compensation, and policy enforcement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Resolution Agent"""
        self.client = OpenAI(api_key=config.get('openai_api_key'))
        self.model = config.get('resolution_model', 'gpt-4o-mini')
        self.temperature = config.get('resolution_temperature', 0.3)
        
        # Load policies
        self.refund_policy = self._load_refund_policy()
        self.return_policy = self._load_return_policy()
        
        # Settings
        self.auto_approve_limit = config.get('auto_approve_limit_usd', 50)
        
    def _load_refund_policy(self) -> Dict:
        """Load refund policy"""
        try:
            with open('data/policies/refund_policy.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _load_return_policy(self) -> Dict:
        """Load return policy"""
        try:
            with open('data/policies/return_policy.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def process_refund(self, order_data: Dict[str, Any], 
                      refund_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process refund request
        
        Args:
            order_data: Order information
            refund_request: Refund details (reason, refund_method, etc.)
            
        Returns:
            Refund processing result
        """
        
        # Check refund eligibility
        eligibility = self._check_refund_eligibility(order_data, refund_request)
        
        if not eligibility['eligible']:
            return {
                'success': False,
                'reason': eligibility['reason'],
                'message': eligibility['message'],
                'alternative_offered': eligibility.get('alternative')
            }
        
        # Determine refund type
        refund_type = eligibility.get('refund_type', 'full_refund')
        refund_amount = self._calculate_refund_amount(order_data, refund_type, refund_request)
        
        # Check if auto-approval is possible
        requires_approval = refund_amount > self.auto_approve_limit
        
        if requires_approval:
            return {
                'success': True,
                'requires_approval': True,
                'message': f"Your refund of ${refund_amount} requires manager approval. I've submitted it for immediate review (typically within 2 hours).",
                'estimated_approval_time': '2 hours',
                'refund_amount': refund_amount
            }
        
        # Process refund
        refund_method = refund_request.get('refund_method', 'original_payment_method')
        result = self._execute_refund(order_data, refund_amount, refund_method, refund_request)
        
        return result
    
    def _check_refund_eligibility(self, order_data: Dict[str, Any], 
                                  refund_request: Dict[str, Any]) -> Dict[str, Any]:
        """Check if order is eligible for refund"""
        
        # Check refund window
        order_date_str = order_data.get('order_date')
        if order_date_str:
            order_date = datetime.fromisoformat(order_date_str)
            days_since_order = (datetime.now() - order_date).days
            
            refund_window = self.refund_policy.get('general_principles', {}).get('refund_window_days', 30)
            
            if days_since_order > refund_window:
                # Check for special circumstances
                reason = refund_request.get('reason', '')
                if reason in ['defective_product', 'wrong_item_sent']:
                    return {
                        'eligible': True,
                        'refund_type': 'full_refund',
                        'reason': 'Special circumstance exception'
                    }
                else:
                    return {
                        'eligible': False,
                        'reason': f'Refund window ({refund_window} days) has expired',
                        'message': f"I'm sorry, but the refund window of {refund_window} days has passed. However, I can offer you store credit with a 10% bonus instead!",
                        'alternative': {
                            'type': 'store_credit',
                            'amount': order_data.get('total', 0) * 1.1,
                            'bonus': '10%'
                        }
                    }
        
        # Check if item is refundable
        category = order_data.get('category', '')
        non_refundable = self.refund_policy.get('non_refundable_items', [])
        
        for non_ref in non_refundable:
            if non_ref.get('item_type', '').lower() in category.lower():
                # Check for exceptions
                reason = refund_request.get('reason', '')
                exceptions = non_ref.get('exceptions', [])
                
                if reason not in exceptions:
                    return {
                        'eligible': False,
                        'reason': non_ref.get('reason', 'Item is non-refundable'),
                        'message': f"I'm sorry, but {category} items are non-refundable per our policy. However, I can help you find an exchange!"
                    }
        
        # Check order status
        if order_data.get('status') == 'cancelled':
            return {
                'eligible': False,
                'reason': 'Order already cancelled',
                'message': "This order has already been cancelled. If you were charged, the refund is processing."
            }
        
        # Determine refund type based on reason
        reason = refund_request.get('reason', 'changed_mind')
        refund_type = 'full_refund'
        
        # Check for partial refund scenarios
        if reason in ['minor_defect', 'missing_accessories']:
            refund_type = 'partial_refund'
        
        return {
            'eligible': True,
            'refund_type': refund_type
        }
    
    def _calculate_refund_amount(self, order_data: Dict[str, Any], 
                                 refund_type: str,
                                 refund_request: Dict[str, Any]) -> float:
        """Calculate refund amount based on type and circumstances"""
        
        base_amount = order_data.get('total', 0)
        shipping_cost = order_data.get('shipping_cost', 0)
        
        if refund_type == 'full_refund':
            # Check if shipping should be refunded
            reason = refund_request.get('reason', '')
            special_circumstances = self.refund_policy.get('special_circumstances', [])
            
            refund_shipping = False
            for circumstance in special_circumstances:
                if circumstance.get('circumstance') == reason:
                    includes = circumstance.get('includes', [])
                    if 'Original shipping' in includes or 'All shipping costs' in includes:
                        refund_shipping = True
                    break
            
            if refund_shipping:
                return base_amount  # Assuming base_amount includes shipping
            else:
                return base_amount - shipping_cost
        
        elif refund_type == 'partial_refund':
            # Calculate partial refund percentage
            refund_percentage = refund_request.get('partial_percentage', 0.5)  # Default 50%
            return base_amount * refund_percentage
        
        elif refund_type == 'shipping_refund':
            return shipping_cost
        
        return base_amount
    
    def _execute_refund(self, order_data: Dict[str, Any], 
                       refund_amount: float,
                       refund_method: str,
                       refund_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the refund"""
        
        # Get refund method details
        refund_methods = self.refund_policy.get('refund_methods', [])
        method_details = None
        
        for method in refund_methods:
            if method.get('method') == refund_method:
                method_details = method
                break
        
        if not method_details:
            method_details = refund_methods[0]  # Default to first method
        
        # Apply bonus if applicable
        final_amount = refund_amount
        bonus_amount = 0
        
        if refund_method == 'store_credit':
            bonus_percentage = method_details.get('bonus_percentage', 0) / 100
            bonus_amount = refund_amount * bonus_percentage
            final_amount = refund_amount + bonus_amount
        
        # Generate refund reference number
        import random
        reference_number = f"REF{datetime.now().strftime('%Y%m%d')}{random.randint(1000, 9999)}"
        
        # Get processing time
        processing_time = method_details.get('processing_time', '5-7 business days')
        
        # Format message based on method
        if refund_method == 'store_credit':
            message = f"✅ Done! I've added ${final_amount:.2f} in store credit to your account (${refund_amount:.2f} + ${bonus_amount:.2f} bonus). It's available to use immediately!"
        elif refund_method == 'original_payment_method':
            payment_method = order_data.get('payment_method', 'original payment method')
            message = f"✅ Refund processed! ${final_amount:.2f} will be refunded to your {payment_method} within {processing_time}."
        else:
            message = f"✅ Your refund of ${final_amount:.2f} has been processed. You'll receive it within {processing_time}."
        
        message += f"\n\nReference: {reference_number}"
        
        # Log the refund (in production, save to database)
        refund_record = {
            'order_id': order_data.get('order_id'),
            'refund_amount': final_amount,
            'refund_method': refund_method,
            'reference_number': reference_number,
            'timestamp': datetime.now().isoformat(),
            'reason': refund_request.get('reason')
        }
        
        return {
            'success': True,
            'refund_amount': final_amount,
            'bonus_amount': bonus_amount,
            'refund_method': refund_method,
            'processing_time': processing_time,
            'reference_number': reference_number,
            'message': message,
            'refund_record': refund_record
        }
    
    def offer_compensation(self, issue_type: str, 
                          order_data: Dict[str, Any],
                          severity: str = 'medium') -> Dict[str, Any]:
        """
        Offer compensation for service failures or issues
        
        Args:
            issue_type: Type of issue (delay, defect, wrong_item, etc.)
            order_data: Order information
            severity: Issue severity (low, medium, high)
            
        Returns:
            Compensation offer details
        """
        
        order_value = order_data.get('total', 0)
        customer_tier = order_data.get('customer_tier', 'regular')
        
        # Define compensation matrix
        compensation_options = {
            'delay': {
                'low': {'type': 'discount_code', 'value': '5%', 'message': '5% off your next order'},
                'medium': {'type': 'shipping_refund_and_discount', 'value': '15%', 'message': 'shipping refund + 15% discount'},
                'high': {'type': 'full_refund_option', 'value': '100%', 'message': 'full refund or free replacement'}
            },
            'defect': {
                'low': {'type': 'partial_refund', 'value': '20%', 'message': '20% refund to keep item'},
                'medium': {'type': 'replacement_and_discount', 'value': '10%', 'message': 'free replacement + 10% off next order'},
                'high': {'type': 'replacement_and_compensation', 'value': '25%', 'message': 'immediate replacement + 25% store credit'}
            },
            'wrong_item': {
                'low': {'type': 'exchange', 'value': 'free', 'message': 'free exchange'},
                'medium': {'type': 'exchange_and_discount', 'value': '10%', 'message': 'free exchange + 10% discount'},
                'high': {'type': 'keep_and_replace', 'value': 'keep_both', 'message': 'keep wrong item + send correct item free'}
            },
            'poor_quality': {
                'low': {'type': 'discount', 'value': '15%', 'message': '15% discount code'},
                'medium': {'type': 'refund_or_exchange', 'value': '100%', 'message': 'full refund or exchange'},
                'high': {'type': 'refund_and_compensation', 'value': '30%', 'message': 'full refund + 30% store credit'}
            }
        }
        
        # Get base compensation
        compensation = compensation_options.get(issue_type, compensation_options['delay']).get(severity)
        
        # Enhance for VIP customers
        if customer_tier in ['VIP', 'Premium']:
            if compensation['type'] == 'discount_code':
                compensation['value'] = str(int(compensation['value'].replace('%', '')) + 5) + '%'
        
        return {
            'success': True,
            'compensation_type': compensation['type'],
            'compensation_value': compensation['value'],
            'message': f"To make this right, I'm offering: {compensation['message']}.",
            'auto_approved': True
        }
    
    def generate_return_label(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate return shipping label
        
        Args:
            order_data: Order information
            
        Returns:
            Return label details
        """
        
        import random
        
        # Determine if return is free
        reason = order_data.get('return_reason', '')
        free_return_conditions = self.return_policy.get('shipping_costs', {}).get('free_return_conditions', [])
        
        is_free = any(condition in reason for condition in ['Defective', 'Wrong item'])
        
        # Generate tracking number
        tracking_number = f"RET{datetime.now().strftime('%Y%m%d')}{random.randint(100000, 999999)}"
        
        cost = 0.00 if is_free else self.return_policy.get('shipping_costs', {}).get('return_label_cost', 7.99)
        
        return {
            'success': True,
            'tracking_number': tracking_number,
            'cost': cost,
            'carrier': 'USPS',
            'message': f"✅ Return label generated! I've emailed it to {order_data.get('email', 'you')}.",
            'instructions': [
                'Print the label and attach it to your package',
                'Drop off at any USPS location',
                'Refund processes once we receive the item (typically 5-7 days)'
            ],
            'is_free': is_free
        }
    
    def answer_policy_question(self, question: str, policy_type: str = 'general') -> str:
        """
        Answer policy-related questions using RAG
        
        Args:
            question: Customer's policy question
            policy_type: Type of policy (refund, return, shipping, exchange)
            
        Returns:
            Policy answer
        """
        
        # Select appropriate policy
        if policy_type == 'refund':
            policy_context = json.dumps(self.refund_policy, indent=2)
        elif policy_type == 'return':
            policy_context = json.dumps(self.return_policy, indent=2)
        else:
            policy_context = json.dumps({
                'refund': self.refund_policy,
                'return': self.return_policy
            }, indent=2)
        
        prompt = f"""Based on our policies, answer the customer's question clearly and helpfully:

POLICY CONTEXT:
{policy_context}

CUSTOMER QUESTION:
{question}

Provide a clear, friendly answer in 2-3 sentences. If there are exceptions or special cases, mention them."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": "You are a helpful customer service agent explaining company policies in simple terms."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return "I'm having trouble accessing our policy information right now. Let me connect you with a specialist who can help!"
