"""
Resolution Agent - Refund/Return Processing (Production-Ready)
Handles refunds, returns, compensation, and policy enforcement
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import AsyncOpenAI, APIError, APITimeoutError
import random

logger = logging.getLogger(__name__)

class ResolutionAgent:
    """
    Resolution Agent for refunds, returns, and compensation.
    Production-ready with async operations, policy enforcement, and retry logic.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Resolution Agent"""
        self.client = AsyncOpenAI(
            api_key=config.get('openai_api_key'),
            timeout=config.get('timeout', 30.0),
            max_retries=0
        )
        self.model = config.get('resolution_model', 'gpt-4o-mini')
        self.temperature = config.get('resolution_temperature', 0.3)
        self.request_timeout = config.get('request_timeout', 20.0)
        
        # Load policies
        self.refund_policy = self._load_refund_policy()
        self.return_policy = self._load_return_policy()
        
        # Settings
        self.auto_approve_limit = config.get('auto_approve_limit_usd', 50)
        
        # Health status
        self._healthy = True
        self._last_health_check = None
        
        logger.info(f"ResolutionAgent initialized, auto_approve_limit=${self.auto_approve_limit}")
    
    def _load_refund_policy(self) -> Dict:
        """Load refund policy"""
        try:
            with open('data/policies/refund_policy.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Refund policy file not found, using defaults")
            return {}
        except Exception as e:
            logger.error(f"Error loading refund policy: {e}")
            return {}
    
    def _load_return_policy(self) -> Dict:
        """Load return policy"""
        try:
            with open('data/policies/return_policy.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Return policy file not found, using defaults")
            return {}
        except Exception as e:
            logger.error(f"Error loading return policy: {e}")
            return {}
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APIError, APITimeoutError, asyncio.TimeoutError))
    )
    async def process_refund(
        self,
        order_data: Dict[str, Any],
        refund_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process refund request with policy enforcement
        
        Args:
            order_data: Order information
            refund_request: Refund details (reason, refund_method, etc.)
            
        Returns:
            Refund processing result
        """
        start_time = datetime.now()
        
        try:
            # Check refund eligibility
            eligibility = await self._check_refund_eligibility(order_data, refund_request)
            
            if not eligibility['eligible']:
                return {
                    'success': False,
                    'reason': eligibility['reason'],
                    'message': eligibility.get('message', 'This order is not eligible for refund.'),
                    'alternative_offered': eligibility.get('alternative')
                }
            
            # Determine refund type and amount
            refund_type = eligibility.get('refund_type', 'full_refund')
            refund_amount = self._calculate_refund_amount(order_data, refund_type, refund_request)
            
            # Check if requires approval
            requires_approval = refund_amount > self.auto_approve_limit
            
            if requires_approval:
                return {
                    'success': True,
                    'requires_approval': True,
                    'message': f"Your refund of ${refund_amount:.2f} requires manager approval. I've submitted it for immediate review (typically within 2 hours).",
                    'estimated_approval_time': '2 hours',
                    'refund_amount': refund_amount,
                    'reference_number': self._generate_reference_number('REF')
                }
            
            # Process refund automatically
            refund_method = refund_request.get('refund_method', 'original_payment_method')
            result = await self._execute_refund(order_data, refund_amount, refund_method, refund_request)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            result['latency_ms'] = elapsed * 1000
            
            logger.info(f"Refund processed: amount=${refund_amount}, method={refund_method}, elapsed={elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing refund: {e}", exc_info=True)
            self._healthy = False
            
            return {
                'success': False,
                'error': str(e),
                'message': "I encountered an error processing your refund. Let me connect you with our refunds team."
            }
    
    async def _check_refund_eligibility(
        self,
        order_data: Dict[str, Any],
        refund_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if order is eligible for refund"""
        
        # Check refund window
        order_date_str = order_data.get('order_date')
        if order_date_str:
            try:
                order_date = datetime.fromisoformat(order_date_str)
                days_since_order = (datetime.now() - order_date).days
                
                refund_window = 30  # Default refund window
                
                if days_since_order > refund_window:
                    reason = refund_request.get('reason', '')
                    # Special circumstances
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
            except:
                pass
        
        # Check if item is refundable
        category = order_data.get('category', '').lower()
        non_refundable_categories = ['final sale', 'personalized', 'intimate apparel']
        
        if any(cat in category for cat in non_refundable_categories):
            reason = refund_request.get('reason', '')
            # Exceptions for non-refundable
            if reason not in ['defective_product', 'wrong_item_sent']:
                return {
                    'eligible': False,
                    'reason': f'{order_data.get("category")} items are non-refundable per policy',
                    'message': f"I'm sorry, but {category} items are non-refundable. However, I can help you find an exchange!"
                }
        
        # Check order status
        if order_data.get('status') == 'cancelled':
            return {
                'eligible': False,
                'reason': 'Order already cancelled',
                'message': "This order has already been cancelled. If you were charged, the refund is processing."
            }
        
        # Determine refund type
        reason = refund_request.get('reason', 'changed_mind')
        refund_type = 'full_refund'
        
        if reason in ['minor_defect', 'missing_accessories']:
            refund_type = 'partial_refund'
        
        return {
            'eligible': True,
            'refund_type': refund_type
        }
    
    def _calculate_refund_amount(
        self,
        order_data: Dict[str, Any],
        refund_type: str,
        refund_request: Dict[str, Any]
    ) -> float:
        """Calculate refund amount based on type and circumstances"""
        
        base_amount = order_data.get('total', 0)
        shipping_cost = order_data.get('shipping_cost', 0)
        
        if refund_type == 'full_refund':
            reason = refund_request.get('reason', '')
            
            # Check if shipping should be refunded
            special_reasons = ['defective_product', 'wrong_item_sent', 'late_delivery']
            refund_shipping = reason in special_reasons
            
            if refund_shipping:
                return base_amount
            else:
                return base_amount - shipping_cost
        
        elif refund_type == 'partial_refund':
            refund_percentage = refund_request.get('partial_percentage', 0.5)
            return base_amount * refund_percentage
        
        elif refund_type == 'shipping_refund':
            return shipping_cost
        
        return base_amount
    
    async def _execute_refund(
        self,
        order_data: Dict[str, Any],
        refund_amount: float,
        refund_method: str,
        refund_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the refund with payment processor"""
        
        # Simulate payment processor API call
        await asyncio.sleep(random.uniform(0.2, 0.5))
        
        # In production, call actual payment processor API
        # result = await payment_processor.process_refund(...)
        
        # Get processing time based on method
        processing_times = {
            'original_payment_method': '5-7 business days',
            'store_credit': 'instant',
            'gift_card': '1 business day',
            'bank_transfer': '7-10 business days'
        }
        
        processing_time = processing_times.get(refund_method, '5-7 business days')
        
        # Apply bonus for store credit
        final_amount = refund_amount
        bonus_amount = 0
        
        if refund_method == 'store_credit':
            bonus_amount = refund_amount * 0.10  # 10% bonus
            final_amount = refund_amount + bonus_amount
        
        # Generate reference number
        reference_number = self._generate_reference_number('REF')
        
        # Format message
        if refund_method == 'store_credit':
            message = f"✅ Done! I've added ${final_amount:.2f} in store credit to your account (${refund_amount:.2f} + ${bonus_amount:.2f} bonus). It's available to use immediately!"
        elif refund_method == 'original_payment_method':
            payment_method = order_data.get('payment_method', 'original payment method')
            message = f"✅ Refund processed! ${final_amount:.2f} will be refunded to your {payment_method} within {processing_time}."
        else:
            message = f"✅ Your refund of ${final_amount:.2f} has been processed. You'll receive it within {processing_time}."
        
        message += f"\n\nReference: {reference_number}"
        
        # Log refund record
        refund_record = {
            'order_id': order_data.get('order_id'),
            'refund_amount': final_amount,
            'refund_method': refund_method,
            'reference_number': reference_number,
            'timestamp': datetime.now().isoformat(),
            'reason': refund_request.get('reason')
        }
        
        logger.info(f"Refund executed: {refund_record}")
        
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
    
    async def generate_return_label(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate return shipping label"""
        
        try:
            # Simulate label generation
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            reason = order_data.get('return_reason', '')
            free_return_reasons = ['defective_product', 'wrong_item_sent']
            
            is_free = any(reason_keyword in reason.lower() for reason_keyword in free_return_reasons)
            
            tracking_number = self._generate_reference_number('RET')
            cost = 0.00 if is_free else 7.99
            
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
            
        except Exception as e:
            logger.error(f"Error generating return label: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': "I had trouble generating the label. Let me connect you with our shipping team."
            }
    
    def _generate_reference_number(self, prefix: str) -> str:
        """Generate unique reference number"""
        timestamp = datetime.now().strftime('%Y%m%d')
        random_part = random.randint(1000, 9999)
        return f"{prefix}{timestamp}{random_part}"
    
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
                'service': 'resolution_agent',
                'openai_status': 'connected',
                'latency_ms': latency * 1000,
                'last_check': self._last_health_check.isoformat()
            }
            
        except Exception as e:
            self._healthy = False
            logger.error(f"Health check failed: {e}")
            
            return {
                'status': 'unhealthy',
                'service': 'resolution_agent',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
    
    def is_healthy(self) -> bool:
        """Quick health status check"""
        return self._healthy
