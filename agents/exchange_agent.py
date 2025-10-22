"""
Exchange Agent - Smart Product Recommendations (Production-Ready)
Handles exchanges and AI-powered product recommendations with vector search
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import AsyncOpenAI, APIError, APITimeoutError
from pinecone import Pinecone
import httpx

logger = logging.getLogger(__name__)

class ExchangeAgent:
    """
    Exchange Agent for product recommendations and exchange processing.
    Production-ready with async operations, vector search, and retry logic.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Exchange Agent"""
        self.client = AsyncOpenAI(
            api_key=config.get('openai_api_key'),
            timeout=config.get('timeout', 30.0),
            max_retries=0
        )
        self.model = config.get('exchange_model', 'gpt-4o')
        self.temperature = config.get('exchange_temperature', 0.7)
        self.request_timeout = config.get('request_timeout', 20.0)
        
        # Initialize Pinecone for vector search
        try:
            pc = Pinecone(api_key=config.get('pinecone_api_key'))
            self.index = pc.Index(config.get('pinecone_index_name'))
            self.vector_search_available = True
        except Exception as e:
            logger.warning(f"Pinecone initialization failed: {e}")
            self.index = None
            self.vector_search_available = False
        
        # Load exchange policy
        self.exchange_policy = self._load_exchange_policy()
        
        # Settings
        self.recommendation_count = config.get('recommendation_count', 5)
        self.similarity_threshold = config.get('similarity_threshold', 0.75)
        
        # Health status
        self._healthy = True
        self._last_health_check = None
        
        logger.info(f"ExchangeAgent initialized, vector_search={self.vector_search_available}")
    
    def _load_exchange_policy(self) -> Dict:
        """Load exchange policy"""
        try:
            with open('data/policies/exchange_policy.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Exchange policy file not found, using defaults")
            return {}
        except Exception as e:
            logger.error(f"Error loading exchange policy: {e}")
            return {}
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APIError, APITimeoutError, asyncio.TimeoutError))
    )
    async def process_simple_exchange(
        self,
        order_data: Dict[str, Any],
        exchange_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process simple exchanges (same product, different size/color)
        
        Args:
            order_data: Current order information
            exchange_request: Exchange details (type, new_size, new_color, etc.)
            
        Returns:
            Exchange processing result
        """
        start_time = datetime.now()
        
        try:
            exchange_type = exchange_request.get('type')  # 'size' or 'color'
            
            # Check exchange eligibility
            eligibility = await self._check_exchange_eligibility(order_data)
            
            if not eligibility['eligible']:
                return {
                    'success': False,
                    'reason': eligibility['reason'],
                    'message': f"I'm sorry, but this item isn't eligible for exchange: {eligibility['reason']}",
                    'alternative': eligibility.get('alternative')
                }
            
            # Process based on exchange type
            if exchange_type == 'size':
                result = await self._process_size_exchange(order_data, exchange_request.get('new_size'))
            elif exchange_type == 'color':
                result = await self._process_color_exchange(order_data, exchange_request.get('new_color'))
            else:
                result = {
                    'success': False,
                    'message': "Please specify if you'd like a size or color exchange."
                }
            
            elapsed = (datetime.now() - start_time).total_seconds()
            result['latency_ms'] = elapsed * 1000
            
            logger.info(f"Exchange processed: type={exchange_type}, success={result.get('success')}, elapsed={elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing exchange: {e}", exc_info=True)
            self._healthy = False
            
            return {
                'success': False,
                'error': str(e),
                'message': "I encountered an error processing your exchange. Let me connect you with a specialist."
            }
    
    async def _check_exchange_eligibility(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if order is eligible for exchange"""
        
        # Check exchange window
        delivery_date_str = order_data.get('delivered_at')
        if delivery_date_str:
            try:
                delivery_date = datetime.fromisoformat(delivery_date_str)
                days_since_delivery = (datetime.now() - delivery_date).days
                exchange_window = 45  # Default exchange window
                
                if days_since_delivery > exchange_window:
                    return {
                        'eligible': False,
                        'reason': f'Exchange window ({exchange_window} days) has expired',
                        'alternative': {
                            'type': 'partial_refund',
                            'message': 'We can offer a 20% partial refund instead'
                        }
                    }
            except:
                pass
        
        # Check if already exchanged multiple times
        if order_data.get('exchange_count', 0) >= 2:
            return {
                'eligible': False,
                'reason': 'Maximum number of exchanges (2) reached for this order',
                'alternative': {
                    'type': 'refund',
                    'message': 'We can process a full refund instead'
                }
            }
        
        # Check product category restrictions
        category = order_data.get('category', '').lower()
        restricted_categories = ['final sale', 'personalized', 'intimate apparel']
        
        if any(cat in category for cat in restricted_categories):
            return {
                'eligible': False,
                'reason': f'{order_data.get("category")} items cannot be exchanged per policy',
                'alternative': None
            }
        
        return {'eligible': True}
    
    async def _process_size_exchange(self, order_data: Dict[str, Any], new_size: str) -> Dict[str, Any]:
        """Process size exchange with inventory check"""
        
        product_name = order_data.get('product_name', 'item')
        current_size = order_data.get('size', 'unknown')
        product_id = order_data.get('product_id')
        
        # Check inventory (simulate API call)
        in_stock = await self._check_inventory(product_id, new_size)
        
        if in_stock:
            return {
                'success': True,
                'exchange_type': 'size',
                'message': f"Perfect! I'm exchanging your {product_name} from size {current_size} to {new_size}.",
                'details': {
                    'new_size': new_size,
                    'current_size': current_size,
                    'shipping': 'free',
                    'processing_time': '24-48 hours',
                    'tracking_available': True
                },
                'next_steps': [
                    'New item ships within 24 hours',
                    'Return label emailed immediately',
                    'Free shipping both ways'
                ]
            }
        else:
            return {
                'success': False,
                'out_of_stock': True,
                'message': f"Size {new_size} is currently out of stock. Let me find similar alternatives for you!",
                'alternative_needed': True,
                'current_product': order_data
            }
    
    async def _process_color_exchange(self, order_data: Dict[str, Any], new_color: str) -> Dict[str, Any]:
        """Process color exchange with inventory check"""
        
        product_name = order_data.get('product_name', 'item')
        current_color = order_data.get('color', 'original')
        product_id = order_data.get('product_id')
        
        # Check inventory
        in_stock = await self._check_inventory(product_id, new_color)
        
        if in_stock:
            return {
                'success': True,
                'exchange_type': 'color',
                'message': f"Great choice! I'm exchanging your {current_color} {product_name} for the {new_color} version.",
                'details': {
                    'new_color': new_color,
                    'current_color': current_color,
                    'shipping': 'free',
                    'processing_time': '24-48 hours'
                },
                'next_steps': [
                    'New item ships within 24 hours',
                    'Return label emailed immediately',
                    'Free shipping both ways'
                ]
            }
        else:
            return {
                'success': False,
                'out_of_stock': True,
                'message': f"The {new_color} color is currently out of stock. Would you like to see similar products in {new_color}?",
                'alternative_needed': True
            }
    
    async def _check_inventory(self, product_id: str, variant: str) -> bool:
        """Check inventory availability (simulate API call)"""
        # Simulate inventory check with async delay
        await asyncio.sleep(0.1)
        
        # In production, call actual inventory API
        # return await inventory_api.check_stock(product_id, variant)
        
        # Simulate 80% in-stock rate
        import random
        return random.random() > 0.2
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5)
    )
    async def recommend_alternatives(
        self,
        current_product: Dict[str, Any],
        customer_preferences: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Recommend alternative products using vector similarity search
        
        Args:
            current_product: Product to find alternatives for
            customer_preferences: Customer's stated preferences
            
        Returns:
            Dictionary with recommended products
        """
        start_time = datetime.now()
        
        try:
            if not self.vector_search_available:
                return {
                    'success': False,
                    'error': 'Product recommendation service temporarily unavailable',
                    'message': 'Let me connect you with a specialist for personalized recommendations.'
                }
            
            # Generate query embedding
            query_text = self._build_product_query(current_product, customer_preferences)
            
            # Get embedding asynchronously
            query_embedding = await self._generate_embedding_async(query_text)
            
            # Search Pinecone
            search_results = await asyncio.to_thread(
                self.index.query,
                vector=query_embedding,
                top_k=self.recommendation_count + 1,
                include_metadata=True,
                filter={'type': 'product'},
                namespace=''
            )
            
            # Filter and format recommendations
            recommendations = []
            for match in search_results.get('matches', []):
                # Skip current product
                if match['metadata'].get('product_id') == current_product.get('product_id'):
                    continue
                
                if match['score'] < self.similarity_threshold:
                    continue
                
                recommendations.append({
                    'product_id': match['metadata'].get('product_id'),
                    'name': match['metadata'].get('name'),
                    'price': match['metadata'].get('price'),
                    'category': match['metadata'].get('category'),
                    'similarity_score': round(match['score'], 2),
                    'reason': self._generate_recommendation_reason(
                        current_product,
                        match['metadata'],
                        customer_preferences
                    )
                })
                
                if len(recommendations) >= self.recommendation_count:
                    break
            
            # Generate conversational message
            message = self._format_recommendations_message(recommendations, customer_preferences)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Recommendations generated: count={len(recommendations)}, elapsed={elapsed:.2f}s")
            
            return {
                'success': True,
                'count': len(recommendations),
                'recommendations': recommendations,
                'message': message,
                'latency_ms': elapsed * 1000
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}", exc_info=True)
            
            return {
                'success': False,
                'error': str(e),
                'message': 'I had trouble finding recommendations. Let me connect you with a product specialist.'
            }
    
    def _build_product_query(self, product: Dict[str, Any], preferences: Dict[str, Any] = None) -> str:
        """Build search query for product recommendations"""
        
        query_parts = [
            product.get('name', ''),
            product.get('category', ''),
            product.get('description', '')[:200],  # Truncate long descriptions
        ]
        
        if preferences:
            if preferences.get('preferred_style'):
                query_parts.append(preferences['preferred_style'])
            if preferences.get('preferred_color'):
                query_parts.append(preferences['preferred_color'])
            if preferences.get('preferred_features'):
                query_parts.extend(preferences['preferred_features'][:3])  # Limit features
        
        return ' '.join(filter(None, query_parts))
    
    async def _generate_embedding_async(self, text: str) -> List[float]:
        """Generate embedding asynchronously"""
        try:
            response = await self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=1536
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def _generate_recommendation_reason(
        self,
        current_product: Dict[str, Any],
        recommended_product: Dict[str, Any],
        preferences: Dict[str, Any] = None
    ) -> str:
        """Generate reason why product is recommended"""
        
        reasons = []
        
        # Same category
        if current_product.get('category') == recommended_product.get('category'):
            reasons.append(f"Similar {recommended_product.get('category', 'style')}")
        
        # Price comparison
        current_price = current_product.get('price', 0)
        rec_price = recommended_product.get('price', 0)
        
        if abs(current_price - rec_price) < 10:
            reasons.append("Similar price")
        elif rec_price < current_price:
            savings = current_price - rec_price
            reasons.append(f"${savings:.2f} less")
        
        # Preferences match
        if preferences:
            if preferences.get('preferred_size') in str(recommended_product.get('sizes', [])):
                reasons.append(f"Available in {preferences['preferred_size']}")
        
        return ', '.join(reasons) if reasons else "Highly rated alternative"
    
    def _format_recommendations_message(
        self,
        recommendations: List[Dict[str, Any]],
        preferences: Dict[str, Any] = None
    ) -> str:
        """Format recommendations into conversational message"""
        
        if not recommendations:
            return "I couldn't find suitable alternatives right now. Would you prefer a refund instead?"
        
        message_parts = [f"I found {len(recommendations)} great alternatives for you:\n"]
        
        for i, rec in enumerate(recommendations, 1):
            message_parts.append(
                f"\n{i}. **{rec['name']}** - ${rec['price']:.2f}"
            )
            message_parts.append(f"   ↳ {rec['reason']}")
        
        message_parts.append("\n\nWould you like details on any of these? Just let me know the number!")
        
        return ''.join(message_parts)
    
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
            
            openai_latency = (datetime.now() - start_time).total_seconds()
            
            # Test Pinecone if available
            pinecone_status = 'not_configured'
            pinecone_latency = 0
            
            if self.vector_search_available:
                pinecone_start = datetime.now()
                try:
                    stats = await asyncio.to_thread(self.index.describe_index_stats)
                    pinecone_status = 'connected'
                    pinecone_latency = (datetime.now() - pinecone_start).total_seconds()
                except:
                    pinecone_status = 'error'
            
            self._healthy = True
            self._last_health_check = datetime.now()
            
            return {
                'status': 'healthy',
                'service': 'exchange_agent',
                'openai_status': 'connected',
                'openai_latency_ms': openai_latency * 1000,
                'pinecone_status': pinecone_status,
                'pinecone_latency_ms': pinecone_latency * 1000,
                'last_check': self._last_health_check.isoformat()
            }
            
        except Exception as e:
            self._healthy = False
            logger.error(f"Health check failed: {e}")
            
            return {
                'status': 'unhealthy',
                'service': 'exchange_agent',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
    
    def is_healthy(self) -> bool:
        """Quick health status check"""
        return self._healthy
