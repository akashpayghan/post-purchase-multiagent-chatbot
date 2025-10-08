"""
Exchange Agent - Product Recommendations
Handles size/color exchanges and recommends alternative products based on preferences
"""

import os
import json
from typing import Dict, List, Any, Optional
from openai import OpenAI
from pinecone import Pinecone

class ExchangeAgent:
    """
    Specialist agent for exchanges and product recommendations.
    Uses vector search to find similar products and provides personalized suggestions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Exchange Agent"""
        self.client = OpenAI(api_key=config.get('openai_api_key'))
        self.model = config.get('exchange_model', 'gpt-4o')
        self.temperature = config.get('exchange_temperature', 0.7)
        
        # Initialize Pinecone
        pc = Pinecone(api_key=config.get('pinecone_api_key'))
        self.index = pc.Index(config.get('pinecone_index_name'))
        
        # Load policies
        self.exchange_policy = self._load_exchange_policy()
        
        # Settings
        self.recommendation_count = config.get('recommendation_count', 5)
        self.similarity_threshold = config.get('similarity_threshold', 0.75)
        
    def _load_exchange_policy(self) -> Dict:
        """Load exchange policy"""
        try:
            with open('data/policies/exchange_policy.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def process_simple_exchange(self, order_data: Dict[str, Any], 
                               exchange_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process simple exchanges (same product, different size/color)
        
        Args:
            order_data: Current order information
            exchange_request: Exchange details (type, new_size, new_color, etc.)
            
        Returns:
            Exchange processing result
        """
        exchange_type = exchange_request.get('type')  # 'size' or 'color'
        product_id = order_data.get('product_id')
        
        # Check exchange eligibility
        eligibility = self._check_exchange_eligibility(order_data)
        
        if not eligibility['eligible']:
            return {
                'success': False,
                'reason': eligibility['reason'],
                'message': f"I'm sorry, but this item isn't eligible for exchange: {eligibility['reason']}"
            }
        
        # Process based on exchange type
        if exchange_type == 'size':
            new_size = exchange_request.get('new_size')
            return self._process_size_exchange(order_data, new_size)
        
        elif exchange_type == 'color':
            new_color = exchange_request.get('new_color')
            return self._process_color_exchange(order_data, new_color)
        
        else:
            return {
                'success': False,
                'message': "Please specify if you'd like a size or color exchange."
            }
    
    def _check_exchange_eligibility(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if order is eligible for exchange"""
        
        # Check exchange window
        from datetime import datetime, timedelta
        
        delivery_date_str = order_data.get('delivered_at')
        if delivery_date_str:
            delivery_date = datetime.fromisoformat(delivery_date_str)
            days_since_delivery = (datetime.now() - delivery_date).days
            exchange_window = self.exchange_policy.get('general_rules', {}).get('exchange_window_days', 45)
            
            if days_since_delivery > exchange_window:
                return {
                    'eligible': False,
                    'reason': f'Exchange window ({exchange_window} days) has expired'
                }
        
        # Check if already exchanged
        if order_data.get('exchange_count', 0) >= 2:
            return {
                'eligible': False,
                'reason': 'Maximum number of exchanges (2) reached for this order'
            }
        
        # Check product category restrictions
        product_category = order_data.get('category')
        if product_category in ['Final Sale', 'Personalized', 'Intimate Apparel']:
            return {
                'eligible': False,
                'reason': f'{product_category} items cannot be exchanged per policy'
            }
        
        return {'eligible': True}
    
    def _process_size_exchange(self, order_data: Dict[str, Any], new_size: str) -> Dict[str, Any]:
        """Process size exchange"""
        
        product_name = order_data.get('product_name')
        current_size = order_data.get('size')
        
        # In production, check inventory
        in_stock = True  # Simulate inventory check
        
        if in_stock:
            return {
                'success': True,
                'exchange_type': 'size',
                'message': f"Perfect! I'm exchanging your {product_name} from size {current_size} to {new_size}.",
                'details': {
                    'new_size': new_size,
                    'shipping': 'free',
                    'processing_time': '3 business days',
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
                'alternative_needed': True
            }
    
    def _process_color_exchange(self, order_data: Dict[str, Any], new_color: str) -> Dict[str, Any]:
        """Process color exchange"""
        
        product_name = order_data.get('product_name')
        current_color = order_data.get('color')
        
        # In production, check inventory
        in_stock = True  # Simulate inventory check
        
        if in_stock:
            return {
                'success': True,
                'exchange_type': 'color',
                'message': f"Great choice! I'm exchanging your {current_color} {product_name} for the {new_color} version.",
                'details': {
                    'new_color': new_color,
                    'shipping': 'free',
                    'processing_time': '3 business days',
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
                'message': f"The {new_color} color is currently out of stock. Would you like to see similar products in {new_color}?",
                'alternative_needed': True
            }
    
    def recommend_alternatives(self, current_product: Dict[str, Any], 
                              customer_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Recommend alternative products using vector similarity search
        
        Args:
            current_product: Product to find alternatives for
            customer_preferences: Customer's stated preferences (size, color, price range, etc.)
            
        Returns:
            Dictionary with recommended products
        """
        
        # Generate embedding for current product
        query_text = self._build_product_query(current_product, customer_preferences)
        query_embedding = self._generate_embedding(query_text)
        
        # Search Pinecone for similar products
        search_results = self.index.query(
            vector=query_embedding,
            top_k=self.recommendation_count + 1,  # +1 to exclude current product
            include_metadata=True,
            filter={'type': 'product'}  # Only search products
        )
        
        # Filter and format recommendations
        recommendations = []
        for match in search_results['matches']:
            # Skip the current product
            if match['metadata'].get('product_id') == current_product.get('product_id'):
                continue
            
            # Check similarity threshold
            if match['score'] < self.similarity_threshold:
                continue
            
            recommendations.append({
                'product_id': match['metadata'].get('product_id'),
                'name': match['metadata'].get('name'),
                'price': match['metadata'].get('price'),
                'category': match['metadata'].get('category'),
                'similarity_score': round(match['score'], 2),
                'why_recommended': self._generate_recommendation_reason(
                    current_product, 
                    match['metadata'], 
                    customer_preferences
                )
            })
            
            if len(recommendations) >= self.recommendation_count:
                break
        
        # Generate conversational message
        message = self._format_recommendations_message(recommendations, customer_preferences)
        
        return {
            'success': True,
            'count': len(recommendations),
            'recommendations': recommendations,
            'message': message
        }
    
    def _build_product_query(self, product: Dict[str, Any], 
                            preferences: Dict[str, Any] = None) -> str:
        """Build search query for product recommendations"""
        
        query_parts = [
            product.get('name', ''),
            product.get('category', ''),
            product.get('description', ''),
        ]
        
        if preferences:
            if preferences.get('preferred_style'):
                query_parts.append(preferences['preferred_style'])
            if preferences.get('preferred_features'):
                query_parts.extend(preferences['preferred_features'])
        
        return ' '.join(filter(None, query_parts))
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    def _generate_recommendation_reason(self, current_product: Dict[str, Any],
                                       recommended_product: Dict[str, Any],
                                       preferences: Dict[str, Any] = None) -> str:
        """Generate reason why product is recommended"""
        
        reasons = []
        
        # Same category
        if current_product.get('category') == recommended_product.get('category'):
            reasons.append(f"Same category ({recommended_product.get('category')})")
        
        # Price comparison
        current_price = current_product.get('price', 0)
        rec_price = recommended_product.get('price', 0)
        
        if abs(current_price - rec_price) < 10:
            reasons.append("Similar price point")
        elif rec_price < current_price:
            savings = current_price - rec_price
            reasons.append(f"${savings:.2f} less expensive")
        
        # Check preferences
        if preferences:
            if preferences.get('preferred_size') in recommended_product.get('sizes', []):
                reasons.append(f"Available in your size ({preferences['preferred_size']})")
            
            if preferences.get('preferred_color') in recommended_product.get('colors', []):
                reasons.append(f"Available in {preferences['preferred_color']}")
        
        return ', '.join(reasons) if reasons else "Similar style and quality"
    
    def _format_recommendations_message(self, recommendations: List[Dict[str, Any]],
                                       preferences: Dict[str, Any] = None) -> str:
        """Format recommendations into conversational message"""
        
        if not recommendations:
            return "I couldn't find suitable alternatives right now. Would you prefer a refund instead?"
        
        message_parts = [
            f"I found {len(recommendations)} great alternatives for you:\n"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            message_parts.append(
                f"\n{i}. **{rec['name']}** - ${rec['price']}"
            )
            message_parts.append(f"   ↳ {rec['why_recommended']}")
        
        message_parts.append("\n\nWould you like details on any of these? Just let me know the number!")
        
        return ''.join(message_parts)
    
    def get_size_recommendation(self, customer_data: Dict[str, Any], 
                               product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend size based on customer's order history and product fit data
        
        Args:
            customer_data: Customer's purchase history and preferences
            product: Product being considered
            
        Returns:
            Size recommendation
        """
        
        # Get customer's typical size from order history
        size_history = customer_data.get('size_history', {})
        category = product.get('category')
        
        typical_size = size_history.get(category, 'M')  # Default to M if no history
        
        # Check if product runs large/small
        return_reasons = product.get('common_return_reasons', [])
        size_tendency = 'true to size'
        
        if 'Size too small' in return_reasons or 'runs small' in str(return_reasons).lower():
            size_tendency = 'small'
            recommended_size = self._size_up(typical_size)
            message = f"Based on customer feedback, this item runs small. I'd recommend size {recommended_size} (one size up from your usual {typical_size})."
        elif 'Size too large' in return_reasons or 'runs large' in str(return_reasons).lower():
            size_tendency = 'large'
            recommended_size = self._size_down(typical_size)
            message = f"This item tends to run large. I'd recommend size {recommended_size} (one size down from your usual {typical_size})."
        else:
            recommended_size = typical_size
            message = f"Based on your order history, size {recommended_size} should be perfect for you!"
        
        return {
            'recommended_size': recommended_size,
            'typical_size': typical_size,
            'size_tendency': size_tendency,
            'message': message,
            'confidence': 0.8
        }
    
    def _size_up(self, size: str) -> str:
        """Get next size up"""
        size_map = {'XS': 'S', 'S': 'M', 'M': 'L', 'L': 'XL', 'XL': 'XXL'}
        return size_map.get(size, size)
    
    def _size_down(self, size: str) -> str:
        """Get next size down"""
        size_map = {'XXL': 'XL', 'XL': 'L', 'L': 'M', 'M': 'S', 'S': 'XS'}
        return size_map.get(size, size)
