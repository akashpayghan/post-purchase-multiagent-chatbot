"""
Visual Agent - AI-Powered Image Analysis (Production-Ready)
Handles product defect detection, quality verification using Gemini Vision
"""

import asyncio
import logging
import base64
from typing import Dict, Any, Optional
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import google.generativeai as genai
from PIL import Image
import io

logger = logging.getLogger(__name__)

class VisualAgent:
    """
    Visual Agent for AI-powered image analysis and defect detection.
    Production-ready with async operations, retry logic, and enhanced validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Visual Agent"""
        api_key = config.get('gemini_api_key')
        
        if not api_key or api_key == 'not-set':
            logger.warning("Gemini API key not set, Visual Agent will be in limited mode")
            self.model = None
        else:
            genai.configure(api_key=api_key)
            model_name = config.get('visual_model', 'gemini-1.5-pro')
            self.model = genai.GenerativeModel(model_name)
        
        self.max_image_size_mb = config.get('max_image_size_mb', 5)
        self.supported_formats = config.get('supported_formats', ['JPEG', 'PNG', 'WEBP', 'JPG'])
        self.request_timeout = config.get('request_timeout', 30.0)
        
        # Health status
        self._healthy = True if self.model else False
        self._last_health_check = None
        
        logger.info(f"VisualAgent initialized, available: {self.model is not None}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    async def analyze_product_image(
        self,
        image_path: str,
        issue_type: str,
        product_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze product image for defects or issues
        
        Args:
            image_path: Path to image file or base64 string
            issue_type: Type of issue (defect, wrong_item, quality_issue, etc.)
            product_info: Optional product information for context
            
        Returns:
            Analysis results with defect detection and recommendations
        """
        start_time = datetime.now()
        product_info = product_info or {}
        
        try:
            if not self.model:
                return {
                    'success': False,
                    'error': 'Visual analysis not available (Gemini API key not configured)',
                    'recommended_action': self._get_fallback_action(issue_type)
                }
            
            # Load and validate image
            image = await self._load_and_validate_image(image_path)
            
            if not image:
                return {
                    'success': False,
                    'error': 'Invalid or unsupported image format',
                    'recommended_action': {'action': 'request_new_image', 'message': 'Please upload a clear photo in JPG or PNG format'}
                }
            
            # Build analysis prompt
            prompt = self._build_analysis_prompt(issue_type, product_info)
            
            # Perform analysis with timeout
            analysis_result = await asyncio.wait_for(
                self._perform_analysis(image, prompt),
                timeout=self.request_timeout
            )
            
            # Parse and structure results
            structured_result = self._parse_analysis_response(analysis_result, issue_type)
            
            # Generate recommendation
            recommendation = self._recommend_action(structured_result, issue_type)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Image analysis complete: issue={issue_type}, defect={structured_result.get('issue_confirmed')}, elapsed={elapsed:.2f}s")
            
            return {
                'success': True,
                'issue_type': issue_type,
                'analysis': structured_result,
                'recommended_action': recommendation,
                'latency_ms': elapsed * 1000
            }
            
        except asyncio.TimeoutError:
            logger.error(f"Image analysis timed out after {self.request_timeout}s")
            return {
                'success': False,
                'error': 'Analysis timeout',
                'recommended_action': self._get_fallback_action(issue_type)
            }
        except Exception as e:
            logger.error(f"Error analyzing image: {e}", exc_info=True)
            self._healthy = False
            
            return {
                'success': False,
                'error': str(e),
                'recommended_action': self._get_fallback_action(issue_type)
            }
    
    async def _load_and_validate_image(self, image_source: str) -> Optional[Image.Image]:
        """Load and validate image with enhanced security checks"""
        try:
            # Load image
            if image_source.startswith('data:image') or ',' in image_source:
                # Base64 encoded
                if ',' in image_source:
                    image_source = image_source.split(',')[1]
                image_data = base64.b64decode(image_source)
                image = Image.open(io.BytesIO(image_data))
            else:
                # File path
                image = Image.open(image_source)
            
            # Validate format
            if image.format not in self.supported_formats:
                logger.warning(f"Unsupported format: {image.format}")
                return None
            
            # Validate size
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format)
            size_mb = img_byte_arr.tell() / (1024 * 1024)
            
            if size_mb > self.max_image_size_mb:
                logger.warning(f"Image too large: {size_mb:.2f}MB")
                # Compress image
                image = self._compress_image(image)
            
            # Convert RGBA to RGB if needed
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None
    
    def _compress_image(self, image: Image.Image, max_size: tuple = (1024, 1024)) -> Image.Image:
        """Compress image while maintaining aspect ratio"""
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image
    
    def _build_analysis_prompt(self, issue_type: str, product_info: Dict[str, Any]) -> str:
        """Build analysis prompt for Gemini Vision"""
        
        prompts = {
            'defect': """Analyze this product image for manufacturing defects or damage.

Look for:
- Stitching issues, loose threads, seam problems
- Surface damage, scratches, dents, cracks
- Stains, discoloration, fading
- Broken parts or missing components
- Any quality issues

Provide:
1. DEFECT PRESENT: YES/NO
2. DESCRIPTION: Detailed description of the defect
3. SEVERITY: minor/moderate/major/critical
4. CONFIDENCE: Your confidence level (0-100%)

Be thorough and objective.""",
            
            'wrong_item': """Compare this product image to the expected product.

Expected Product: {product_name}

Analyze:
1. Does this match the expected product?
2. If different, what is it?
3. Are there differences in color, style, or variant?

Provide:
1. MATCHES EXPECTED: YES/NO
2. ACTUAL ITEM: Description of what you see
3. DIFFERENCES: List specific differences
4. CONFIDENCE: Your confidence level (0-100%)""",
            
            'quality_issue': """Evaluate the overall quality of this product.

Assess:
- Material quality and finish
- Construction quality
- Color accuracy and consistency
- Overall craftsmanship
- Value for typical e-commerce standards

Provide:
1. QUALITY ASSESSMENT: poor/fair/good/excellent
2. ISSUES FOUND: List any quality concerns
3. RECOMMENDATION: Should customer keep, exchange, or return?
4. CONFIDENCE: Your confidence level (0-100%)""",
            
            'color_mismatch': """Analyze if the product color matches expectations.

Compare the actual product color to typical e-commerce product photos.

Provide:
1. COLOR MATCH: YES/NO
2. ACTUAL COLOR: Describe the color you see
3. DIFFERENCE: Describe the difference if any
4. SEVERITY: minor/moderate/significant
5. CONFIDENCE: Your confidence level (0-100%)"""
        }
        
        prompt = prompts.get(issue_type, prompts['defect'])
        
        if product_info.get('name'):
            prompt = prompt.replace('{product_name}', product_info['name'])
        
        return prompt
    
    async def _perform_analysis(self, image: Image.Image, prompt: str) -> str:
        """Perform actual Gemini Vision analysis"""
        try:
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            # Generate content
            response = await asyncio.to_thread(
                self.model.generate_content,
                [prompt, image]
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def _parse_analysis_response(self, response: str, issue_type: str) -> Dict[str, Any]:
        """Parse Gemini response into structured data"""
        
        response_lower = response.lower()
        
        # Extract key information
        issue_confirmed = 'yes' in response_lower.split('defect present:')[1].split('\n')[0] if 'defect present:' in response_lower else False
        
        if not issue_confirmed and issue_type == 'wrong_item':
            issue_confirmed = 'no' in response_lower.split('matches expected:')[1].split('\n')[0] if 'matches expected:' in response_lower else False
        
        # Extract severity
        severity = 'minor'
        if 'severity:' in response_lower:
            severity_text = response_lower.split('severity:')[1].split('\n')[0].strip()
            if 'major' in severity_text or 'critical' in severity_text:
                severity = 'major'
            elif 'moderate' in severity_text:
                severity = 'moderate'
        
        # Extract confidence
        confidence = 0.8
        if 'confidence:' in response_lower:
            try:
                conf_text = response_lower.split('confidence:')[1].split('\n')[0].strip()
                confidence = float(''.join(c for c in conf_text if c.isdigit() or c == '.')) / 100
            except:
                pass
        
        return {
            'issue_confirmed': issue_confirmed,
            'severity': severity,
            'confidence': confidence,
            'description': response,
            'raw_response': response
        }
    
    def _recommend_action(self, analysis: Dict[str, Any], issue_type: str) -> Dict[str, Any]:
        """Generate recommended action based on analysis"""
        
        if not analysis.get('issue_confirmed'):
            return {
                'action': 'no_action_needed',
                'message': 'No significant issues detected. If you still have concerns, please contact support.',
                'priority': 'low'
            }
        
        severity = analysis.get('severity', 'minor')
        
        if severity in ['major', 'critical']:
            return {
                'action': 'immediate_replacement',
                'message': 'We\'ve confirmed a significant issue. We\'ll send a replacement with express shipping immediately. You can keep the defective item.',
                'priority': 'high',
                'compensation': 'express_shipping_upgrade',
                'process_time': 'immediate'
            }
        elif severity == 'moderate':
            return {
                'action': 'replacement',
                'message': 'We\'ve identified an issue with your item. We\'ll send a replacement with free return shipping for the original item.',
                'priority': 'medium',
                'compensation': 'free_return',
                'process_time': '24_hours'
            }
        else:  # minor
            return {
                'action': 'offer_options',
                'message': 'We detected a minor issue. You can: 1) Keep item with 20% refund, 2) Exchange for new item, or 3) Full refund.',
                'priority': 'low',
                'options': ['partial_refund', 'exchange', 'full_refund']
            }
    
    def _get_fallback_action(self, issue_type: str) -> Dict[str, Any]:
        """Fallback action when analysis is unavailable"""
        return {
            'action': 'manual_review',
            'message': 'We\'ll have our team review your case personally. You\'ll hear from us within 24 hours with a solution.',
            'priority': 'medium',
            'requires_human_review': True
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            if not self.model:
                return {
                    'status': 'unavailable',
                    'service': 'visual_agent',
                    'reason': 'Gemini API key not configured'
                }
            
            # Test with dummy image analysis
            start_time = datetime.now()
            
            # Create test image
            test_image = Image.new('RGB', (100, 100), color='red')
            
            test_response = await asyncio.wait_for(
                self._perform_analysis(test_image, "Describe this image briefly"),
                timeout=10.0
            )
            
            latency = (datetime.now() - start_time).total_seconds()
            
            self._healthy = True
            self._last_health_check = datetime.now()
            
            return {
                'status': 'healthy',
                'service': 'visual_agent',
                'gemini_status': 'connected',
                'latency_ms': latency * 1000,
                'last_check': self._last_health_check.isoformat()
            }
            
        except Exception as e:
            self._healthy = False
            logger.error(f"Health check failed: {e}")
            
            return {
                'status': 'unhealthy',
                'service': 'visual_agent',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
    
    def is_healthy(self) -> bool:
        """Quick health status check"""
        return self._healthy
