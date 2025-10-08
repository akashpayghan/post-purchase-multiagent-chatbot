"""
Visual Agent - Image Analysis Using Gemini
Handles image uploads, analyzes product condition, verifies defects, and matches against catalog
"""

import os
import base64
from typing import Dict, List, Any, Optional
from PIL import Image
import io
import google.generativeai as genai

class VisualAgent:
    """
    Specialist agent for visual verification and image analysis.
    Uses Gemini Vision API to analyze product photos, detect defects, and verify claims.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Visual Agent"""
        genai.configure(api_key=config.get('gemini_api_key'))
        self.model = genai.GenerativeModel(config.get('visual_model', 'gemini-1.5-pro'))
        self.max_image_size_mb = config.get('max_image_size_mb', 5)
        self.supported_formats = config.get('supported_formats', ['jpg', 'jpeg', 'png', 'webp'])
        
    def analyze_product_image(self, image_path: str, 
                             issue_type: str,
                             product_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze uploaded product image for defects or issues
        
        Args:
            image_path: Path to uploaded image or base64 string
            issue_type: Type of issue (defect, wrong_item, damage, color_mismatch, quality_issue)
            product_info: Information about the expected product
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Load image
            image = self._load_image(image_path)
            
            # Build prompt
            prompt = self._build_analysis_prompt(issue_type, product_info)
            
            # Analyze with Gemini
            response = self.model.generate_content([prompt, image])
            
            # Parse response
            analysis_result = self._parse_analysis_response(response.text, issue_type)
            
            return {
                'success': True,
                'issue_type': issue_type,
                'analysis': analysis_result,
                'confidence': analysis_result.get('confidence', 0.8),
                'recommended_action': self._recommend_action(analysis_result, issue_type)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': "I had trouble analyzing the image. Could you try uploading it again?"
            }
    
    def _load_image(self, image_source: str) -> Image.Image:
        """Load image from file path or base64 string"""
        try:
            if os.path.exists(image_source):
                return Image.open(image_source)
            # Try base64
            image_data = base64.b64decode(image_source)
            return Image.open(io.BytesIO(image_data))
        except Exception as e:
            raise ValueError(f"Could not load image: {e}")
    
    def _build_analysis_prompt(self, issue_type: str, product_info: Dict[str, Any] = None) -> str:
        """Build specific analysis prompt based on issue type"""
        
        prompts = {
            'defect': """Analyze this product image for defects or quality issues.

Check for: stitching problems, material defects, structural issues, surface damage.

Provide in this exact format:
DEFECT PRESENT: [YES/NO]
DESCRIPTION: [detailed description]
SEVERITY: [minor/moderate/major]
USABLE: [YES/NO]
CONFIDENCE: [0-100]%""",

            'wrong_item': f"""Compare this image to expected product.

Expected: {product_info.get('name', 'N/A') if product_info else 'N/A'}
Expected Color: {product_info.get('color', 'N/A') if product_info else 'N/A'}

Provide in this exact format:
CORRECT ITEM: [YES/NO]
ACTUAL ITEM: [description]
DIFFERENCES: [list key differences]
CONFIDENCE: [0-100]%""",

            'damage': """Examine for shipping or handling damage.

Check for: package damage, crushed/bent items, broken components, mishandling signs.

Provide in this exact format:
DAMAGE PRESENT: [YES/NO]
DESCRIPTION: [detailed description]
SEVERITY: [minor/moderate/major]
FUNCTIONAL: [YES/NO]
CONFIDENCE: [0-100]%""",

            'color_mismatch': f"""Compare color in image to expected color.

Expected Color: {product_info.get('color', 'N/A') if product_info else 'N/A'}

Provide in this exact format:
COLOR MATCH: [YES/NO]
ACTUAL COLOR: [description]
DIFFERENCE: [description]
CONFIDENCE: [0-100]%""",

            'quality_issue': """Assess overall product quality.

Evaluate: material quality, construction, appearance, standards.

Provide in this exact format:
QUALITY RATING: [poor/fair/good/excellent]
CONCERNS: [list any concerns]
MEETS STANDARDS: [YES/NO]
CONFIDENCE: [0-100]%"""
        }
        
        return prompts.get(issue_type, prompts['defect'])
    
    def _parse_analysis_response(self, response_text: str, issue_type: str) -> Dict[str, Any]:
        """Parse Gemini's response into structured data"""
        
        response_lower = response_text.lower()
        
        # Check if issue confirmed
        issue_confirmed = any(keyword in response_lower for keyword in [
            'defect present: yes', 'damage present: yes', 'correct item: no',
            'color match: no', 'quality rating: poor', 'quality rating: fair'
        ])
        
        # Extract confidence
        confidence = 0.8
        if 'confidence:' in response_lower:
            try:
                conf_line = [line for line in response_text.split('\n') if 'confidence' in line.lower()][0]
                conf_value = ''.join(filter(str.isdigit, conf_line))
                confidence = float(conf_value) / 100 if conf_value else 0.8
            except:
                pass
        
        # Extract severity
        severity = 'moderate'
        if 'severity: major' in response_lower:
            severity = 'major'
        elif 'severity: minor' in response_lower:
            severity = 'minor'
        
        return {
            'issue_confirmed': issue_confirmed,
            'full_analysis': response_text,
            'severity': severity,
            'confidence': confidence,
            'summary': response_text.split('\n')[0]
        }
    
    def _recommend_action(self, analysis: Dict[str, Any], issue_type: str) -> Dict[str, Any]:
        """Recommend action based on analysis results"""
        
        if not analysis.get('issue_confirmed'):
            return {
                'action': 'no_action_needed',
                'message': "The image doesn't show a clear issue. If you're still concerned, let me know!",
                'priority': 'low'
            }
        
        severity = analysis.get('severity', 'moderate')
        
        # Action recommendations
        if issue_type in ['defect', 'damage']:
            if severity == 'major':
                return {
                    'action': 'immediate_replacement',
                    'message': "I'm sending a replacement immediately with express shipping. Keep the defective item.",
                    'compensation': '10% discount code',
                    'priority': 'high'
                }
            elif severity == 'moderate':
                return {
                    'action': 'replacement',
                    'message': "I'll send you a replacement with free return shipping.",
                    'priority': 'medium'
                }
            else:  # minor
                return {
                    'action': 'partial_refund_or_replacement',
                    'message': "I can send a replacement or offer 20% partial refund if you'd like to keep it.",
                    'priority': 'medium'
                }
        
        elif issue_type == 'wrong_item':
            return {
                'action': 'immediate_correct_item',
                'message': "I'm so sorry! Shipping the correct item now with express shipping. Keep the wrong item.",
                'priority': 'high'
            }
        
        elif issue_type == 'color_mismatch':
            return {
                'action': 'exchange_or_refund',
                'message': "I can exchange it for the correct color or process a full refund.",
                'priority': 'medium'
            }
        
        else:  # quality_issue
            return {
                'action': 'refund_or_replacement',
                'message': "This doesn't meet our quality standards. I can send a replacement or process a refund.",
                'priority': 'medium'
            }
    
    def verify_return_condition(self, image_path: str, product_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify product condition for returns (checks if item is in returnable condition)
        
        Args:
            image_path: Path to image of product being returned
            product_info: Product information
            
        Returns:
            Verification results
        """
        prompt = """Verify if this product is in acceptable return condition.

Check for:
- Item appears unused/unworn
- Tags still attached (if visible)
- No visible wear or damage
- Original condition
- Clean and presentable

Provide in this exact format:
RETURNABLE: [YES/NO]
CONDITION: [excellent/good/fair/poor]
TAGS ATTACHED: [YES/NO/NOT VISIBLE]
WEAR VISIBLE: [YES/NO]
NOTES: [any relevant observations]
CONFIDENCE: [0-100]%"""

        try:
            image = self._load_image(image_path)
            response = self.model.generate_content([prompt, image])
            
            response_lower = response.text.lower()
            returnable = 'returnable: yes' in response_lower
            
            return {
                'success': True,
                'returnable': returnable,
                'analysis': response.text,
                'recommendation': 'approve_return' if returnable else 'partial_refund_only'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'recommendation': 'manual_review'
            }
