"""
Image Processing Utilities
"""

import os
import base64
from io import BytesIO
from typing import Tuple, Optional
from PIL import Image

class ImageProcessor:
    """Image processing and validation utilities"""
    
    def __init__(self, max_size_mb: int = 5, supported_formats: list = None):
        """
        Initialize image processor
        
        Args:
            max_size_mb: Maximum file size in MB
            supported_formats: List of supported formats
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.supported_formats = supported_formats or ['JPEG', 'PNG', 'JPG', 'WEBP']
    
    def load_image(self, image_source: str) -> Image.Image:
        """
        Load image from file path or base64 string
        
        Args:
            image_source: File path or base64 encoded string
            
        Returns:
            PIL Image object
        """
        # Try loading from file path
        if os.path.exists(image_source):
            return Image.open(image_source)
        
        # Try decoding as base64
        try:
            # Remove data URL prefix if present
            if ',' in image_source:
                image_source = image_source.split(',')[1]
            
            image_data = base64.b64decode(image_source)
            return Image.open(BytesIO(image_data))
        except Exception as e:
            raise ValueError(f"Could not load image: {e}")
    
    def validate_image(self, image: Image.Image) -> Tuple[bool, Optional[str]]:
        """
        Validate image format and size
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check format
        if image.format not in self.supported_formats:
            return False, f"Unsupported format. Supported: {', '.join(self.supported_formats)}"
        
        # Check size
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format=image.format)
        size_bytes = img_byte_arr.tell()
        
        if size_bytes > self.max_size_bytes:
            return False, f"Image too large. Maximum size: {self.max_size_bytes / (1024*1024)}MB"
        
        return True, None
    
    def resize_image(self, image: Image.Image, max_width: int = 1024, 
                    max_height: int = 1024) -> Image.Image:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: PIL Image object
            max_width: Maximum width
            max_height: Maximum height
            
        Returns:
            Resized image
        """
        # Calculate new dimensions
        width, height = image.size
        
        if width <= max_width and height <= max_height:
            return image
        
        # Calculate scaling factor
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def convert_to_base64(self, image: Image.Image, format: str = 'JPEG') -> str:
        """
        Convert PIL Image to base64 string
        
        Args:
            image: PIL Image object
            format: Output format
            
        Returns:
            Base64 encoded string
        """
        buffered = BytesIO()
        
        # Convert RGBA to RGB if saving as JPEG
        if format == 'JPEG' and image.mode == 'RGBA':
            image = image.convert('RGB')
        
        image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str
    
    def compress_image(self, image: Image.Image, quality: int = 85) -> Image.Image:
        """
        Compress image to reduce file size
        
        Args:
            image: PIL Image object
            quality: JPEG quality (1-100)
            
        Returns:
            Compressed image
        """
        output = BytesIO()
        
        # Convert to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        image.save(output, format='JPEG', quality=quality, optimize=True)
        output.seek(0)
        
        return Image.open(output)
    
    def get_image_info(self, image: Image.Image) -> dict:
        """
        Get image metadata
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with image info
        """
        return {
            'format': image.format,
            'mode': image.mode,
            'size': image.size,
            'width': image.width,
            'height': image.height
        }
    
    def save_image(self, image: Image.Image, output_path: str, quality: int = 95):
        """
        Save image to file
        
        Args:
            image: PIL Image object
            output_path: Output file path
            quality: JPEG quality
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Determine format from file extension
        _, ext = os.path.splitext(output_path)
        format_map = {'.jpg': 'JPEG', '.jpeg': 'JPEG', '.png': 'PNG', '.webp': 'WEBP'}
        save_format = format_map.get(ext.lower(), 'JPEG')
        
        # Convert RGBA to RGB for JPEG
        if save_format == 'JPEG' and image.mode == 'RGBA':
            image = image.convert('RGB')
        
        image.save(output_path, format=save_format, quality=quality)
