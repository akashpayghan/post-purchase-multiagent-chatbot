"""
Utility modules for the Smart Post-Purchase AI Guardian
"""

from .embeddings import EmbeddingGenerator
from .image_processing import ImageProcessor
from .text_processing import TextProcessor
from .logger import setup_logger, get_logger

__all__ = [
    'EmbeddingGenerator',
    'ImageProcessor',
    'TextProcessor',
    'setup_logger',
    'get_logger'
]

__version__ = '1.0.0'
