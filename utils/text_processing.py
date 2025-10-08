"""
Text Processing Utilities
"""

import re
from typing import List, Dict, Any

class TextProcessor:
    """Text processing and NLP utilities"""
    
    def __init__(self):
        """Initialize text processor"""
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text.strip()
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords
            
        Returns:
            List of keywords
        """
        # Tokenize and clean
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stop words and short words
        keywords = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        # Count frequency
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, freq in sorted_keywords[:max_keywords]]
    
    def detect_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Simple sentiment detection based on keywords
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment info
        """
        text_lower = text.lower()
        
        # Positive keywords
        positive_words = ['happy', 'great', 'excellent', 'love', 'perfect', 'amazing', 
                         'thank', 'satisfied', 'good', 'wonderful', 'fantastic']
        
        # Negative keywords
        negative_words = ['unhappy', 'bad', 'terrible', 'hate', 'awful', 'disappointed',
                         'angry', 'frustrated', 'poor', 'horrible', 'worst', 'unsatisfied']
        
        # Urgent keywords
        urgent_words = ['urgent', 'immediately', 'asap', 'emergency', 'now', 'quickly']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        urgent_count = sum(1 for word in urgent_words if word in text_lower)
        
        # Calculate sentiment score (-1 to 1)
        total = pos_count + neg_count
        if total == 0:
            sentiment_score = 0
        else:
            sentiment_score = (pos_count - neg_count) / total
        
        # Determine sentiment label
        if sentiment_score > 0.3:
            sentiment = 'positive'
        elif sentiment_score < -0.3:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'score': sentiment_score,
            'is_urgent': urgent_count > 0,
            'positive_count': pos_count,
            'negative_count': neg_count
        }
    
    def truncate_text(self, text: str, max_length: int = 100, 
                     suffix: str = '...') -> str:
        """
        Truncate text to maximum length
        
        Args:
            text: Input text
            max_length: Maximum length
            suffix: Suffix to add if truncated
            
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)].strip() + suffix
    
    def extract_order_number(self, text: str) -> str:
        """
        Extract order number from text
        
        Args:
            text: Input text
            
        Returns:
            Order number or empty string
        """
        # Common order number patterns
        patterns = [
            r'#?(\d{6,10})',  # 6-10 digits
            r'ORD[- ]?(\d{6,10})',  # ORD prefix
            r'ORDER[- ]?(\d{6,10})',  # ORDER prefix
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1) if match.lastindex else match.group(0)
        
        return ""
    
    def extract_email(self, text: str) -> str:
        """
        Extract email address from text
        
        Args:
            text: Input text
            
        Returns:
            Email address or empty string
        """
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(pattern, text)
        return match.group(0) if match else ""
    
    def extract_phone(self, text: str) -> str:
        """
        Extract phone number from text
        
        Args:
            text: Input text
            
        Returns:
            Phone number or empty string
        """
        # US phone number patterns
        patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # 123-456-7890
            r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b',  # (123) 456-7890
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, 
                   overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for period, question mark, or exclamation
                last_sentence = max(
                    text.rfind('.', start, end),
                    text.rfind('?', start, end),
                    text.rfind('!', start, end)
                )
                if last_sentence > start:
                    end = last_sentence + 1
            
            chunks.append(text[start:end].strip())
            start = end - overlap
        
        return chunks
    
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text
        
        Args:
            text: Input text
            
        Returns:
            Text without URLs
        """
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def count_words(self, text: str) -> int:
        """Count words in text"""
        return len(text.split())
    
    def count_sentences(self, text: str) -> int:
        """Count sentences in text"""
        return len(re.findall(r'[.!?]+', text))
