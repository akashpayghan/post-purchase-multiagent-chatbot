"""
Logging Configuration
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logger(
    name: str = 'ai_guardian',
    level: str = 'INFO',
    log_to_file: bool = True,
    log_dir: str = 'logs'
) -> logging.Logger:
    """
    Setup and configure logger
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Create log file with date
        log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = 'ai_guardian') -> logging.Logger:
    """
    Get existing logger or create new one
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up
    if not logger.handlers:
        return setup_logger(name)
    
    return logger

class LoggerContext:
    """Context manager for temporary log level changes"""
    
    def __init__(self, logger: logging.Logger, level: str):
        """
        Initialize logger context
        
        Args:
            logger: Logger instance
            level: Temporary log level
        """
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.old_level = logger.level
    
    def __enter__(self):
        """Set temporary log level"""
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original log level"""
        self.logger.setLevel(self.old_level)

# Convenience functions
def log_agent_action(logger: logging.Logger, agent_name: str, 
                    action: str, details: Optional[dict] = None):
    """
    Log agent action with structured format
    
    Args:
        logger: Logger instance
        agent_name: Name of the agent
        action: Action performed
        details: Additional details
    """
    msg = f"[{agent_name}] {action}"
    if details:
        msg += f" | {details}"
    logger.info(msg)

def log_conversation(logger: logging.Logger, conversation_id: str, 
                    message: str, role: str = 'user'):
    """
    Log conversation message
    
    Args:
        logger: Logger instance
        conversation_id: Conversation identifier
        message: Message content
        role: Message role (user/assistant)
    """
    logger.info(f"[Conversation:{conversation_id}] [{role}] {message}")

def log_error(logger: logging.Logger, error: Exception, 
             context: Optional[str] = None):
    """
    Log error with context
    
    Args:
        logger: Logger instance
        error: Exception object
        context: Additional context
    """
    msg = f"Error: {str(error)}"
    if context:
        msg = f"{context} | {msg}"
    logger.error(msg, exc_info=True)
