import logging
import os
import sys
from datetime import datetime
from typing import Optional

class AppLogger:
    """
    Enterprise-grade logging manager for TalkingData.
    
    This class handles the initialization and configuration of logging
    for the entire application. It outputs logs to both the console
    (stdout) and a dynamically created file based on the current date.
    
    Attributes:
        logger (logging.Logger): The configured logger instance.
        log_dir (str): The directory where log files are stored.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Implement the Singleton pattern to ensure only one logger instance exists."""
        if not cls._instance:
            cls._instance = super(AppLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, log_level: int = logging.INFO, log_dir: str = "logs"):
        """
        Initializes the logger if it hasn't been initialized already.
        
        Args:
            log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
            log_dir (str): Directory to save log files.
        """
        if hasattr(self, 'logger_initialized') and self.logger_initialized:
            return
            
        self.log_dir = log_dir
        self.log_level = log_level
        self._create_log_directory()
        self.logger = self._setup_logger()
        self.logger_initialized = True

    def _create_log_directory(self) -> None:
        """Creates the logging directory if it doesn't exist."""
        if not os.path.exists(self.log_dir):
            try:
                os.makedirs(self.log_dir)
            except OSError as e:
                print(f"Failed to create log directory {self.log_dir}: {e}")
                
    def _setup_logger(self) -> logging.Logger:
        """
        Configures the root logger with File and Stream handlers.
        
        Returns:
            logging.Logger: The fully configured logger.
        """
        logger = logging.getLogger("TalkingData")
        logger.setLevel(self.log_level)
        
        # Prevent duplicate logs if handlers already exist
        if logger.handlers:
            return logger
            
        # Create formatter
        log_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File Handler (Daily rotating log file logic could be added here)
        current_date = datetime.now().strftime("%Y-%m-%d")
        file_path = os.path.join(self.log_dir, f"talkingdata_{current_date}.log")
        
        try:
            file_handler = logging.FileHandler(file_path, encoding='utf-8')
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(log_format)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to setup file handler: {e}")
            
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
        
        return logger
        
    @staticmethod
    def get_logger(name: Optional[str] = None) -> logging.Logger:
        """
        Retrieves a logger instance, ensuring the AppLogger is initialized first.
        
        Args:
            name (Optional[str]): Optional sub-logger name.
            
        Returns:
            logging.Logger: The logger instance.
        """
        # Ensure base logger is set up
        AppLogger() 
        if name:
            return logging.getLogger(f"TalkingData.{name}")
        return logging.getLogger("TalkingData")

# Initialize the global logger instance
logger = AppLogger.get_logger()
