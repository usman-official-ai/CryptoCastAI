"""Logging utilities."""

import logging
import sys
from typing import Optional


def setup_logging(
    name: str = "cryptocastai",
    log_level: str = "INFO",
    console_output: bool = True
) -> logging.Logger:
    """Setup logging configuration."""
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


class LoggerMixin:
    """Mixin class for logging."""
    
    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(self.__class__.__name__)
        return self._logger
    
    def log_info(self, message: str, extra: dict = None):
        self.logger.info(message, extra={'extra_data': extra} if extra else {})
    
    def log_warning(self, message: str, extra: dict = None):
        self.logger.warning(message, extra={'extra_data': extra} if extra else {})
    
    def log_error(self, message: str, extra: dict = None):
        self.logger.error(message, extra={'extra_data': extra} if extra else {})
    
    def log_debug(self, message: str, extra: dict = None):
        self.logger.debug(message, extra={'extra_data': extra} if extra else {})