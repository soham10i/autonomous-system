#!/usr/bin/env python3

"""
Centralized Logging Configuration for ROSBot SLAM System
Provides structured logging with performance monitoring and error tracking
"""

import logging
import logging.handlers
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional

class SLAMLogger:
    """Centralized logger for SLAM system with performance monitoring"""
    
    _instance = None
    _loggers: Dict[str, logging.Logger] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SLAMLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.setup_logging()
            self._initialized = True
            self.performance_data: Dict[str, Any] = {}
            self.error_counts: Dict[str, int] = {}
    
    def setup_logging(self):
        """Setup logging configuration for all SLAM modules"""
        # Create logs directory
        log_dir = "slam_logs"
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
            except:
                log_dir = "."  # Fallback to current directory
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler for all logs
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"slam_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Error handler for critical issues
        error_handler = logging.FileHandler(
            os.path.join(log_dir, f"slam_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        )
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(file_formatter)
        
        # Add handlers to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(error_handler)
        
        print(f"📝 Logging system initialized - Logs saved to: {log_dir}")
    
    def get_logger(self, module_name: str) -> logging.Logger:
        """Get logger for specific SLAM module"""
        if module_name not in self._loggers:
            logger = logging.getLogger(f"SLAM.{module_name}")
            logger.setLevel(logging.DEBUG)
            self._loggers[module_name] = logger
            
            # Initialize performance tracking
            self.performance_data[module_name] = {
                'call_count': 0,
                'total_time': 0.0,
                'avg_time': 0.0,
                'max_time': 0.0,
                'min_time': float('inf'),
                'last_call': None
            }
            self.error_counts[module_name] = 0
        
        return self._loggers[module_name]
    
    def log_performance(self, module_name: str, function_name: str, execution_time: float):
        """Log performance metrics for analysis"""
        if module_name not in self.performance_data:
            self.performance_data[module_name] = {
                'call_count': 0, 'total_time': 0.0, 'avg_time': 0.0,
                'max_time': 0.0, 'min_time': float('inf'), 'last_call': None
            }
        
        perf = self.performance_data[module_name]
        perf['call_count'] += 1
        perf['total_time'] += execution_time
        perf['avg_time'] = perf['total_time'] / perf['call_count']
        perf['max_time'] = max(perf['max_time'], execution_time)
        perf['min_time'] = min(perf['min_time'], execution_time)
        perf['last_call'] = time.time()
        
        logger = self.get_logger(module_name)
        if execution_time > 0.1:  # Log slow operations
            logger.warning(f"{function_name} took {execution_time:.3f}s (slow operation)")
        else:
            logger.debug(f"{function_name} completed in {execution_time:.3f}s")
    
    def log_error(self, module_name: str, error: Exception, context: str = ""):
        """Log errors with context for debugging"""
        self.error_counts[module_name] = self.error_counts.get(module_name, 0) + 1
        
        logger = self.get_logger(module_name)
        logger.error(f"Error in {context}: {type(error).__name__}: {str(error)}")
        logger.debug(f"Error count for {module_name}: {self.error_counts[module_name]}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all modules"""
        return {
            'performance': self.performance_data,
            'errors': self.error_counts,
            'timestamp': datetime.now().isoformat()
        }
    
    def print_performance_report(self):
        """Print performance report to console"""
        print("\n" + "="*60)
        print("📊 SLAM SYSTEM PERFORMANCE REPORT")
        print("="*60)
        
        for module, perf in self.performance_data.items():
            if perf['call_count'] > 0:
                print(f"🔧 {module}:")
                print(f"   Calls: {perf['call_count']}")
                print(f"   Avg Time: {perf['avg_time']:.3f}s")
                print(f"   Max Time: {perf['max_time']:.3f}s")
                print(f"   Errors: {self.error_counts.get(module, 0)}")
        print("="*60)



def get_slam_logger(module_name: str) -> logging.Logger:
    """Convenience function to get SLAM logger"""
    return SLAMLogger().get_logger(module_name)

# Initialize global logging system
_slam_logger = SLAMLogger()

def performance_monitor(func):
    """Decorator to log performance of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Attempt to get module name from the function's __module__ attribute
        module_name = func.__module__.split('.')[-1] if hasattr(func, '__module__') else 'unknown_module'
        
        SLAMLogger().log_performance(module_name, func.__name__, execution_time)
        return result
    return wrapper


