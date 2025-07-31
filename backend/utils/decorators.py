"""Common decorators for error handling and logging."""

import logging
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)


T = TypeVar('T')


def handle_errors(
    operation_name: str,
    return_on_error: Optional[Any] = None,
    raise_on_error: bool = True
) -> Callable[[Callable[..., T]], Callable[..., Union[T, Any]]]:
    """Generic decorator to handle async/sync operation errors with logging.
    
    Args:
        operation_name: Human-readable name of the operation for logging
        return_on_error: Value to return on error (if raise_on_error=False)
        raise_on_error: Whether to re-raise exceptions or return error value
        
    Returns:
        Decorated function with error handling and logging
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Union[T, Any]]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Union[T, Any]:
            try:
                result = await func(*args, **kwargs)
                # Only log success for operations that return meaningful results
                if result is not None and result is not True:
                    logger.info(f"{operation_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Failed to {operation_name.lower()}: {e}")
                if raise_on_error:
                    raise
                return return_on_error
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Union[T, Any]:
            try:
                result = func(*args, **kwargs)
                if result is not None and result is not True:
                    logger.info(f"{operation_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Failed to {operation_name.lower()}: {e}")
                if raise_on_error:
                    raise
                return return_on_error
        
        # Return appropriate wrapper based on whether function is async
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        return sync_wrapper
    
    return decorator
