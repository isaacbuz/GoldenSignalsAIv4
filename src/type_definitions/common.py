"""
Common type definitions
"""

from typing import TypeVar, Generic, Optional, List, Dict, Any, Union, Callable, Awaitable
from datetime import datetime

# Type variables
T = TypeVar('T')
E = TypeVar('E', bound=Exception)

# Result type for error handling
class Result(Generic[T, E]):
    def __init__(self, value: Optional[T] = None, error: Optional[E] = None):
        self.value = value
        self.error = error
        
    @property
    def is_ok(self) -> bool:
        return self.error is None
        
    @property
    def is_err(self) -> bool:
        return self.error is not None
        
# Common type aliases
JsonDict = Dict[str, Any]
JsonList = List[JsonDict]
AsyncFunc = Callable[..., Awaitable[T]]
SyncFunc = Callable[..., T]

# API Response types
class APIResponse(TypedDict):
    status: int
    data: Optional[Any]
    error: Optional[str]
    timestamp: datetime
