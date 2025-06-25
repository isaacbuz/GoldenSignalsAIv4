"""Dependency Injection Container"""

from typing import Dict, Type, Any

class DIContainer:
    def __init__(self):
        self._services = {}
        self._singletons = {}
        
    def register(self, interface: Type, implementation: Type, singleton: bool = False):
        """Register service"""
        self._services[interface] = (implementation, singleton)
        
    def resolve(self, interface: Type) -> Any:
        """Resolve service"""
        if interface in self._services:
            impl, is_singleton = self._services[interface]
            if is_singleton:
                if interface not in self._singletons:
                    self._singletons[interface] = impl()
                return self._singletons[interface]
            return impl()
        raise ValueError(f"Service {interface} not registered")

# Global container
container = DIContainer()
