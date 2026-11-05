from typing import Any

class _Config:
    def update(self, *args: Any, **kwargs: Any) -> None: ...

config: _Config

def jit(*args: Any, **kwargs: Any) -> Any: ...
def vmap(*args: Any, **kwargs: Any) -> Any: ...

class _Lax:
    def cond(self, *args: Any, **kwargs: Any) -> Any: ...

lax: _Lax
