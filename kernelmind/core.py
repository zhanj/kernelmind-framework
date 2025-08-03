import asyncio, warnings, copy, time, inspect
from typing import Any, Dict, Optional

class Point:
    def __init__(self, retries: int = 1, wait: float = 0, parallel: bool = False, is_async: bool = False):
        self.next_points: Dict[str, 'Point'] = {}
        self.retries, self.wait, self.parallel, self.is_async = retries, wait, parallel, is_async
        if is_async and not inspect.iscoroutinefunction(self.process):
            warnings.warn("is_async=True but process method is not async. This may cause runtime errors.")
    
    def next(self, node: 'Point', action: str = "default") -> 'Point': 
        if action in self.next_points: warnings.warn(f"Overwriting successor for action '{action}'")
        self.next_points[action] = node; return node
    def __rshift__(self, other: 'Point') -> 'Point': return self.next(other)
    def __sub__(self, action: str) -> '_Transition': 
        if not isinstance(action, str): raise TypeError("Action must be string")
        return _Transition(self, action)
    
    def load(self, memory: Dict[str, Any]) -> Any: return "start"
    def process(self, item: Any) -> Any: pass
    def save(self, memory: Dict[str, Any], prep_result: Any, exec_result: Any) -> Optional[str]: pass
    def on_fail(self, item: Any, exc: Exception) -> Any: 
        raise exc
    
    def _sync_single(self, item: Any) -> Any:
        for i in range(self.retries):
            try: return self.process(item)
            except Exception as e:
                if i == self.retries-1: return self.on_fail(item, e)
                if self.wait > 0: time.sleep(self.wait)
    
    def _sync_batch(self, items: list) -> list:
        return [self._sync_single(item) for item in items]
    
    async def _async_single(self, item: Any) -> Any:
        for i in range(self.retries):
            try:
                if inspect.iscoroutinefunction(self.process):
                    return await self.process(item)
                else:
                    result = self.process(item)
                    return await result if inspect.iscoroutine(result) else result
            except Exception as e:
                if i == self.retries-1:
                    fallback_result = self.on_fail(item, e)
                    return await fallback_result if inspect.iscoroutine(fallback_result) else fallback_result
                if self.wait > 0: await asyncio.sleep(self.wait)
    
    async def _async_batch(self, items: list) -> list:
        if self.parallel:
            return await asyncio.gather(*[self._async_single(item) for item in items])
        else:
            return [await self._async_single(item) for item in items]
    
    def run(self, memory: Dict[str, Any]) -> Optional[str]:
        if self.next_points: warnings.warn("Point has next_points but running independently. Use Line for connected nodes.")
        
        prep_result = self.load(memory)
        if prep_result is None: return self.save(memory, prep_result, None)
        
        if self.is_async:
            if isinstance(prep_result, (list, tuple)):
                exec_result = asyncio.run(self._async_batch(list(prep_result)))
            else:
                exec_result = asyncio.run(self._async_single(prep_result))
        else:
            if inspect.iscoroutinefunction(self.process):
                raise RuntimeError("Cannot call async process in sync context. Set is_async=True")
            if isinstance(prep_result, (list, tuple)):
                exec_result = self._sync_batch(list(prep_result))
            else:
                exec_result = self._sync_single(prep_result)
        
        return self.save(memory, prep_result, exec_result)

class _Transition:
    def __init__(self, src: Point, action: str): self.src, self.action = src, action
    def __rshift__(self, target: Point) -> Point: return self.src.next(target, self.action)

class Line(Point):
    def __init__(self, entry: Optional[Point] = None, **kwargs):
        super().__init__(**kwargs)
        self.entry = entry
    
    def run(self, memory: Dict[str, Any]) -> Optional[str]:
        current_point = copy.copy(self.entry) if self.entry else None
        last_step = None       
        while current_point:
            last_step = current_point.run(memory)
            next_node = current_point.next_points.get(last_step or "default")
            if not next_node : 
                print(f"Line ends: no successor for action '{last_step}' in {list(current_point.next_points.keys())}")
                break
            current_point = copy.copy(next_node) if next_node else None
        return last_step