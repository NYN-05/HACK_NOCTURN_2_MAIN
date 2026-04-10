from __future__ import annotations

import copy
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class CacheClaim:
    state: str
    value: Dict[str, Any] | None = None
    future: asyncio.Future | None = None


class RequestCache:
    def __init__(self, max_items: int) -> None:
        self._max_items = max_items
        self._lock = asyncio.Lock()
        self._entries: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._inflight: Dict[str, asyncio.Future] = {}

    async def claim(self, key: str) -> CacheClaim:
        loop = asyncio.get_running_loop()
        async with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return CacheClaim(state="hit", value=copy.deepcopy(self._entries[key]))

            future = self._inflight.get(key)
            if future is None:
                future = loop.create_future()
                self._inflight[key] = future
                return CacheClaim(state="owner", future=future)

            return CacheClaim(state="wait", future=future)

    async def settle(self, key: str, value: Dict[str, Any] | None = None, exc: Exception | None = None) -> None:
        async with self._lock:
            future = self._inflight.pop(key, None)
            if exc is None and value is not None:
                self._entries[key] = copy.deepcopy(value)
                self._entries.move_to_end(key)
                while len(self._entries) > self._max_items:
                    self._entries.popitem(last=False)

        if future is not None and not future.done():
            if exc is None and value is not None:
                future.set_result(copy.deepcopy(value))
            elif exc is not None:
                future.set_exception(exc)