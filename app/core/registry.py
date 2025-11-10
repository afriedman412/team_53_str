# app/core/registry.py
from __future__ import annotations
from typing import Optional
from threading import RLock
from app.core.store import DataStore


class _NotInitialized(RuntimeError):
    pass


_store: Optional[DataStore] = None
_lock = RLock()


def set_store(store: DataStore) -> None:
    """Called once during startup (per worker)."""
    global _store
    with _lock:
        _store = store


def get_store() -> DataStore:
    """
    Access the singleton DataStore for this worker.
    Raises if called before startup (e.g., at import time).
    """
    s = _store
    if s is None:
        raise _NotInitialized("DataStore not initialized yet (startup not completed).")
    return s
