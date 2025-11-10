# app/deps.py
from __future__ import annotations
from typing import Optional
from app.core.store import DataStore

_STORE: Optional[DataStore] = None


def set_store(store: DataStore) -> None:
    """Set once at startup."""
    global _STORE
    _STORE = store


def get_store(raise_if_missing: bool = True) -> Optional[DataStore]:
    """Fetch the current store (or None if not set)."""
    if _STORE is None and raise_if_missing:
        raise RuntimeError("Global DataStore is not initialized yet.")
    return _STORE
