"""Execution context shared by the operation engines."""
from __future__ import annotations
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

SEED = 42
MODEL_CACHE_SIZE = 8
# Bump when training/selection logic changes so stale cached models are never reused.
MODEL_CONFIG_VERSION = "3"


class ModelCache:
    """Small per-session LRU cache of trained models, keyed by (task, target, features, filters, config)."""

    def __init__(self, size: int = MODEL_CACHE_SIZE):
        self._d: "OrderedDict[Any, Any]" = OrderedDict()
        self._lock = threading.RLock()
        self.size = size
        self.hits = 0
        self.misses = 0

    def get_or_create(self, key, factory: Callable[[], Any]):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                self.hits += 1
                return self._d[key], True
        value = factory()           # train outside the lock for other keys; sessions are single-user
        with self._lock:
            self.misses += 1
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.size:
                self._d.popitem(last=False)
        return value, False

    def clear(self):
        with self._lock:
            self._d.clear()
            self.hits = self.misses = 0

    def __len__(self):
        return len(self._d)


@dataclass
class ExecContext:
    roles: Dict[str, Dict[str, Any]]
    profile: Dict[str, Any]
    cache: ModelCache = field(default_factory=ModelCache)
    seed: int = SEED
    dataset_id: str = ""
    semantics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)      # per-dataset computed artefacts (relationships, quality, ...)
    full_data: Any = None                                         # the whole (unfiltered) dataset, for whole-dataset operations

    def memo(self, key: str, factory: Callable[[], Any]) -> Any:
        """Compute a dataset-level artefact once per dataset version."""
        if key not in self.artifacts:
            self.artifacts[key] = factory()
        return self.artifacts[key]
