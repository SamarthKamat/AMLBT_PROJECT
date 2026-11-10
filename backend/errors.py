"""Structured errors. Every failure the user can see carries a code, a message and a retry hint."""
from __future__ import annotations
from typing import Any, Dict, Optional

ERROR_CODES = {
    "GEMINI_UNAVAILABLE": 503,
    "GEMINI_MODEL_UNAVAILABLE": 503,
    "INVALID_PLAN": 422,
    "INVALID_COLUMN": 422,
    "INSUFFICIENT_DATA": 422,
    "UNSUPPORTED_OPERATION": 422,
    "MODEL_TRAINING_ERROR": 500,
    "DATASET_PARSE_ERROR": 400,
    "FILE_TOO_LARGE": 413,
    "BACKEND_ERROR": 500,
    # request-level problems (still structured)
    "NO_DATASET": 400,
    "BAD_REQUEST": 400,
    "SESSION_NOT_FOUND": 404,
    "UNAUTHORIZED": 401,
    "FORBIDDEN": 403,
    "RATE_LIMITED": 429,
    "JOB_NOT_FOUND": 404,
}


class UDIError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False,
                 details: Optional[Dict[str, Any]] = None, status: Optional[int] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details or {}
        self.status = status or ERROR_CODES.get(code, 500)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"code": self.code, "error": self.message, "retryable": self.retryable}
        if self.details:
            out["details"] = self.details
        return out
