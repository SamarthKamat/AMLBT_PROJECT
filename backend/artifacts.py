"""Lazily computed, per-dataset artefacts shared by the engines (semantics, relationships, quality)."""
from __future__ import annotations
from typing import Any, Dict, Optional

import pandas as pd

from context import ExecContext
from data_quality import assess_quality
from relationships import analyse_relationships
from semantics import infer_semantics


def dataset(ctx: ExecContext, fallback: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    return ctx.full_data if ctx.full_data is not None else fallback


def get_semantics(ctx: ExecContext, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if not ctx.semantics:
        ctx.semantics = infer_semantics(dataset(ctx, df), ctx.profile)
    return ctx.semantics


def get_relationships(ctx: ExecContext, df: pd.DataFrame) -> Dict[str, Any]:
    d = dataset(ctx, df)
    return ctx.memo("relationships", lambda: analyse_relationships(d, ctx.profile, get_semantics(ctx, d)))


def get_quality(ctx: ExecContext, df: pd.DataFrame) -> Dict[str, Any]:
    d = dataset(ctx, df)
    return ctx.memo("quality", lambda: assess_quality(d, ctx.profile, get_semantics(ctx, d)))
