from __future__ import annotations

import polars as pl


class VintageMetrics:
    """Provide placeholder vintage and roll-rate helpers."""

    @staticmethod
    def roll_rate(frame: pl.DataFrame, month_col: str, status_col: str) -> pl.DataFrame:
        return frame.group_by([month_col, status_col]).len().sort([month_col, status_col])

    @staticmethod
    def vintage_analysis(frame: pl.DataFrame, vintage_col: str, default_col: str) -> pl.DataFrame:
        return frame.group_by([vintage_col, default_col]).len().sort([vintage_col, default_col])
