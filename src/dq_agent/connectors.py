"""Data connectors — the first step of both workflows.

Every connector loads into a Polars DataFrame, the internal representation for all
profiling (scoping time) and rule execution (run time). Never hand raw cursors or
pandas frames to the profiler or engine."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


def load_csv(path: str | Path) -> pl.DataFrame:
    return pl.read_csv(path, try_parse_dates=True)


def load_parquet(path: str | Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def load_postgres(uri: str, *, table: str | None = None, query: str | None = None) -> pl.DataFrame:
    """Load from Postgres via ConnectorX. Provide either a table name (optionally
    schema-qualified) or a full SQL query, not both."""
    if (table is None) == (query is None):
        raise ValueError("provide exactly one of 'table' or 'query'")
    if table is not None:
        if not _IDENTIFIER.match(table):
            raise ValueError(f"invalid table name: {table!r}")
        query = f"SELECT * FROM {table}"

    try:
        return pl.read_database_uri(query, uri, engine="connectorx")
    except ImportError as exc:
        raise ImportError(
            "the Postgres connector requires connectorx — install with: uv sync --extra postgres"
        ) from exc
