"""Data connectors — the first step of both workflows.

Every connector loads into a Polars DataFrame, the internal representation for all
profiling (scoping time) and rule execution (run time). Never hand raw cursors or
pandas frames to the profiler or engine."""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import polars as pl

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")

# TABLESAMPLE SYSTEM is block-level and approximate, so oversample the target fraction
# a little before the exact LIMIT, to make filling the cap likely; floor the fraction so
# a huge table never rounds to 0%.
_SAMPLE_OVERSHOOT = 1.5
_MIN_SAMPLE_PCT = 0.000001


def load_csv(path: str | Path) -> pl.DataFrame:
    return pl.read_csv(path, try_parse_dates=True)


def load_parquet(path: str | Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def load_postgres(
    uri: str,
    *,
    table: str | None = None,
    query: str | None = None,
    sample_rows: int | None = None,
) -> pl.DataFrame:
    """Load from Postgres via ConnectorX. Provide either a table name (optionally
    schema-qualified) or a full SQL query, not both.

    `sample_rows` caps transfer with a random `ORDER BY random() LIMIT` pushed down to
    the database, so only that many rows cross the wire. This is an unbiased sample but
    the server still scans (and sorts) the whole table — fine for a modestly sized table
    or when you specifically need a uniform sample. For large tables prefer
    `load_postgres_profiling`, which estimates the size first and uses block-level
    `TABLESAMPLE` to avoid the full scan. Profiling any sample yields *estimated*
    statistics — pass `sampled=True` to `profiler.profile`. Sampling applies only to
    `table`; a custom `query` already controls its own size."""
    if (table is None) == (query is None):
        raise ValueError("provide exactly one of 'table' or 'query'")
    if table is not None:
        if not _IDENTIFIER.match(table):
            raise ValueError(f"invalid table name: {table!r}")
        query = f"SELECT * FROM {table}"
        if sample_rows is not None:
            query += f" ORDER BY random() LIMIT {int(sample_rows)}"
    elif sample_rows is not None:
        raise ValueError("sample_rows applies to 'table', not a custom 'query'")

    return _read(query, uri)


def estimate_row_count(uri: str, table: str) -> int | None:
    """Estimate a table's row count from the planner statistics (`pg_class.reltuples`).
    This is instant — it reads a catalog estimate, not the table — unlike `COUNT(*)`,
    which scans. Returns None when the planner has no estimate yet (e.g. a table never
    analyzed reports `reltuples = -1`); callers should then sample defensively rather
    than trust a zero."""
    if not _IDENTIFIER.match(table):
        raise ValueError(f"invalid table name: {table!r}")
    schema, _, name = table.rpartition(".")
    schema = schema or "public"
    query = (
        "SELECT c.reltuples::bigint AS estimate FROM pg_class c "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        f"WHERE c.relname = '{name}' AND n.nspname = '{schema}' "
        "AND c.relkind IN ('r', 'p', 'm')"
    )
    result = _read(query, uri)
    if result.is_empty():
        return None
    estimate = result["estimate"][0]
    return int(estimate) if estimate is not None and estimate > 0 else None


class ProfilingLoad(NamedTuple):
    """An adaptive load: the DataFrame plus whether it is a sample (feed straight into
    `profiler.profile(df, ..., sampled=load.sampled)`) and the planner estimate used."""

    df: pl.DataFrame
    sampled: bool
    estimated_rows: int | None


def load_postgres_profiling(
    uri: str, *, table: str, max_rows: int, seed: int = 0
) -> ProfilingLoad:
    """Load a table sized for profiling. Estimates the row count from the planner first
    (instant); if it fits under `max_rows` the table is loaded whole, otherwise a
    block-level `TABLESAMPLE` of about `max_rows` rows is pulled — cheap on large tables
    because it reads a fraction of disk blocks instead of scanning. The decision is
    deterministic (the caller sets `max_rows`, not the LLM), and `REPEATABLE (seed)`
    makes the sample reproducible. When the estimate is unknown, it falls back to a plain
    `LIMIT max_rows` — a bounded but storage-order-biased read — and still flags the
    load as sampled."""
    estimate = estimate_row_count(uri, table)

    if estimate is not None and estimate <= max_rows:
        return ProfilingLoad(load_postgres(uri, table=table), sampled=False, estimated_rows=estimate)

    if estimate is not None:
        pct = min(100.0, max(_MIN_SAMPLE_PCT, max_rows / estimate * 100.0 * _SAMPLE_OVERSHOOT))
        query = (
            f"SELECT * FROM {table} TABLESAMPLE SYSTEM ({pct:.6f}) "
            f"REPEATABLE ({int(seed)}) LIMIT {int(max_rows)}"
        )
    else:
        # no estimate: bound the transfer without a scan; biased toward stored order
        query = f"SELECT * FROM {table} LIMIT {int(max_rows)}"

    return ProfilingLoad(_read(query, uri), sampled=True, estimated_rows=estimate)


def _read(query: str, uri: str) -> pl.DataFrame:
    try:
        return pl.read_database_uri(query, uri, engine="connectorx")
    except ImportError as exc:
        raise ImportError(
            "the Postgres connector requires connectorx — install with: uv sync --extra postgres"
        ) from exc
