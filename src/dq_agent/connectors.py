"""Data connectors — the first step of both workflows.

Every connector loads into a Polars DataFrame, the internal representation for all
profiling (scoping time) and rule execution (run time). Never hand raw cursors or
pandas frames to the profiler or engine."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import NamedTuple
from urllib.parse import quote_plus

import polars as pl

# DSN environment variable read when a loader is called without an explicit `uri`,
# mirroring how build_graph resolves the model from DQ_AGENT_MODEL: a coded env var
# with the value staying injectable for tests.
DEFAULT_DSN_ENV = "DATABASE_DSN__datasets_1"

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")
# a single unqualified column name (no schema dot): used by the bounds-confirm query
_COLUMN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def is_valid_table(name: str) -> bool:
    """Whether `name` is an accepted table locator: an identifier, optionally schema-
    qualified ("schema.table" or "table"). The single source of truth for that rule —
    the loaders enforce it before building SQL, and a UI can call it to validate input
    client-side without duplicating the pattern."""
    return bool(_IDENTIFIER.match(name))
# a custom query must be read-only; reject anything that could mutate, even though the
# agent never supplies raw SQL — this path is reachable programmatically
_FORBIDDEN_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|grant|revoke|create)\b", re.IGNORECASE
)

# TABLESAMPLE SYSTEM is block-level and approximate, so oversample the target fraction
# a little before the exact LIMIT, to make filling the cap likely; floor the fraction so
# a huge table never rounds to 0%.
_SAMPLE_OVERSHOOT = 1.5
_MIN_SAMPLE_PCT = 0.000001


def resolve_dsn(env_var: str = DEFAULT_DSN_ENV) -> str:
    """Resolve a ConnectorX URI from the DSN environment variable.

    This is the single, credential-bearing entry point for env-based connection: a
    driver or agent tool calls it once and passes the result down, so the DSN is read
    in one place and never reaches the LLM (the model only ever names a table). The
    connector loaders also call it internally when invoked without an explicit `uri`,
    so a CLI or notebook can omit the URI and rely on the work-environment default."""
    return _normalise_postgres_uri(_get_connection_string(env_var))


def _get_connection_string(env_var: str) -> str:
    try:
        return os.environ[env_var]
    except KeyError:
        raise KeyError(f"environment variable {env_var!r} is not set") from None


def _normalise_postgres_uri(conn_str: str) -> str:
    """Accept either a ready `postgresql://` URI or a libpq DSN (`host=... dbname=...`)
    and return a ConnectorX-compatible URI. The libpq form is parsed with psycopg,
    imported lazily so this module still loads without the `postgres` extra."""
    if conn_str.startswith("postgresql://") or conn_str.startswith("postgres://"):
        return conn_str
    try:
        from psycopg.conninfo import conninfo_to_dict
    except ImportError as exc:
        raise ImportError(
            "parsing a libpq DSN requires psycopg — install with: uv sync --extra postgres"
        ) from exc
    d = conninfo_to_dict(conn_str)
    return (
        f"postgresql://{quote_plus(d['user'])}:{quote_plus(d['password'])}"
        f"@{d['host']}:{d['port']}/{d['dbname']}"
    )


def load_csv(path: str | Path) -> pl.DataFrame:
    return pl.read_csv(path, try_parse_dates=True)


def load_parquet(path: str | Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def load_postgres(
    uri: str | None = None,
    *,
    table: str | None = None,
    query: str | None = None,
    sample_rows: int | None = None,
    env_var: str = DEFAULT_DSN_ENV,
) -> pl.DataFrame:
    """Load from Postgres via ConnectorX. Provide either a table name (optionally
    schema-qualified) or a full SQL query, not both. When `uri` is omitted it is
    resolved from `env_var` (see `resolve_dsn`).

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
        if not is_valid_table(table):
            raise ValueError(f"invalid table name: {table!r}")
        query = f"SELECT * FROM {table}"
        if sample_rows is not None:
            query += f" ORDER BY random() LIMIT {int(sample_rows)}"
    else:
        if sample_rows is not None:
            raise ValueError("sample_rows applies to 'table', not a custom 'query'")
        # a caller-supplied query is read-only by contract: it must be a SELECT and
        # carry no mutating keyword. The agent never reaches here (it passes table
        # names), but the function is callable directly.
        stripped = query.lstrip().lower()
        if not (stripped.startswith("select") or stripped.startswith("with")):
            raise ValueError("only read-only SELECT queries are allowed")
        if _FORBIDDEN_RE.search(query):
            raise ValueError("query contains a forbidden (mutating) keyword")

    if uri is None:
        uri = resolve_dsn(env_var)
    return _read(query, uri)


def estimate_row_count(uri: str, table: str) -> int | None:
    """Estimate a table's row count from the planner statistics (`pg_class.reltuples`).
    This is instant — it reads a catalog estimate, not the table — unlike `COUNT(*)`,
    which scans. Returns None when the planner has no estimate yet (e.g. a table never
    analyzed reports `reltuples = -1`); callers should then sample defensively rather
    than trust a zero."""
    if not is_valid_table(table):
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
    uri: str | None = None, *, table: str, max_rows: int, seed: int = 0,
    env_var: str = DEFAULT_DSN_ENV,
) -> ProfilingLoad:
    """Load a table sized for profiling. When `uri` is omitted it is resolved from
    `env_var` (see `resolve_dsn`). Estimates the row count from the planner first
    (instant); if it fits under `max_rows` the table is loaded whole, otherwise a
    block-level `TABLESAMPLE` of about `max_rows` rows is pulled — cheap on large tables
    because it reads a fraction of disk blocks instead of scanning. The decision is
    deterministic (the caller sets `max_rows`, not the LLM), and `REPEATABLE (seed)`
    makes the sample reproducible. When the estimate is unknown, it falls back to a plain
    `LIMIT max_rows` — a bounded but storage-order-biased read — and still flags the
    load as sampled."""
    if uri is None:
        uri = resolve_dsn(env_var)
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


def column_bounds(uri: str, table: str, column: str) -> tuple[object, object]:
    """Exact ``(min, max)`` for one column, computed server-side. Confirms the true
    extremes that a sampled profile can only bound: a sampled max is a *lower* bound on
    the real max, so a ``range_check`` parameter taken from it would raise false
    positives against the full table at run time.

    min/max is the cheapest statistic to confirm exactly — with a btree index Postgres
    answers from the index without scanning, and even without one it is a single
    aggregate with no row transfer (unlike ``count(distinct)`` for uniqueness, which
    forces a full scan). So this is the one to call before locking a range bound on a
    table large enough to have been sampled. Returns ``(None, None)`` for an empty or
    all-null column."""
    if not is_valid_table(table):
        raise ValueError(f"invalid table name: {table!r}")
    if not _COLUMN.match(column):
        raise ValueError(f"invalid column name: {column!r}")
    query = f"SELECT min({column}) AS lo, max({column}) AS hi FROM {table}"
    result = _read(query, uri)
    if result.is_empty():
        return None, None
    return result["lo"][0], result["hi"][0]


def _read(query: str, uri: str) -> pl.DataFrame:
    try:
        return pl.read_database_uri(query, uri, engine="connectorx")
    except ImportError as exc:
        raise ImportError(
            "the Postgres connector requires connectorx — install with: uv sync --extra postgres"
        ) from exc
