"""Scoping-time half of the tool: deterministic dataset profiler.

Produces a structured report (column stats, table stats, semantic hints) that is the
primary input for the scoping conversation — it informs which rules to *propose*.
It plays no part at run time: the engine never profiles. No LLM involvement anywhere
in this module; the profiler is pure code that prepares facts for the agent.

Privacy: `redact()` strips raw cell values from a report before it is sent to an LLM.
Numeric and temporal min/max and the distribution sketch are kept in the redacted
variant — they are treated as aggregates. String columns never report min/max at all.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

import polars as pl
from pydantic import BaseModel

TOP_N_VALUES = 5

# hint patterns are deliberately loose: a column full of slightly dirty emails should
# still be hinted as email — the rules catch the dirt, the hint only routes attention
EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
PHONE_PATTERN = r"^\+?[\d\s().-]{7,20}$"
ID_NAME_PATTERN = r"(?i)(^|[_-])id$"
HINT_MATCH_THRESHOLD = 0.8


class TopValue(BaseModel):
    value: str
    count: int


class NumericSummary(BaseModel):
    mean: float | None
    std: float | None
    p25: float | None
    p50: float | None
    p75: float | None


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    # None means not measurable (zero rows / zero non-null values), never "unknown"
    null_rate: float | None
    uniqueness_ratio: float | None  # distinct non-null values / non-null count
    min: float | int | str | None = None  # numeric and temporal columns only
    max: float | int | str | None = None
    numeric: NumericSummary | None = None
    top_values: list[TopValue] | None = None  # None only in redacted reports
    semantic_hint: str | None = None  # id | email | phone | date


class TableProfile(BaseModel):
    row_count: int
    duplicate_row_count: int  # surplus rows: row_count minus distinct rows
    schema_fingerprint: str


class ProfileReport(BaseModel):
    dataset: str
    profiled_at: datetime
    redacted: bool = False
    table: TableProfile
    columns: list[ColumnProfile]


def profile(
    df: pl.DataFrame,
    dataset: str,
    *,
    profiled_at: datetime | None = None,
    hint_sample_rows: int | None = None,
) -> ProfileReport:
    """Profile a DataFrame into a structured report.

    `hint_sample_rows` is a scale hook: when set and the table is taller, semantic-hint
    pattern matching runs on a seeded sample instead of every row. All other statistics
    are always exact.
    """
    row_count = len(df)
    hint_df = df
    if hint_sample_rows is not None and row_count > hint_sample_rows:
        hint_df = df.sample(hint_sample_rows, seed=0)

    return ProfileReport(
        dataset=dataset,
        profiled_at=profiled_at or datetime.now(timezone.utc),
        table=TableProfile(
            row_count=row_count,
            duplicate_row_count=row_count - df.unique().height if row_count else 0,
            schema_fingerprint=_schema_fingerprint(df),
        ),
        columns=[
            _profile_column(df[name], row_count, hint_df[name]) for name in df.columns
        ],
    )


def redact(report: ProfileReport) -> ProfileReport:
    """Return a copy safe for LLM consumption: raw cell value *examples* (top_values)
    are dropped. Bounded aggregates — null rate, uniqueness, and numeric/temporal
    min/max and quantiles — are retained. min/max are real extreme values, disclosed
    deliberately as aggregates (they drive range proposals), not as value listings."""
    clone = report.model_copy(deep=True)
    clone.redacted = True
    for column in clone.columns:
        column.top_values = None
    return clone


def _schema_fingerprint(df: pl.DataFrame) -> str:
    schema = "|".join(f"{name}:{dtype}" for name, dtype in df.schema.items())
    return hashlib.sha256(schema.encode()).hexdigest()[:16]


def _profile_column(s: pl.Series, row_count: int, hint_s: pl.Series) -> ColumnProfile:
    non_null = s.drop_nulls()
    profile = ColumnProfile(
        name=s.name,
        dtype=str(s.dtype),
        null_rate=s.null_count() / row_count if row_count else None,
        uniqueness_ratio=non_null.n_unique() / len(non_null) if len(non_null) else None,
        top_values=[
            TopValue(value=str(value), count=count)
            for value, count in non_null.value_counts(sort=True).head(TOP_N_VALUES).rows()
        ],
        semantic_hint=_semantic_hint(hint_s),
    )

    if len(non_null) == 0:
        return profile

    if s.dtype.is_numeric():
        profile.min = non_null.min()
        profile.max = non_null.max()
        profile.numeric = NumericSummary(
            mean=non_null.mean(),
            std=non_null.std(),
            p25=non_null.quantile(0.25),
            p50=non_null.quantile(0.5),
            p75=non_null.quantile(0.75),
        )
    elif s.dtype.is_temporal():
        profile.min = non_null.min().isoformat()
        profile.max = non_null.max().isoformat()

    return profile


def _semantic_hint(s: pl.Series) -> str | None:
    if s.dtype.is_temporal():
        return "date"
    if re.search(ID_NAME_PATTERN, s.name):
        return "id"

    if s.dtype != pl.String:
        return None
    non_null = s.drop_nulls()
    if len(non_null) == 0:
        return None

    if non_null.str.contains(EMAIL_PATTERN).mean() >= HINT_MATCH_THRESHOLD:
        return "email"
    # date before phone: ISO date strings also match the loose phone pattern,
    # but phone numbers never parse as dates
    if _date_parse_rate(non_null) >= HINT_MATCH_THRESHOLD:
        return "date"
    if non_null.str.contains(PHONE_PATTERN).mean() >= HINT_MATCH_THRESHOLD:
        return "phone"
    return None


def _date_parse_rate(non_null: pl.Series) -> float:
    try:
        parsed = non_null.str.to_date(strict=False)
    except pl.exceptions.PolarsError:
        return 0.0
    return 1.0 - parsed.null_count() / len(non_null)
