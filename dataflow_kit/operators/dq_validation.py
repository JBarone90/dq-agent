"""Run an approved data quality contract with Great Expectations 0.18.2.

Three pieces make up a data quality check, and only the third changes per table:

- this module, ``operators/dq_validation.py``: loads a contract YAML, validates it, and runs it as an in-memory Expectation Suite and Checkpoint;
- the validation DAG, ``dq_check_<table>_pipeline.py``: calls this module from a ``@task.virtualenv`` task;
- the contract, ``contracts/<schema>.<table>.yaml``: produced by the dq-agent scoping agent and approved by the dataset owner.

The contract uses GX expectation names and kwargs directly. There is no translation layer. Connection strings never live in the contract: the caller passes one in from the environment.

Great Expectations is imported inside the functions that need it, so parsing the DAG file never imports GX.
"""

from __future__ import annotations

import datetime
import logging
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

CONTRACT_VERSION = 1
GX_VERSION = "0.18.2"
CONTRACTS_DIR = Path(__file__).parent.parent / "contracts"

# Expectations a contract may use -> kwargs required in addition to GX's own
# required configuration.
ALLOWED_EXPECTATIONS: dict[str, frozenset[str]] = {
    "expect_column_values_to_not_be_null": frozenset(),
    "expect_column_values_to_be_unique": frozenset(),
    "expect_column_values_to_be_in_set": frozenset(),
    "expect_column_values_to_be_between": frozenset(),
    # GX silently defaults regex to match everything, which would make the
    # check vacuous.
    "expect_column_values_to_match_regex": frozenset({"regex"}),
    "expect_table_row_count_to_be_between": frozenset(),
    "expect_column_values_to_be_in_type_list": frozenset(),
    "expect_column_min_to_be_between": frozenset(),
    "expect_column_max_to_be_between": frozenset(),
    "expect_column_values_to_not_be_in_set": frozenset(),
    "expect_column_values_to_not_match_regex": frozenset(),
    "expect_column_value_lengths_to_be_between": frozenset(),
    "expect_column_pair_values_a_to_be_greater_than_b": frozenset(),
    "expect_compound_columns_to_be_unique": frozenset(),
    "expect_select_column_values_to_be_unique_within_record": frozenset(),
    "expect_table_columns_to_match_set": frozenset(),
}

# Valid ignore_row_if values by domain. GX accepts any string here and only
# fails at run time.
_IGNORE_ROW_IF = {
    "column_A": frozenset({"both_values_are_missing", "either_value_is_missing", "neither"}),
    "column_list": frozenset({"all_values_are_missing", "any_value_is_missing", "never"}),
}

# Accepted by GX, but these change what data is checked, how results are
# reported, or allow GX's profiler to choose parameters. None belongs in a
# reviewed contract. Each reason completes "not allowed in contracts because ...".
_FORBIDDEN_REASONS: dict[str, str] = {
    "batch_id": "it changes which data is checked",
    "table": "it changes which data is checked",
    "row_condition": "it changes which rows are checked",
    "condition_parser": "it changes which rows are checked",
    "result_format": "it changes how results are reported",
    "include_config": "it changes how results are reported",
    "catch_exceptions": "it changes how results are reported",
    "auto": "it lets GX's profiler choose the parameters",
    "profiler_config": "it lets GX's profiler choose the parameters",
    "parse_strings_as_datetimes": "it changes how values are compared",
    "allow_cross_type_comparisons": "it changes how values are compared",
    "meta": "notes and severity go in the expectation's meta, next to kwargs",
}
FORBIDDEN_KWARGS = frozenset(_FORBIDDEN_REASONS)

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# The only evaluation parameter a contract may carry: a relative date bound
# for freshness. Anything else under "$PARAMETER" is rejected, so GX never
# evaluates a free-form expression.
_DATE_PARAMETER = re.compile(r"^now\(\) - timedelta\(days=[1-9][0-9]*\)$")


class ContractError(ValueError):
    """The contract YAML is malformed or uses something outside the allowlist."""


class DataQualityCheckFailed(Exception):
    """The contract ran and at least one error-severity expectation failed."""


def allowed_kwargs(expectation_type: str) -> list[str]:
    """Return kwargs that a contract may pass to an allowed expectation."""
    from great_expectations.expectations.registry import get_expectation_impl

    impl = get_expectation_impl(expectation_type)
    keys = dict.fromkeys([*impl.domain_keys, *impl.success_keys])
    return [key for key in keys if key not in FORBIDDEN_KWARGS]


def contract_path(
    schema: str,
    table: str,
    root: Path = CONTRACTS_DIR,
) -> Path:
    """Return the contract path: ``<schema>.<table>.yaml``."""
    for name in (schema, table):
        if not _IDENTIFIER.match(name):
            raise ContractError(f"invalid identifier '{name}'")
    return root / f"{schema}.{table}.yaml"


def load_table_contract(
    schema: str,
    table: str,
    root: Path = CONTRACTS_DIR,
) -> dict[str, Any]:
    """Load a contract and confirm it describes the requested table."""
    path = contract_path(schema, table, root)
    contract = load_contract(path)
    if (contract["dataset"]["schema"], contract["dataset"]["table"]) != (
        schema,
        table,
    ):
        raise ContractError(
            f"{path.name} describes {contract['suite']['name']}, not {schema}.{table}"
        )
    return contract


def load_contract(path: str | Path) -> dict[str, Any]:
    """Read and validate a contract, raising ContractError if it is invalid."""
    try:
        contract = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"{path}: could not be read ({exc})") from exc
    except yaml.YAMLError as exc:
        raise ContractError(f"{path}: not valid YAML ({exc})") from exc

    validate_contract(contract, name=str(path))
    return contract


def validate_contract(contract: Any, name: str = "contract") -> None:
    """Validate a contract and raise one error containing every problem."""
    errors = contract_errors(contract)
    if errors:
        count = f"{len(errors)} problem{'s' if len(errors) > 1 else ''}"
        raise ContractError(f"{name} is invalid ({count}):\n- " + "\n- ".join(errors))


def contract_errors(contract: Any) -> list[str]:
    """Return all structural, policy, and GX configuration errors."""
    if not isinstance(contract, dict):
        return ["contract must be a mapping"]

    errors = _unknown_keys(
        contract,
        {"contract_version", "gx_version", "dataset", "approval", "suite"},
        "contract",
    )

    if contract.get("contract_version") != CONTRACT_VERSION:
        errors.append(f"contract_version must be {CONTRACT_VERSION}")
    if contract.get("gx_version") != GX_VERSION:
        errors.append(f'gx_version must be "{GX_VERSION}"')

    dataset = contract.get("dataset")
    if not isinstance(dataset, dict):
        errors.append("dataset must be a mapping with schema and table")
        dataset = {}
    errors += _unknown_keys(dataset, {"schema", "table"}, "dataset")
    for key in ("schema", "table"):
        value = dataset.get(key)
        if not isinstance(value, str) or not _IDENTIFIER.match(value):
            errors.append(f"dataset.{key} must be a plain SQL identifier")

    approval = contract.get("approval")
    if not isinstance(approval, dict):
        errors.append("approval is required: no contract runs without an approved_at and approved_by")
    else:
        errors += _unknown_keys(
            approval,
            {"approved_at", "approved_by"},
            "approval",
        )
        errors += _approval_timestamp_errors(approval.get("approved_at"))
        if not isinstance(approval.get("approved_by"), str) or not approval["approved_by"].strip():
            errors.append("approval.approved_by must be a non-empty string")

    suite = contract.get("suite")
    if not isinstance(suite, dict):
        return errors + ["suite must be a mapping with name and expectations"]

    errors += _unknown_keys(suite, {"name", "expectations"}, "suite")
    expected_name = f"{dataset.get('schema')}.{dataset.get('table')}"
    if suite.get("name") != expected_name:
        errors.append(f"suite.name must be '{expected_name}'")

    expectations = suite.get("expectations")
    if not isinstance(expectations, list) or not expectations:
        return errors + ["suite.expectations must be a non-empty list"]

    for index, expectation in enumerate(expectations):
        errors += expectation_errors(index, expectation)
    return errors


def expectation_errors(index: int, expectation: Any) -> list[str]:
    """Return validation errors for one expectation entry."""
    # Numbered from 1, as a person counts the entries in the file.
    number = index + 1
    if not isinstance(expectation, dict):
        return [f"expectation {number}: must be a mapping"]

    kind = expectation.get("expectation_type")
    kwargs = expectation.get("kwargs")
    columns = _columns_label(kwargs) if isinstance(kwargs, dict) else None
    where = f"expectation {number} ({kind}{f' on {columns}' if columns else ''})"
    errors = _unknown_keys(
        expectation,
        {"expectation_type", "kwargs", "meta"},
        where,
    )

    if kind not in ALLOWED_EXPECTATIONS:
        return errors + [
            f"expectation {number}: '{kind}' is not an allowed expectation_type "
            f"(allowed: {', '.join(sorted(ALLOWED_EXPECTATIONS))})"
        ]

    if not isinstance(kwargs, dict):
        return errors + [f"{where}: kwargs must be a mapping"]

    errors += [f"{where}: {problem}" for problem in kwarg_errors(kind, kwargs)]
    if "meta" in expectation:
        errors += _meta_errors(where, expectation["meta"])
    return errors


def kwarg_errors(kind: str, kwargs: dict[str, Any]) -> list[str]:
    """Return problems with an expectation's kwargs."""
    allowed = allowed_kwargs(kind)
    errors = [
        f"unexpected kwarg '{name}': not allowed in contracts because {_FORBIDDEN_REASONS[name]}"
        if name in _FORBIDDEN_REASONS
        else f"unexpected kwarg '{name}' (allowed: {', '.join(allowed)})"
        for name in sorted(kwargs.keys() - set(allowed), key=str)
    ]
    errors += [
        f"missing required kwarg '{name}' (GX's default would let the check pass on any data)"
        for name in sorted(ALLOWED_EXPECTATIONS.get(kind, frozenset()) - kwargs.keys())
    ]
    # Policy only judges values of kwargs a contract may set: the ones rejected
    # above already have their error.
    errors += _policy_errors(kind, {name: value for name, value in kwargs.items() if name in allowed})

    if not errors:
        errors += _gx_errors(kind, kwargs)
    return errors


def _policy_errors(kind: str, kwargs: dict[str, Any]) -> list[str]:
    """Return policy violations that GX 0.18.2 otherwise accepts."""
    errors: list[str] = []

    for name in ("column", "column_A", "column_B"):
        if name in kwargs and not _is_name(kwargs[name]):
            errors.append(f"{name} must be a non-empty string")

    # One column would check nothing (within a record) or duplicate
    # expect_column_values_to_be_unique (compound).
    if "column_list" in kwargs:
        columns = kwargs["column_list"]
        if not (isinstance(columns, list) and len(columns) >= 2 and all(map(_is_name, columns))):
            errors.append("column_list must list at least two column names")
        elif len(set(columns)) != len(columns):
            errors.append("column_list must not repeat a column")

    if "column_set" in kwargs and not (
        isinstance(kwargs["column_set"], list) and kwargs["column_set"] and all(map(_is_name, kwargs["column_set"]))
    ):
        errors.append("column_set must be a non-empty list of column names")

    if "value_set" in kwargs and not kwargs["value_set"]:
        errors.append("value_set must not be empty")

    for name in ("strict_min", "strict_max", "or_equal", "exact_match"):
        if name in kwargs and not isinstance(kwargs[name], bool):
            errors.append(f"{name} must be true or false")

    if "ignore_row_if" in kwargs:
        valid = next((values for domain, values in _IGNORE_ROW_IF.items() if domain in kwargs), frozenset())
        if kwargs["ignore_row_if"] not in valid:
            errors.append(f"ignore_row_if must be one of: {', '.join(sorted(valid))}")

    if kind.endswith("_to_be_between") and kwargs.get("min_value") is None and kwargs.get("max_value") is None:
        errors.append("needs at least one of min_value, max_value")

    for name, value in kwargs.items():
        if isinstance(value, dict) and not (
            set(value) == {"$PARAMETER"}
            and isinstance(value["$PARAMETER"], str)
            and _DATE_PARAMETER.match(value["$PARAMETER"])
        ):
            errors.append(f'{name}: the only allowed parameter is {{"$PARAMETER": "now() - timedelta(days=N)"}}')

    return errors


def _is_name(value: Any) -> bool:
    """Return whether a value can be a column name."""
    return isinstance(value, str) and bool(value)


def _columns_label(kwargs: dict[str, Any]) -> str | None:
    """Return the column(s) an expectation checks, for messages; None if table-level."""
    if _is_name(kwargs.get("column")):
        return kwargs["column"]
    if _is_name(kwargs.get("column_A")) and _is_name(kwargs.get("column_B")):
        return f"{kwargs['column_A']}, {kwargs['column_B']}"
    if isinstance(kwargs.get("column_list"), list) and all(map(_is_name, kwargs["column_list"])):
        return ", ".join(kwargs["column_list"])
    return None


def _gx_errors(kind: str, kwargs: dict[str, Any]) -> list[str]:
    """Return errors found by GX's own configuration validation."""
    from great_expectations.core.expectation_configuration import (
        ExpectationConfiguration,
    )
    from great_expectations.exceptions import (
        InvalidExpectationConfigurationError,
    )
    from great_expectations.expectations.registry import get_expectation_impl

    try:
        configuration = ExpectationConfiguration(
            expectation_type=kind,
            kwargs=kwargs,
        )
        get_expectation_impl(kind)(configuration).validate_configuration(configuration)
    except (InvalidExpectationConfigurationError, TypeError, ValueError) as exc:
        return [str(exc)]
    return []


def _approval_timestamp_errors(value: Any) -> list[str]:
    """Require approval.approved_at to be a valid ISO-8601 timestamp."""
    if isinstance(value, datetime.datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return ["approval.approved_at must be a valid ISO-8601 timestamp"]
    else:
        # YAML may parse a date-only value as datetime.date. Approval should be
        # a timestamp rather than a date without a time, so reject plain dates.
        return ["approval.approved_at must be a valid ISO-8601 timestamp"]

    # A YAML timestamp without an offset parses as a naive datetime: hold it to
    # the same rule as the string form.
    if parsed.tzinfo is None:
        return ["approval.approved_at must include a timezone, such as Z or +00:00"]
    return []


def _meta_errors(where: str, meta: Any) -> list[str]:
    """Allow notes and pipeline-level severity in expectation meta.

    An ``error`` failure blocks the Airflow task. A ``warning`` failure is
    persisted and logged, but does not by itself fail the task. If severity is
    omitted, it defaults to ``error`` so existing contracts retain their
    current behaviour.
    """
    if not isinstance(meta, dict):
        return [f"{where}: meta must be a mapping"]

    errors: list[str] = []

    unknown_keys = set(meta) - {"notes", "severity"}
    errors += [f"{where}: meta contains unknown key '{key}'" for key in sorted(unknown_keys, key=str)]

    severity = meta.get("severity", "error")
    if severity not in {"error", "warning"}:
        errors.append(f"{where}: meta.severity must be 'error' or 'warning'")

    if "notes" in meta:
        notes = meta["notes"]
        if (
            not isinstance(notes, dict)
            or set(notes) != {"format", "content"}
            or notes.get("format") not in {"string", "markdown"}
            or not isinstance(notes.get("content"), str)
        ):
            errors.append(f"{where}: meta.notes must be {{format: string|markdown, content: <text>}}")

    return errors


def _unknown_keys(
    mapping: dict[Any, Any],
    allowed: set[str],
    where: str,
) -> list[str]:
    """Return unknown-key errors for a mapping."""
    return [f"{where}: unknown key '{key}'" for key in sorted(mapping.keys() - allowed, key=str)]


def failed_expectations(
    checkpoint_result: Any,
    severity: str,
) -> list[Any]:
    """Return failed expectation results matching a severity level.

    A check that could not run (a missing column, a regex Postgres rejects) is
    always an error: as a warning it would let the run complete with the rule
    not enforced.
    """
    failures: list[Any] = []

    for validation in checkpoint_result.list_validation_results():
        for result in validation.results:
            if result.success:
                continue

            raised = bool((result.exception_info or {}).get("raised_exception"))
            meta = result.expectation_config.meta or {}
            result_severity = "error" if raised else meta.get("severity", "error")

            if result_severity == severity:
                failures.append(result)

    return failures


def run_contract(
    context: Any,
    contract: dict[str, Any],
    *,
    connection_string: str,
    checkpoint_name: str,
    run_name: str,
    raise_on_failure: bool = True,
) -> Any:
    """Run a validated contract against its PostgreSQL table.

    ``context`` comes from ``dataflow.validation_utils.gx_get_context()``. Its
    Validation Store controls where the result is persisted. The suite and
    checkpoint are built in memory for this run only; the YAML contract
    remains the durable definition of the rules.
    """
    validate_contract(contract)

    schema = contract["dataset"]["schema"]
    table = contract["dataset"]["table"]
    suite_name = contract["suite"]["name"]

    source = context.sources.add_postgres(
        name="datasets",
        connection_string=connection_string,
    )
    asset = source.add_table_asset(
        name=suite_name,
        schema_name=schema,
        table_name=table,
    )
    suite = context.add_expectation_suite(
        expectation_suite_name=suite_name,
        expectations=expectation_configurations(contract),
    )
    checkpoint = context.add_checkpoint(
        name=checkpoint_name,
        run_name_template=run_name,
        validator=context.get_validator(
            batch_request=asset.build_batch_request(),
            expectation_suite_name=suite.name,
        ),
    )

    result = checkpoint.run()
    summary = summarize(result)
    logger.info("Data quality result for %s:\n%s", suite_name, summary)

    blocking_failures = failed_expectations(result, severity="error")
    warning_failures = failed_expectations(result, severity="warning")

    if warning_failures:
        logger.warning(
            "%s warning-level data quality check(s) failed for %s. "
            "The validation results were persisted, but these failures "
            "will not fail the Airflow task.",
            len(warning_failures),
            suite_name,
        )

    if raise_on_failure and blocking_failures:
        raise DataQualityCheckFailed(
            f"{len(blocking_failures)} blocking data quality check(s) failed for {suite_name}:\n{summary}"
        )

    return result


def expectation_configurations(contract: dict[str, Any]) -> list[Any]:
    """Build GX configurations and attach contract approval provenance."""
    from great_expectations.core.expectation_configuration import (
        ExpectationConfiguration,
    )

    provenance = {
        "contract_version": contract["contract_version"],
        "approved_by": contract["approval"]["approved_by"],
        "approved_at": str(contract["approval"]["approved_at"]),
    }

    return [
        ExpectationConfiguration(
            expectation_type=expectation["expectation_type"],
            kwargs=expectation["kwargs"],
            meta={
                **expectation.get("meta", {}),
                "dq_contract": provenance,
            },
        )
        for expectation in contract["suite"]["expectations"]
    ]


def summarize(checkpoint_result: Any) -> str:
    """Return one line per expectation without exposing unexpected row values."""
    lines: list[str] = []

    for validation in checkpoint_result.list_validation_results():
        for result in validation.results:
            config = result.expectation_config
            column = _columns_label(config.kwargs) or "<table>"
            unexpected = (result.result or {}).get("unexpected_count")

            meta = config.meta or {}
            raised = bool((result.exception_info or {}).get("raised_exception"))
            severity = "ERROR" if raised else meta.get("severity", "error").upper()

            unexpected_text = f" ({unexpected} unexpected)" if unexpected is not None else ""

            lines.append(
                f"{'PASS' if result.success else 'FAIL'} "
                f"[{severity}] "
                f"{config.expectation_type} on {column}"
                f"{unexpected_text}"
                f"{' (could not run)' if raised else ''}"
            )

    return "\n".join(lines)
