"""What a dq-agent contract may contain, and how to validate and run one with Great Expectations 0.18.2.

The contract uses GX expectation names and kwargs directly, with no translation layer. These rules are the ones data-flow applies when it loads a contract (`dataflow_kit/operators/dq_validation.py`, copied into data-flow), so anything the agent saves passes there. The two modules cannot share code: the kit may not import dq_agent, and dq_agent does not import the kit. `tests/test_gx_contract.py` holds them to the same allowlist and the same verdict on every contract it tries, so change both together. What a contract may contain is decided here, mostly by GX itself:

- `ALLOWED_EXPECTATIONS` lists which expectations may be used at all;
- the kwargs each one accepts come from GX (its domain and success keys), minus `FORBIDDEN_KWARGS`, which would change how or on what data a check runs;
- GX's own `validate_configuration` checks the kwarg values;
- a few generic rules cover what GX lets through (see `_policy_errors`).

So a contract can never smuggle in arbitrary classes, actions or code. Connection strings never live in the contract: the caller passes one in from the environment.

`run_contract` and `run_contract_on_asset` exist to try a contract (see `notebooks/engine_gx_run.ipynb`); the agent itself never runs one. Great Expectations is imported inside the functions that need it. This module must not import the rest of dq_agent: the registry, models and agent import it.
"""

from __future__ import annotations

import datetime
import logging
import re
from pathlib import Path
from typing import Any, Final

import yaml

logger = logging.getLogger(__name__)

CONTRACT_VERSION = 1
GX_VERSION = "0.18.2"

# What a failed expectation does to the data-flow task. An `error` failure fails the task, so the
# run cannot complete. A `warning` failure is persisted and logged and the run still completes,
# unless the check could not run at all (see `effective_severity`).
# An expectation with no severity is an `error`, so a contract written before severity existed
# keeps its behaviour.
SEVERITIES = ("error", "warning")
DEFAULT_SEVERITY: Final = "error"

# Expectations a contract may use -> kwargs we require on top of what GX requires.
ALLOWED_EXPECTATIONS: dict[str, frozenset[str]] = {
    "expect_column_values_to_not_be_null": frozenset(),
    "expect_column_values_to_be_unique": frozenset(),
    "expect_column_values_to_be_in_set": frozenset(),
    "expect_column_values_to_be_between": frozenset(),
    # GX silently defaults regex to match-everything, which would make the check vacuous
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

# Valid ignore_row_if values by domain. GX accepts any string here and only fails at run time.
_IGNORE_ROW_IF = {
    "column_A": frozenset({"both_values_are_missing", "either_value_is_missing", "neither"}),
    "column_list": frozenset({"all_values_are_missing", "any_value_is_missing", "never"}),
}

# Accepted by GX, but they change what data is checked, how results are reported, or let
# GX's profiler pick the parameters — none of that belongs in a reviewed contract.
# Each reason completes "not allowed in contracts because ...".
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
# The only evaluation parameter a contract may carry: a relative date bound for freshness.
# Anything else under "$PARAMETER" is rejected, so GX never evaluates a free-form expression.
_DATE_PARAMETER = re.compile(r"^now\(\) - timedelta\(days=[1-9][0-9]*\)$")


class ContractError(ValueError):
    """The contract YAML is malformed or uses something outside the allowlist."""


class DataQualityCheckFailed(Exception):
    """The contract ran and at least one error-severity expectation failed."""


def allowed_kwargs(expectation_type: str) -> list[str]:
    """The kwargs a contract may pass to an allowed expectation, straight from GX."""
    from great_expectations.expectations.registry import get_expectation_impl

    impl = get_expectation_impl(expectation_type)
    keys = dict.fromkeys([*impl.domain_keys, *impl.success_keys])  # ordered, de-duplicated
    return [k for k in keys if k not in FORBIDDEN_KWARGS]


def contract_path(schema: str, table: str, root: Path) -> Path:
    """Where the contract for a table lives: one file per table, named `<schema>.<table>.yaml`."""
    for name in (schema, table):
        if not _IDENTIFIER.match(name):
            raise ContractError(f"invalid identifier '{name}'")
    return root / f"{schema}.{table}.yaml"


def load_table_contract(schema: str, table: str, root: Path) -> dict[str, Any]:
    """Load the contract for a table and confirm it describes that table."""
    path = contract_path(schema, table, root)
    contract = load_contract(path)
    if (contract["dataset"]["schema"], contract["dataset"]["table"]) != (schema, table):
        raise ContractError(f"{path.name} describes {contract['suite']['name']}, not {schema}.{table}")
    return contract


def load_contract(path: str | Path) -> dict[str, Any]:
    """Read and validate a contract. Raises ContractError naming what is wrong."""
    try:
        contract = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"{path}: could not be read ({exc})") from exc
    except yaml.YAMLError as exc:
        raise ContractError(f"{path}: not valid YAML ({exc})") from exc
    validate_contract(contract, name=str(path))
    return contract


def validate_contract(contract: Any, name: str = "contract") -> None:
    """Raise one ContractError listing every problem, if there are any."""
    errors = contract_errors(contract)
    if errors:
        count = f"{len(errors)} problem{'s' if len(errors) > 1 else ''}"
        raise ContractError(f"{name} is invalid ({count}):\n- " + "\n- ".join(errors))


def contract_errors(contract: Any) -> list[str]:
    """Every structural, policy and GX configuration problem with a contract."""
    if not isinstance(contract, dict):
        return ["contract must be a mapping"]
    errors = _unknown_keys(contract, {"contract_version", "gx_version", "dataset", "approval", "suite"}, "contract")

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
        if not isinstance(dataset.get(key), str) or not _IDENTIFIER.match(dataset[key]):
            errors.append(f"dataset.{key} must be a plain SQL identifier")

    approval = contract.get("approval")
    if not isinstance(approval, dict):
        errors.append("approval is required: no contract runs without an approved_at and approved_by")
    else:
        errors += _unknown_keys(approval, {"approved_at", "approved_by"}, "approval")
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
    """Problems with one expectation entry, numbered from 1 as a person counts them in the file."""
    number = index + 1
    if not isinstance(expectation, dict):
        return [f"expectation {number}: must be a mapping"]
    kind = expectation.get("expectation_type")
    kwargs = expectation.get("kwargs")
    columns = _columns_label(kwargs) if isinstance(kwargs, dict) else None
    where = f"expectation {number} ({kind}{f' on {columns}' if columns else ''})"
    errors = _unknown_keys(expectation, {"expectation_type", "kwargs", "meta"}, where)

    if kind not in ALLOWED_EXPECTATIONS:
        return errors + [f"expectation {number}: '{kind}' is not an allowed expectation_type "
                         f"(allowed: {', '.join(sorted(ALLOWED_EXPECTATIONS))})"]
    if not isinstance(kwargs, dict):
        return errors + [f"{where}: kwargs must be a mapping"]

    errors += [f"{where}: {problem}" for problem in kwarg_errors(kind, kwargs)]
    if "meta" in expectation:
        errors += _meta_errors(where, expectation["meta"])
    return errors


def kwarg_errors(kind: str, kwargs: dict[str, Any]) -> list[str]:
    """Problems with an expectation's kwargs: unknown or forbidden names, our policy rules,
    then GX's own checks. Works for any GX expectation, allowed or not."""
    allowed = allowed_kwargs(kind)
    errors = [
        f"unexpected kwarg '{name}': not allowed in contracts because {_FORBIDDEN_REASONS[name]}"
        if name in _FORBIDDEN_REASONS
        else f"unexpected kwarg '{name}' (allowed: {', '.join(allowed)})"
        for name in sorted(kwargs.keys() - set(allowed), key=str)
    ]
    errors += [f"missing required kwarg '{name}' (GX's default would let the check pass on any data)"
               for name in sorted(ALLOWED_EXPECTATIONS.get(kind, frozenset()) - kwargs.keys())]
    # policy only judges kwargs a contract may set: the rejected ones already have their error
    errors += _policy_errors(kind, {name: value for name, value in kwargs.items() if name in allowed})
    if not errors:
        # GX's own checks: required kwargs, value types, value ranges
        errors += _gx_errors(kind, kwargs)
    return errors


def _policy_errors(kind: str, kwargs: dict[str, Any]) -> list[str]:
    """What GX 0.18.2 accepts but a reviewed contract should not."""
    errors = []
    for name in ("column", "column_A", "column_B"):
        if name in kwargs and not _is_name(kwargs[name]):
            errors.append(f"{name} must be a non-empty string")
    # one column would check nothing (within a record) or duplicate expect_column_values_to_be_unique
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
    if kind.endswith("_to_be_between") \
            and kwargs.get("min_value") is None and kwargs.get("max_value") is None:
        errors.append("needs at least one of min_value, max_value")  # else it checks nothing
    for name, value in kwargs.items():
        if isinstance(value, dict) and not (
            set(value) == {"$PARAMETER"}
            and isinstance(value["$PARAMETER"], str)
            and _DATE_PARAMETER.match(value["$PARAMETER"])
        ):
            errors.append(f'{name}: the only allowed parameter is {{"$PARAMETER": "now() - timedelta(days=N)"}}')
    return errors


def _is_name(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _columns_label(kwargs: dict[str, Any]) -> str | None:
    """The column(s) an expectation checks, for messages; None if it is table-level."""
    if _is_name(kwargs.get("column")):
        return kwargs["column"]
    if _is_name(kwargs.get("column_A")) and _is_name(kwargs.get("column_B")):
        return f"{kwargs['column_A']}, {kwargs['column_B']}"
    if isinstance(kwargs.get("column_list"), list) and all(map(_is_name, kwargs["column_list"])):
        return ", ".join(kwargs["column_list"])
    return None


def _gx_errors(kind: str, kwargs: dict[str, Any]) -> list[str]:
    from great_expectations.core.expectation_configuration import (
        ExpectationConfiguration,
    )
    from great_expectations.exceptions import InvalidExpectationConfigurationError
    from great_expectations.expectations.registry import get_expectation_impl

    try:
        configuration = ExpectationConfiguration(expectation_type=kind, kwargs=kwargs)
        get_expectation_impl(kind)(configuration).validate_configuration(configuration)
    except (InvalidExpectationConfigurationError, TypeError, ValueError) as exc:
        return [str(exc)]
    return []


def _approval_timestamp_errors(value: Any) -> list[str]:
    """approval.approved_at must be a full ISO-8601 timestamp with a timezone, not a bare date."""
    if isinstance(value, datetime.datetime):  # YAML parses an unquoted timestamp to a datetime
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return ["approval.approved_at must be a valid ISO-8601 timestamp"]
    else:  # including a date-only value, which YAML parses to a date: approval needs a time
        return ["approval.approved_at must be a valid ISO-8601 timestamp"]
    if parsed.tzinfo is None:
        return ["approval.approved_at must include a timezone, such as Z or +00:00"]
    return []


def _meta_errors(where: str, meta: Any) -> list[str]:
    """Expectation meta may carry the dashboard `notes` and the pipeline-level `severity`."""
    if not isinstance(meta, dict):
        return [f"{where}: meta must be a mapping"]
    errors = [f"{where}: meta contains unknown key '{key}'" for key in sorted(set(meta) - {"notes", "severity"}, key=str)]
    if meta.get("severity", DEFAULT_SEVERITY) not in SEVERITIES:
        errors.append(f"{where}: meta.severity must be 'error' or 'warning'")
    if "notes" in meta:
        notes = meta["notes"]
        if not isinstance(notes, dict) or set(notes) != {"format", "content"} \
                or notes.get("format") not in ("string", "markdown") \
                or not isinstance(notes.get("content"), str):
            errors.append(f"{where}: meta.notes must be {{format: string|markdown, content: <text>}}")
    return errors


def _unknown_keys(mapping: dict, allowed: set[str], where: str) -> list[str]:
    return [f"{where}: unknown key '{key}'" for key in sorted(mapping.keys() - allowed, key=str)]


def could_not_run(result: Any) -> bool:
    """True when the check itself broke (a missing column, a regex Postgres rejects), as opposed
    to the data breaking the rule. GX reports both as `success=False`."""
    return bool((result.exception_info or {}).get("raised_exception"))


def effective_severity(result: Any) -> str:
    """What a failed result counts as. A check that could not run is always an error: as a warning
    it would let the run complete with the rule silently not enforced."""
    if could_not_run(result):
        return "error"
    return (result.expectation_config.meta or {}).get("severity", DEFAULT_SEVERITY)


def failed_expectations(checkpoint_result: Any, severity: str) -> list[Any]:
    """The failed expectation results of one severity. An expectation without one is an error."""
    return [
        r
        for validation in checkpoint_result.list_validation_results()
        for r in validation.results
        if not r.success and effective_severity(r) == severity
    ]


def run_contract(
    context: Any,
    contract: dict[str, Any],
    *,
    connection_string: str,
    checkpoint_name: str,
    run_name: str,
    raise_on_failure: bool = True,
) -> Any:
    """Run a validated contract against its Postgres table and return the CheckpointResult.

    Like the data-flow task, it raises DataQualityCheckFailed when an error-severity expectation
    failed; warning-severity failures are only logged. Pass `raise_on_failure=False` to get the
    result either way and decide from `failed_expectations`.

    The suite and checkpoint are built in memory for this run only: the contract YAML stays
    the durable definition of the rules. Each expectation's meta carries the approval, so a
    stored result records who approved the rules it ran.
    """
    validate_contract(contract)  # never trust that the caller validated
    schema, table = contract["dataset"]["schema"], contract["dataset"]["table"]

    # add_or_update, so a second run on the same context replaces the source instead of colliding
    source = context.sources.add_or_update_postgres(name="datasets", connection_string=connection_string)
    asset = source.add_table_asset(name=contract["suite"]["name"], schema_name=schema, table_name=table)
    return run_contract_on_asset(
        context, contract, asset,
        checkpoint_name=checkpoint_name, run_name=run_name, raise_on_failure=raise_on_failure,
    )


def run_contract_on_asset(
    context: Any,
    contract: dict[str, Any],
    asset: Any,
    *,
    checkpoint_name: str,
    run_name: str,
    raise_on_failure: bool = True,
    batch_request: Any = None,
) -> Any:
    """Run a validated contract against any GX data asset; `run_contract` is this on its
    Postgres table. Same verdict rules; `batch_request` defaults to the asset's own."""
    validate_contract(contract)
    suite_name = contract["suite"]["name"]
    suite = context.add_or_update_expectation_suite(
        expectation_suite_name=suite_name,
        expectations=expectation_configurations(contract),
    )
    checkpoint = context.add_or_update_checkpoint(
        name=checkpoint_name,
        run_name_template=run_name,
        validator=context.get_validator(
            batch_request=batch_request or asset.build_batch_request(),
            expectation_suite_name=suite.name,
        ),
    )
    result = checkpoint.run()
    summary = summarize(result)
    logger.info("Data quality result for %s:\n%s", suite_name, summary)

    blocking = failed_expectations(result, "error")
    warnings = failed_expectations(result, "warning")
    if warnings:
        logger.warning("%s warning-level data quality check(s) failed for %s; they do not fail the task",
                       len(warnings), suite_name)
    if raise_on_failure and blocking:
        raise DataQualityCheckFailed(
            f"{len(blocking)} blocking data quality check(s) failed for {suite_name}:\n{summary}"
        )
    return result


def expectation_configurations(contract: dict[str, Any]) -> list[Any]:
    """The contract's expectations as GX ExpectationConfigurations, each stamped with the approval.

    Suite-level meta stays in the in-memory suite store; each expectation's config is copied
    into the stored validation result, so the approval rides there to be persisted.
    """
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
            expectation_type=e["expectation_type"],
            kwargs=e["kwargs"],
            meta={**e.get("meta", {}), "dq_contract": provenance},
        )
        for e in contract["suite"]["expectations"]
    ]


def summarize(checkpoint_result: Any) -> str:
    """One line per expectation: pass/fail, severity, type, column(s) and unexpected count. No row values."""
    lines = []
    for validation in checkpoint_result.list_validation_results():
        for r in validation.results:
            config = r.expectation_config
            column = _columns_label(config.kwargs) or "<table>"
            unexpected = (r.result or {}).get("unexpected_count")
            lines.append(
                f"{'PASS' if r.success else 'FAIL'} [{effective_severity(r).upper()}] "
                f"{config.expectation_type} on {column}"
                + (f" ({unexpected} unexpected)" if unexpected is not None else "")
                + (" (could not run)" if could_not_run(r) else "")  # the reason stays out: it can quote data
            )
    return "\n".join(lines)
