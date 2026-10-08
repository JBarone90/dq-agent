"""Run an approved data quality contract with Great Expectations 0.18.2.

Staged in dq-agent, copied into the `data-flow` repo at work. Three pieces make up a data quality check there, and only the third changes per table:

- this module: loads a contract YAML, validates it, and runs it as an in-memory Expectation Suite + Checkpoint;
- the validation DAG (`dags/dq_check_example.py`): calls this module from a `@task.virtualenv` task;
- the contract: `dq_contracts/<schema>.<table>.yaml`, produced by the dq-agent scoping agent and approved by the dataset owner.

The contract uses GX expectation names and kwargs directly — there is no translation layer. What a contract may contain is decided here, mostly by GX itself:

- `ALLOWED_EXPECTATIONS` lists which expectations may be used at all;
- the kwargs each one accepts come from GX (its domain and success keys), minus `FORBIDDEN_KWARGS`, which would change how or on what data a check runs;
- GX's own `validate_configuration` checks the kwarg values;
- a few generic rules cover what GX lets through (see `_policy_errors`).

So a contract can never smuggle in arbitrary classes, actions or code. Connection strings never live in the contract: the caller passes one in from the environment.

Great Expectations is imported inside the functions that need it, so parsing the DAG file never imports GX. This module must not import dq_agent; dq-agent imports it, so both sides use exactly the same rules.
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
CONTRACTS_DIR = Path(__file__).parent / "dq_contracts"

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
}

# Accepted by GX, but they change what data is checked, how results are reported, or let
# GX's profiler pick the parameters — none of that belongs in a reviewed contract.
FORBIDDEN_KWARGS = frozenset({
    "batch_id", "table", "row_condition", "condition_parser", "result_format",
    "include_config", "catch_exceptions", "auto", "profiler_config",
    "parse_strings_as_datetimes", "allow_cross_type_comparisons", "meta",
})

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# The only evaluation parameter a contract may carry: a relative date bound for freshness.
# Anything else under "$PARAMETER" is rejected, so GX never evaluates a free-form expression.
_DATE_PARAMETER = re.compile(r"^now\(\) - timedelta\(days=[1-9][0-9]*\)$")


class ContractError(ValueError):
    """The contract YAML is malformed or uses something outside the allowlist."""


class DataQualityCheckFailed(Exception):
    """The contract ran and at least one expectation failed."""


def allowed_kwargs(expectation_type: str) -> list[str]:
    """The kwargs a contract may pass to an allowed expectation, straight from GX."""
    from great_expectations.expectations.registry import get_expectation_impl

    impl = get_expectation_impl(expectation_type)
    keys = dict.fromkeys([*impl.domain_keys, *impl.success_keys])  # ordered, de-duplicated
    return [k for k in keys if k not in FORBIDDEN_KWARGS]


def contract_path(schema: str, table: str, root: Path = CONTRACTS_DIR) -> Path:
    """Where the contract for a table lives: one file per table, named `<schema>.<table>.yaml`."""
    for name in (schema, table):
        if not _IDENTIFIER.match(name):
            raise ContractError(f"invalid identifier '{name}'")
    return root / f"{schema}.{table}.yaml"


def load_table_contract(schema: str, table: str, root: Path = CONTRACTS_DIR) -> dict[str, Any]:
    """Load the contract for a table and check it really is that table's contract."""
    path = contract_path(schema, table, root)
    contract = load_contract(path)
    if (contract["dataset"]["schema"], contract["dataset"]["table"]) != (schema, table):
        raise ContractError(
            f"{path.name} describes {contract['suite']['name']}, not {schema}.{table}"
        )
    return contract


def load_contract(path: str | Path) -> dict[str, Any]:
    """Read and validate a contract. Raises ContractError naming what is wrong."""
    try:
        contract = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ContractError(f"{path}: not valid YAML ({exc})") from exc
    validate_contract(contract)
    return contract


def validate_contract(contract: Any) -> None:
    errors = contract_errors(contract)
    if errors:
        raise ContractError("invalid contract:\n- " + "\n- ".join(errors))


def contract_errors(contract: Any) -> list[str]:
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
        if not isinstance(approval.get("approved_at"), (datetime.datetime, datetime.date, str)):
            errors.append("approval.approved_at must be a timestamp")
        if not isinstance(approval.get("approved_by"), str) or not approval["approved_by"]:
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
    if not isinstance(expectation, dict):
        return [f"expectation #{index}: must be a mapping"]
    kind = expectation.get("expectation_type")
    where = f"expectation #{index} ({kind})"
    errors = _unknown_keys(expectation, {"expectation_type", "kwargs", "meta"}, where)

    if kind not in ALLOWED_EXPECTATIONS:
        return errors + [f"expectation #{index}: '{kind}' is not an allowed expectation_type"]
    kwargs = expectation.get("kwargs")
    if not isinstance(kwargs, dict):
        return errors + [f"{where}: kwargs must be a mapping"]

    allowed = allowed_kwargs(kind)
    for name in sorted(kwargs.keys() - set(allowed), key=str):
        errors.append(f"{where}: unexpected kwarg '{name}' (allowed: {', '.join(allowed)})")
    for name in sorted(ALLOWED_EXPECTATIONS[kind] - kwargs.keys()):
        errors.append(f"{where}: missing required kwarg '{name}'")
    errors += [f"{where}: {problem}" for problem in _policy_errors(kind, kwargs)]
    if not errors:
        # GX's own checks: required kwargs, value types, value ranges
        errors += [f"{where}: {problem}" for problem in _gx_errors(kind, kwargs)]

    if "meta" in expectation:
        errors += _meta_errors(where, expectation["meta"])
    return errors


def _policy_errors(kind: str, kwargs: dict[str, Any]) -> list[str]:
    """What GX 0.18.2 accepts but a reviewed contract should not."""
    errors = []
    if "column" in kwargs and not (isinstance(kwargs["column"], str) and kwargs["column"]):
        errors.append("column must be a non-empty string")
    if "value_set" in kwargs and not kwargs["value_set"]:
        errors.append("value_set must not be empty")
    for name in ("strict_min", "strict_max"):
        if name in kwargs and not isinstance(kwargs[name], bool):
            errors.append(f"{name} must be true or false")
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


def _gx_errors(kind: str, kwargs: dict[str, Any]) -> list[str]:
    from great_expectations.core.expectation_configuration import ExpectationConfiguration
    from great_expectations.exceptions import InvalidExpectationConfigurationError
    from great_expectations.expectations.registry import get_expectation_impl

    try:
        configuration = ExpectationConfiguration(expectation_type=kind, kwargs=kwargs)
        get_expectation_impl(kind)(configuration).validate_configuration(configuration)
    except (InvalidExpectationConfigurationError, TypeError, ValueError) as exc:
        return [str(exc)]
    return []


def _meta_errors(where: str, meta: Any) -> list[str]:
    # Only the dashboard's short description is allowed: {notes: {format, content}}
    if not isinstance(meta, dict) or set(meta) - {"notes"}:
        return [f"{where}: meta may only contain 'notes'"]
    notes = meta.get("notes")
    if not isinstance(notes, dict) or set(notes) != {"format", "content"} \
            or notes["format"] not in ("string", "markdown") \
            or not isinstance(notes["content"], str):
        return [f"{where}: meta.notes must be {{format: string|markdown, content: <text>}}"]
    return []


def _unknown_keys(mapping: dict, allowed: set[str], where: str) -> list[str]:
    return [f"{where}: unknown key '{key}'" for key in sorted(mapping.keys() - allowed, key=str)]


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

    `context` comes from `dataflow.validation_utils.gx_get_context()`, whose Validation Store
    decides where the result is persisted. The suite and checkpoint are built in memory for
    this run only — the contract YAML stays the durable definition of the rules. Each
    expectation's meta carries the approval, so every stored result records who approved the
    rules it ran.
    """
    from great_expectations.core.expectation_configuration import ExpectationConfiguration

    validate_contract(contract)  # never trust that the caller validated
    schema, table = contract["dataset"]["schema"], contract["dataset"]["table"]
    suite_name = contract["suite"]["name"]
    # Suite-level meta stays in the in-memory suite store; each expectation's config is copied
    # into the stored validation result, so the approval rides there to be persisted.
    provenance = {
        "contract_version": contract["contract_version"],
        "approved_by": contract["approval"]["approved_by"],
        "approved_at": str(contract["approval"]["approved_at"]),
    }

    source = context.sources.add_postgres(name="datasets", connection_string=connection_string)
    asset = source.add_table_asset(name=suite_name, schema_name=schema, table_name=table)
    suite = context.add_expectation_suite(
        expectation_suite_name=suite_name,
        expectations=[
            ExpectationConfiguration(
                expectation_type=e["expectation_type"],
                kwargs=e["kwargs"],
                meta={**e.get("meta", {}), "dq_contract": provenance},
            )
            for e in contract["suite"]["expectations"]
        ],
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
    logger.info(summarize(result))

    if raise_on_failure and not result["success"]:
        raise DataQualityCheckFailed(f"data quality check failed for {suite_name}:\n{summarize(result)}")
    return result


def summarize(checkpoint_result: Any) -> str:
    """One line per expectation: pass/fail, type, column and unexpected count. No row values."""
    lines = []
    for validation in checkpoint_result.list_validation_results():
        for r in validation.results:
            config = r.expectation_config
            column = config.kwargs.get("column", "<table>")
            unexpected = (r.result or {}).get("unexpected_count")
            lines.append(
                f"{'PASS' if r.success else 'FAIL'} {config.expectation_type} on {column}"
                + (f" ({unexpected} unexpected)" if unexpected else "")
            )
    return "\n".join(lines)
