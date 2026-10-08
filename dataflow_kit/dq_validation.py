"""Run an approved data quality contract with Great Expectations 0.18.2.

Staged in dq-agent, copied into the `data-flow` repo at work. Three pieces make up a data quality check there, and only the third changes per table:

- this module: loads a contract YAML, validates it against a small allowlist, and runs it as an in-memory Expectation Suite + Checkpoint;
- the validation DAG (`dags/dq_check_example.py`): calls this module from a `@task.virtualenv` task;
- the contract: `dq_contracts/<schema>.<table>.yaml`, produced by the dq-agent scoping agent and approved by the dataset owner.

The contract uses GX expectation names and kwargs directly — there is no translation layer. The allowlist below only decides *which* expectations and kwargs a contract may use, so a contract can never smuggle in arbitrary classes, actions or code. Connection strings never live in the contract: the caller passes one in from the environment.

Great Expectations is imported inside `run_contract` only, so loading and validating a contract (and parsing the DAG file) does not need GX installed.

This module is deliberately self-contained (stdlib + PyYAML + GX): it must not import dq_agent. dq-agent's own registry mirrors `ALLOWED_EXPECTATIONS` and a dq-agent test keeps the two in step.
"""

from __future__ import annotations

import datetime
import logging
import re
from pathlib import Path
from typing import Any, Callable

import yaml

logger = logging.getLogger(__name__)

CONTRACT_VERSION = 1
GX_VERSION = "0.18.2"
CONTRACTS_DIR = Path(__file__).parent / "dq_contracts"

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# The only evaluation parameter a contract may carry: a relative freshness bound. Anything
# else under "$PARAMETER" is rejected, so GX never evaluates a free-form expression.
_FRESHNESS_PARAMETER = re.compile(r"^now\(\) - timedelta\(days=[1-9][0-9]*\)$")


class ContractError(ValueError):
    """The contract YAML is malformed or uses something outside the allowlist."""


class DataQualityCheckFailed(Exception):
    """The contract ran and at least one expectation failed."""


# --- kwarg checks: each returns an error message, or None when the value is acceptable ---

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _string(value: Any) -> str | None:
    return None if isinstance(value, str) and value else "must be a non-empty string"


def _number(value: Any) -> str | None:
    return None if _is_number(value) else "must be a number"


def _count(value: Any) -> str | None:
    ok = isinstance(value, int) and not isinstance(value, bool) and value >= 0
    return None if ok else "must be a non-negative integer"


def _fraction(value: Any) -> str | None:
    return None if _is_number(value) and 0 <= value <= 1 else "must be a number between 0 and 1"


def _boolean(value: Any) -> str | None:
    return None if isinstance(value, bool) else "must be true or false"


def _non_empty_list(value: Any) -> str | None:
    if not isinstance(value, list) or not value:
        return "must be a non-empty list"
    if any(isinstance(v, (dict, list)) for v in value):
        return "must contain plain values only"
    return None


def _string_list(value: Any) -> str | None:
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
        return "must be a non-empty list of strings"
    return None


def _regex(value: Any) -> str | None:
    # GX evaluates this in the database (Postgres `~`, POSIX syntax), so Python's re.compile
    # is only a sanity check that the pattern is not obviously broken.
    if not isinstance(value, str) or not value:
        return "must be a non-empty string"
    try:
        re.compile(value)
    except re.error as exc:
        return f"is not a valid regex ({exc})"
    return None


def _freshness_bound(value: Any) -> str | None:
    if isinstance(value, dict) and set(value) == {"$PARAMETER"} and isinstance(value["$PARAMETER"], str) \
            and _FRESHNESS_PARAMETER.match(value["$PARAMETER"]):
        return None
    return 'must be {"$PARAMETER": "now() - timedelta(days=N)"}'


# expectation_type -> {"required": {kwarg: check}, "optional": {kwarg: check}}.
# Adding a standard expectation is one entry here (plus the matching dq-agent registry YAML).
ALLOWED_EXPECTATIONS: dict[str, dict[str, dict[str, Callable[[Any], str | None]]]] = {
    "expect_column_values_to_not_be_null": {
        "required": {"column": _string},
        "optional": {"mostly": _fraction},
    },
    "expect_column_values_to_be_unique": {
        "required": {"column": _string},
        "optional": {"mostly": _fraction},
    },
    "expect_column_values_to_be_in_set": {
        "required": {"column": _string, "value_set": _non_empty_list},
        "optional": {"mostly": _fraction},
    },
    "expect_column_values_to_be_between": {
        "required": {"column": _string},
        "optional": {"min_value": _number, "max_value": _number, "strict_min": _boolean,
                     "strict_max": _boolean, "mostly": _fraction},
    },
    "expect_column_values_to_match_regex": {
        "required": {"column": _string, "regex": _regex},
        "optional": {"mostly": _fraction},
    },
    "expect_table_row_count_to_be_between": {
        "required": {},
        "optional": {"min_value": _count, "max_value": _count},
    },
    "expect_column_values_to_be_in_type_list": {
        "required": {"column": _string, "type_list": _string_list},
        "optional": {},
    },
    "expect_column_min_to_be_between": {
        "required": {"column": _string, "min_value": _freshness_bound},
        "optional": {},
    },
}

# Expectations that need at least one of these kwargs — a range with no bound checks nothing.
_ONE_OF = {
    "expect_column_values_to_be_between": ("min_value", "max_value"),
    "expect_table_row_count_to_be_between": ("min_value", "max_value"),
}


def contract_path(schema: str, table: str, root: Path = CONTRACTS_DIR) -> Path:
    """Where the contract for a table lives: one file per table, named `<schema>.<table>.yaml`."""
    for name in (schema, table):
        if not _IDENTIFIER.match(name):
            raise ContractError(f"invalid identifier '{name}'")
    return root / f"{schema}.{table}.yaml"


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

    spec = ALLOWED_EXPECTATIONS.get(kind)
    if spec is None:
        return errors + [f"expectation #{index}: '{kind}' is not an allowed expectation_type"]

    kwargs = expectation.get("kwargs")
    if not isinstance(kwargs, dict):
        return errors + [f"{where}: kwargs must be a mapping"]
    allowed = {**spec["required"], **spec["optional"]}
    for name in kwargs.keys() - allowed.keys():
        errors.append(f"{where}: unexpected kwarg '{name}'")
    for name in spec["required"].keys() - kwargs.keys():
        errors.append(f"{where}: missing required kwarg '{name}'")
    for name, value in kwargs.items():
        if name in allowed and (problem := allowed[name](value)):
            errors.append(f"{where}: {name} {problem}")
    one_of = _ONE_OF.get(kind)
    if one_of and not any(kwargs.get(k) is not None for k in one_of):
        errors.append(f"{where}: needs at least one of {', '.join(one_of)}")

    if "meta" in expectation:
        errors += _meta_errors(where, expectation["meta"])
    return errors


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
    this run only — the contract YAML stays the durable definition of the rules.
    """
    from great_expectations.core.expectation_configuration import ExpectationConfiguration

    validate_contract(contract)  # never trust that the caller validated
    schema, table = contract["dataset"]["schema"], contract["dataset"]["table"]
    suite_name = contract["suite"]["name"]

    source = context.sources.add_postgres(name="datasets", connection_string=connection_string)
    asset = source.add_table_asset(name=suite_name, schema_name=schema, table_name=table)
    suite = context.add_expectation_suite(
        expectation_suite_name=suite_name,
        expectations=[
            ExpectationConfiguration(
                expectation_type=e["expectation_type"],
                kwargs=e["kwargs"],
                meta=e.get("meta", {}),
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
