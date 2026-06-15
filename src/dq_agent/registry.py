"""Loads and indexes the YAML rule definitions from registry/rules/ at startup.

Rules are config, the engine is the runner: each rule YAML names the module and
function that implement it, and the registry resolves that reference (cached
importlib lookup). Adding a rule means adding a YAML file plus a pure function —
never an engine change.

The registry serves both workflows: at scoping time its tags and parameter specs
are the catalogue the agent queries when proposing a contract (Phase 3); at run
time the engine uses it to validate params and route rule_ids to callables.
"""

from __future__ import annotations

import datetime
import importlib
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import BaseModel


class ParameterSpec(BaseModel):
    type: str
    required: bool = False
    default: Any = None


class ExecutionSpec(BaseModel):
    module: str
    function: str


class RuleDefinition(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str]
    severity: str
    parameters: dict[str, ParameterSpec]
    execution: ExecutionSpec


class Registry:
    def __init__(self, rules_dir: Path) -> None:
        self._rules: dict[str, RuleDefinition] = {}
        self._callables: dict[str, Callable] = {}
        self._load(rules_dir)

    def _load(self, rules_dir: Path) -> None:
        for path in sorted(rules_dir.glob("*.yaml")):
            with path.open() as f:
                data = yaml.safe_load(f)
            rule = RuleDefinition.model_validate(data)
            if rule.id in self._rules:
                raise ValueError(
                    f"duplicate rule id '{rule.id}' (second definition in {path.name})"
                )
            self._rules[rule.id] = rule

    def get(self, rule_id: str) -> RuleDefinition:
        if rule_id not in self._rules:
            raise KeyError(f"rule '{rule_id}' not found in registry")
        return self._rules[rule_id]

    def resolve(self, rule_id: str) -> Callable:
        if rule_id not in self._callables:
            rule = self.get(rule_id)
            module = importlib.import_module(rule.execution.module)
            fn = getattr(module, rule.execution.function)
            self._callables[rule_id] = fn
        return self._callables[rule_id]

    def validate_params(self, rule_id: str, params: dict[str, Any]) -> list[str]:
        rule = self.get(rule_id)
        errors = []
        for name, spec in rule.parameters.items():
            if spec.required and name not in params:
                errors.append(f"rule '{rule_id}': missing required param '{name}'")
        for name in sorted(set(params) - set(rule.parameters)):
            errors.append(f"rule '{rule_id}': unknown param '{name}'")
        for name, value in params.items():
            spec = rule.parameters.get(name)
            # None means "use the default"; unknown params are already reported above
            if spec is None or value is None:
                continue
            if not _type_matches(spec.type, value):
                errors.append(
                    f"rule '{rule_id}': param '{name}' must be {spec.type}, "
                    f"got {type(value).__name__}"
                )
        return errors

    @property
    def rule_ids(self) -> list[str]:
        return list(self._rules.keys())


def _type_matches(type_name: str, value: Any) -> bool:
    if type_name == "str":
        return isinstance(value, str)
    if type_name == "int":
        # bool is a subclass of int — a flag is not a count
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "float":
        # lenient: a whole number is a valid float (5 is a fine max_days, 0 a fine rate)
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "list":
        return isinstance(value, list)
    if type_name == "date":
        return _is_date_like(value)
    return True  # unrecognised type spec: don't block, it's documentation only


def _is_date_like(value: Any) -> bool:
    if isinstance(value, (datetime.date, datetime.datetime)):
        return True
    if isinstance(value, str):
        try:
            datetime.date.fromisoformat(value)
            return True
        except ValueError:
            return False
    return False
