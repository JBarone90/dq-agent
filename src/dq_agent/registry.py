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
        return errors

    @property
    def rule_ids(self) -> list[str]:
        return list(self._rules.keys())
