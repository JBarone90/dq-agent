"""Loads and indexes the expectation catalogue in registry/rules/ at startup.

Each YAML file describes one Great Expectations 0.18.2 expectation the scoping agent may propose: its GX name (`id`), a plain description, tags, its kwargs and a `summary` template for the approval gate. The registry is the agent's catalogue only. Whether a contract is actually valid is decided by the data-flow allowlist (`dataflow_kit.dq_validation`), which is what runs at work; a test keeps the two in step, so the agent can never offer something data-flow would reject.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class KwargSpec(BaseModel):
    type: str
    required: bool = False
    description: str | None = None


class RuleDefinition(BaseModel):
    id: str  # the GX expectation_type
    name: str
    description: str
    tags: list[str]
    kwargs: dict[str, KwargSpec]
    # plain-English phrasing for the approval gate; `{kwarg}` placeholders are filled in
    summary: str


class Registry:
    def __init__(self, rules_dir: Path) -> None:
        self._rules: dict[str, RuleDefinition] = {}
        self._load(rules_dir)

    def _load(self, rules_dir: Path) -> None:
        for path in sorted(rules_dir.glob("*.yaml")):
            with path.open(encoding="utf-8") as f:
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

    @property
    def rule_ids(self) -> list[str]:
        return list(self._rules.keys())

    def summarize(self, expectation_type: str, kwargs: dict[str, Any]) -> str:
        """Render the rule's summary template with its kwargs, for a non-engineer."""
        try:
            rule = self.get(expectation_type)
        except KeyError:
            return expectation_type
        values = {name: _display(kwargs.get(name)) for name in rule.kwargs}
        text = rule.summary.format_map(values)
        mostly = kwargs.get("mostly")
        if mostly is not None and mostly < 1:
            text += f" (for at least {mostly * 100:g}% of rows)"
        return text


def _display(value: Any) -> str:
    if value is None:
        return "no limit"
    if isinstance(value, dict) and "$PARAMETER" in value:
        # the freshness bound: "now() - timedelta(days=N)" reads better as "N days ago"
        expression = str(value["$PARAMETER"])
        if expression.startswith("now() - timedelta(days=") and expression.endswith(")"):
            return expression.removeprefix("now() - timedelta(days=").removesuffix(")") + " days"
        return expression
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)
