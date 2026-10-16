"""The scoping agent's catalogue of expectations, built from Great Expectations at startup.

Nothing about what a contract may contain is written here. For each expectation in the data-flow allowlist (`gx_contract.ALLOWED_EXPECTATIONS`), the registry asks GX 0.18.2 for:

- the expectation's one-line description (first line of its docstring);
- the kwargs a contract may pass (`gx_contract.allowed_kwargs`: GX's domain and success keys minus the forbidden ones);
- each kwarg's type and description (the docstring's Args / Keyword Args sections) and its default.

`registry/expectations.yaml` adds only our own layer: tags for filtering, a plain-English `summary` for the approval gate, and `notes` with usage advice and backend caveats GX does not document. So adding an expectation is one line in the allowlist, plus optionally a notes entry — the agent can never be offered something data-flow would reject.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from dq_agent import gx_contract

_DOC_SECTIONS = ("Args:", "Keyword Args:")
_DOC_HEADER = re.compile(r"^[A-Z][A-Za-z ]*:$")
_DOC_ARG = re.compile(r"^(\w+) \(([^)]*)\):\s*(.*)$")


class KwargSpec(BaseModel):
    type: str | None = None  # as GX's docstring states it, e.g. "comparable type or None"
    description: str | None = None
    default: Any = None


class RuleDefinition(BaseModel):
    id: str  # the GX expectation_type
    description: str
    tags: list[str] = []
    kwargs: dict[str, KwargSpec]
    required: list[str] = []  # what our policy requires beyond GX (GX reports its own on validation)
    notes: str | None = None
    summary: str | None = None


class Registry:
    def __init__(self, notes_file: Path) -> None:
        with Path(notes_file).open(encoding="utf-8") as f:
            ours = yaml.safe_load(f) or {}
        unknown = set(ours) - set(gx_contract.ALLOWED_EXPECTATIONS)
        if unknown:
            raise ValueError(f"{notes_file} describes expectations not in the allowlist: {sorted(unknown)}")
        self._rules = {
            name: _from_gx(name, ours.get(name) or {})
            for name in gx_contract.ALLOWED_EXPECTATIONS
        }

    def get(self, rule_id: str) -> RuleDefinition:
        if rule_id not in self._rules:
            raise KeyError(f"rule '{rule_id}' not found in registry")
        return self._rules[rule_id]

    def describe_unlisted(self, expectation_type: str) -> RuleDefinition:
        """The same GX-derived description for an expectation the allowlist lacks — what
        creative mode needs to fill in a proposal's kwargs."""
        return _from_gx(expectation_type, {})

    @property
    def rule_ids(self) -> list[str]:
        return list(self._rules.keys())

    def summarize(self, expectation_type: str, kwargs: dict[str, Any]) -> str:
        """Render the rule's summary template with its kwargs, for a non-engineer."""
        rule = self._rules.get(expectation_type)
        if rule is None or rule.summary is None:
            shown = ", ".join(f"{k}={_display(v)}" for k, v in kwargs.items() if k != "mostly")
            text = expectation_type + (f" ({shown})" if shown else "")
        else:
            text = _one_sided(rule.summary, kwargs).format_map(
                {name: _display(kwargs.get(name)) for name in rule.kwargs})
        mostly = kwargs.get("mostly")
        if mostly is not None and mostly < 1:
            text += f" (for at least {mostly * 100:g}% of rows)"
        return text


def _from_gx(name: str, ours: dict[str, Any]) -> RuleDefinition:
    from great_expectations.expectations.registry import get_expectation_impl

    impl = get_expectation_impl(name)
    documented = _documented_args(impl.__doc__ or "")
    defaults = impl.default_kwarg_values
    kwargs = {}
    for kwarg in gx_contract.allowed_kwargs(name):
        doc_type, doc = documented.get(kwarg, (None, None))
        default = defaults.get(kwarg)
        kwargs[kwarg] = KwargSpec(
            type=doc_type,
            description=doc,
            # an empty list default (value_set) is a placeholder, not a usable default
            default=default if default not in (None, [], {}) else None,
        )
    return RuleDefinition(
        id=name,
        description=(impl.__doc__ or name).strip().splitlines()[0],
        tags=ours.get("tags", []),
        kwargs=kwargs,
        required=sorted(gx_contract.ALLOWED_EXPECTATIONS.get(name, frozenset())),
        notes=ours.get("notes"),
        summary=ours.get("summary"),
    )


def _documented_args(doc: str) -> dict[str, tuple[str, str]]:
    """kwarg -> (type, description) from a GX docstring's Args / Keyword Args sections."""
    found: dict[str, tuple[str, str]] = {}
    in_section = False
    for line in doc.splitlines():
        stripped = line.strip()
        if _DOC_HEADER.match(stripped):  # "Args:", "Other Parameters:", "Returns:", ...
            in_section = stripped in _DOC_SECTIONS
            continue
        match = _DOC_ARG.match(stripped)
        if in_section and match:
            # docstrings pad descriptions with runs of spaces; drop the "For more detail" links
            text = re.sub(r"\s+", " ", match.group(3)).split(" For more detail")[0].strip()
            found[match.group(1)] = (match.group(2), text)
    return found


_RANGE = "between {min_value} and {max_value}"


def _one_sided(template: str, kwargs: dict[str, Any]) -> str:
    """A range with one bound reads as that bound, not "between 0 and no limit"; a relative date
    bound reads as a date ("no earlier than 30 days ago")."""
    low, high = kwargs.get("min_value"), kwargs.get("max_value")
    if _RANGE not in template or (low is None) == (high is None):
        return template
    if high is not None:
        return template.replace(_RANGE, "at most {max_value}")
    is_date = isinstance(low, dict) and "$PARAMETER" in low
    return template.replace(_RANGE, "no earlier than {min_value}" if is_date else "at least {min_value}")


def _display(value: Any) -> str:
    if value is None:
        return "no limit"
    if isinstance(value, dict) and "$PARAMETER" in value:
        # the date bound: "now() - timedelta(days=N)" reads better as "N days ago"
        expression = str(value["$PARAMETER"])
        if expression.startswith("now() - timedelta(days=") and expression.endswith(")"):
            return expression.removeprefix("now() - timedelta(days=").removesuffix(")") + " days ago"
        return expression
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)
