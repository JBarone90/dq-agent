import pytest

from dq_agent.registry import Registry
from dq_agent.registry.loader import RuleDefinition
from tests.conftest import RULES_DIR

EXPECTED_RULE_IDS = {
    "null_check",
    "unique_check",
    "range_check",
    "allowed_values",
    "regex_match",
    "freshness",
}


def test_registry_loads_all_rules(registry: Registry):
    assert set(registry.rule_ids) == EXPECTED_RULE_IDS


def test_registry_get_returns_rule_definition(registry: Registry):
    rule = registry.get("null_check")
    assert isinstance(rule, RuleDefinition)
    assert rule.id == "null_check"
    assert rule.severity == "error"
    assert "completeness" in rule.tags


def test_registry_get_raises_for_unknown_rule(registry: Registry):
    with pytest.raises(KeyError, match="not found in registry"):
        registry.get("nonexistent_rule")


def test_registry_resolve_returns_callable(registry: Registry):
    fn = registry.resolve("null_check")
    assert callable(fn)
    assert fn.__name__ == "null_check"


def test_registry_resolve_caches_callable(registry: Registry):
    fn1 = registry.resolve("range_check")
    fn2 = registry.resolve("range_check")
    assert fn1 is fn2


def test_registry_validate_params_passes_when_required_present(registry: Registry):
    errors = registry.validate_params("null_check", {"column": "order_id"})
    assert errors == []


def test_registry_validate_params_catches_missing_required(registry: Registry):
    errors = registry.validate_params("null_check", {})
    assert len(errors) == 1
    assert "column" in errors[0]


def test_registry_validate_params_catches_multiple_missing(registry: Registry):
    errors = registry.validate_params("regex_match", {})
    missing = {e for e in errors}
    assert len(missing) == 2


def test_registry_validate_params_optional_not_required(registry: Registry):
    errors = registry.validate_params("null_check", {"column": "x", "max_null_rate": 0.05})
    assert errors == []


def test_registry_freshness_rule_is_warning(registry: Registry):
    rule = registry.get("freshness")
    assert rule.severity == "warning"


def test_registry_rule_parameters_include_required_flag(registry: Registry):
    rule = registry.get("freshness")
    assert rule.parameters["column"].required is True
    assert rule.parameters["max_days"].required is True
