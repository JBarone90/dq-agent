"""The default chat model, `dq_agent.agents.llm`: how it is pointed at the proxy and what it
puts in a request. Nothing here sends one; `test_live_proxy.py` does that at work."""

import pytest

pytest.importorskip("langchain_anthropic")

from langchain_core.messages import HumanMessage, SystemMessage

from dq_agent.agents import llm


@pytest.fixture
def proxy_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://proxy.invalid/anthropic")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token")


def test_the_model_uses_the_proxy_url_and_token(proxy_env):
    model = llm.default_model()
    assert model.model == llm.DEFAULT_MODEL_ID
    assert str(model._client.base_url).startswith("http://proxy.invalid/anthropic")
    assert model.anthropic_api_key.get_secret_value() == "token"


def test_requests_carry_automatic_caching_with_a_1h_ttl(proxy_env):
    payload = llm.default_model()._get_request_payload(
        [SystemMessage("You scope data quality."), HumanMessage("hi")])
    assert payload["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert payload["max_tokens"] == llm.MAX_TOKENS


def test_caching_can_be_switched_off(proxy_env):
    payload = llm.default_model(cache_prompt=False)._get_request_payload([HumanMessage("hi")])
    assert "cache_control" not in payload


@pytest.mark.parametrize("unset", ["ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"])
def test_without_the_proxy_environment_the_error_says_what_to_do(proxy_env, monkeypatch, unset):
    """Off the work network the variables are absent; without the base URL ChatAnthropic would
    quietly aim at api.anthropic.com instead."""
    monkeypatch.delenv(unset)
    with pytest.raises(RuntimeError, match=f"{unset} not set.*build_graph\\(model=...\\)"):
        llm.default_model()
