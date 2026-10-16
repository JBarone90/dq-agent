"""The approval-gate protocol shared by the CLI and the Streamlit app."""

import pytest

pytest.importorskip("langgraph")

from langgraph.types import Command, Interrupt

from dq_agent.agents.gate import decision_from_text, pending_interrupt, resume, unwrap_decision

PAYLOAD = {"action_requests": [{"name": "approve_contract"}], "review_configs": []}


@pytest.mark.parametrize("result", [
    {"__interrupt__": [Interrupt(value=PAYLOAD)]},
    {"__interrupt__": (Interrupt(value=PAYLOAD),)},
    {"__interrupt__": [PAYLOAD]},  # a snapshot rebuilt without Interrupt objects
])
def test_pending_interrupt_returns_the_payload(result):
    assert pending_interrupt(result) == PAYLOAD


@pytest.mark.parametrize("result", [None, {}, {"__interrupt__": []}, {"messages": []}])
def test_no_pending_interrupt(result):
    assert pending_interrupt(result) is None


def test_resume_wraps_one_decision():
    command = resume({"type": "approve"})
    assert isinstance(command, Command)
    assert command.resume == {"decisions": [{"type": "approve"}]}


@pytest.mark.parametrize("text", ["approve", "  Yes ", "looks good", "OK"])
def test_approve_words_approve(text):
    assert decision_from_text(text) == {"type": "approve"}


def test_anything_else_is_feedback():
    assert decision_from_text(" make amount a warning ") == {"type": "reject", "message": "make amount a warning"}


@pytest.mark.parametrize("response, decision", [
    ({"decisions": [{"type": "approve"}]}, {"type": "approve"}),
    ([{"type": "approve"}], {"type": "approve"}),
    ({"type": "reject", "message": "x"}, {"type": "reject", "message": "x"}),
    ("approve", {"type": "approve"}),
    ("loosen the null rate", {"type": "reject", "message": "loosen the null rate"}),
    ({"decisions": []}, {}),
    ([], {}),
    (None, {}),
])
def test_unwrap_decision_accepts_every_resume_shape(response, decision):
    assert unwrap_decision(response) == decision
