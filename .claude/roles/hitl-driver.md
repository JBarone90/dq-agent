---
role: hitl-driver
applies_when: Editing the scoping agent's approval gate or interrupt payloads (src/dq_agent/agents/scoping.py), or the local driver that consumes them (scripts/scoping_cli.py, an optional Streamlit app)
description: The contract between this project's LangGraph approval gate and the local human-in-the-loop driver (interrupt schema, resume schema, markdown rendering, checkpointer requirement)
---

# HITL Driver Role

This branch (`feat/bedrock-proxy-adapter`) is air-gapped with no npm mirror, so there is no agent-chat-ui and no `langgraph dev` server. The human-in-the-loop interface is a **local, in-process driver** — the CLI in `scripts/scoping_cli.py` today, optionally a Streamlit chat panel. A driver's only coupling to the graph is three things: it must **compile the graph with a checkpointer**, it reads the **interrupt payload** the `approval` node emits, and it sends back a **resume payload**. Keep those three shapes stable and any driver keeps working.

## Checkpointer is mandatory — this is the #1 gotcha

The approval gate calls `interrupt()` (`_approval_node` in `scoping.py`). `interrupt()` pauses the graph, persists the run to the checkpointer keyed by `thread_id`, and returns from `invoke()`. The resume call (`Command(resume=...)`) looks the run back up by `thread_id` and re-enters the node. **With no checkpointer there is nowhere to persist the paused run, so the gate never completes and no contract is written** — the exact symptom of "fails after the human-in-the-loop step."

- `build_graph(checkpointer=...)` is the injection point. `MemorySaver` for an ephemeral session; `SqliteSaver` (from `langgraph-checkpoint-sqlite`) for a thread that survives a process restart.
- Every `invoke` for one conversation must pass the same `config={"configurable": {"thread_id": ...}}`.
- `interrupt()` re-runs its node **from the top** on resume. Never put a side effect (file write, proxy call) before the `interrupt()` line in `_approval_node` — it would fire twice. Today the pre-interrupt code only builds the description, which is idempotent.

## Outgoing: the interrupt payload the gate emits

`_approval_node` emits the project's HITL contract — the plural shape, non-empty arrays:

```python
interrupt({
    "action_requests": [{
        "name": "approve_contract",
        "args": {"contract": draft},
        "description": "<markdown shown to the owner>",
    }],
    "review_configs": [{
        "action_name": "approve_contract",
        "allowed_decisions": ["approve", "edit", "reject"],
    }],
})
```

A driver detects the pause with `result.get("__interrupt__")`, then reads
`__interrupt__[0].value["action_requests"][0]["description"]` to show the owner and
`...["review_configs"][0]["allowed_decisions"]` for the choices to offer.

## Incoming: the resume payload the driver sends back

Resume with `Command(resume={"decisions": [decision]})`. Each `decision` is one of:

- `{"type": "approve"}`
- `{"type": "reject", "message"?: str}`
- `{"type": "edit", "edited_action": {"name": str, "args": {"contract": {...}}}}` — the owner's modified contract lives in `edited_action["args"]["contract"]`.

`scoping.py:_decision()` normalizes all of this — `{"decisions": [...]}`, a bare list, a bare dict, or a **free-text string** — into one decision dict. The free-text path is what lets a plain CLI accept typed input (`approve`, or any other text → reject-with-feedback) without rendering buttons. Keep that normalizer if you touch the gate.

## A driver loop must separate "turn" from "resume"

A scoping conversation is multi-turn: the agent asks clarifying questions and returns to `END` (no tool call) **before** ever reaching the approval interrupt. So a driver cannot assume every input after the first is a resume. Branch on the interrupt:

```python
if result.get("__interrupt__"):          # paused at the gate → resume
    result = graph.invoke(Command(resume={"decisions": [decision]}), config)
else:                                     # ordinary conversational turn → new message
    result = graph.invoke({"messages": [{"role": "user", "content": text}]}, config)
```

Stop when `result.get("contract_path")` is set — that is the approved-and-persisted signal.

## The description is markdown — emit markdown

`describe_contract` (`report.py`) is rendered as markdown by the driver's approval view. Use `- ` list markers and fenced ` ```yaml ` blocks for YAML; a literal `•` list or raw-indented YAML collapses. Anything new added to the `description` must stay valid markdown.

## Our side of the contract

- `src/dq_agent/agents/scoping.py` — `_approval_node` (emits the interrupt, handles the decision), `_decision` (resume normalizer), `build_graph(checkpointer=...)`.
- `src/dq_agent/report.py` — `describe_contract` (the plain-English, markdown rule summary).
- `scripts/scoping_cli.py` — the reference driver; mirror its turn-vs-resume handling in any new UI.
- Tests: `tests/test_agents.py` (interrupt shape, approve/edit/reject/text-string resume) and `tests/test_report.py` (markdown list markers).
