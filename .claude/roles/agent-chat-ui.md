---
role: agent-chat-ui
applies_when: Editing the scoping agent's approval gate or interrupt payloads (src/dq_agent/agents/scoping.py), or debugging how the approval card renders in agent-chat-ui
description: The integration contract between this project's LangGraph approval gate and LangChain's agent-chat-ui (interrupt schema, resume schema, markdown rendering)
---

# agent-chat-ui Integration Role

The chat UI ([agent-chat-ui](https://github.com/langchain-ai/agent-chat-ui)) is off-the-shelf and we do not fork it. Our only coupling to it is two shapes: the **interrupt payload** we emit from `interrupt()`, and the **resume payload** it sends back. Both are owned by the UI's `main` branch and have changed before without notice — if the approval widget regresses, suspect a schema drift here first, and verify against source (paths below) rather than memory. This file captures what was true when verified against `main` on **2026-06-18**; treat it as a strong hint, not gospel — re-check the source files if behaviour disagrees.

## The widget-vs-raw-JSON decision

`src/components/thread/messages/ai.tsx` renders the interactive approval card (`ThreadView`) only when `isAgentInboxInterruptSchema(interrupt)` returns true; otherwise it falls back to `GenericInterruptView`, which **dumps the raw interrupt JSON**. So "I see raw JSON instead of approve/edit/reject buttons" always means our payload failed that schema check.

`isAgentInboxInterruptSchema` (`src/lib/agent-inbox-interrupt.ts`) requires the **plural HITL** shape — non-empty arrays, specific keys:

- `interrupt.value.action_requests`: non-empty array; each item needs `name: string` and `args: object`.
- `interrupt.value.review_configs`: non-empty array; each item needs `action_name: string` and `allowed_decisions: array`.

An older `HumanInterrupt` schema (singular `action_request` + `config` with `allow_accept`/`allow_edit`/… booleans) is what most LangGraph docs and tutorials still show. It no longer renders — it fails the check on the first missing `action_requests` key.

## Outgoing: the interrupt payload we must emit

Emit a single object (not a list) matching the HITL schema. This is exactly what `_approval_node` in `scoping.py` builds:

```python
interrupt({
    "action_requests": [{
        "name": "approve_contract",
        "args": {"contract": draft},
        "description": "<markdown shown in the Description tab>",
    }],
    "review_configs": [{
        "action_name": "approve_contract",
        "allowed_decisions": ["approve", "edit", "reject"],
    }],
})
```

`allowed_decisions` values are the `DecisionType` union: `"approve" | "edit" | "reject"` (`types.ts`).

## Incoming: the resume payload it sends back

On submit the UI resumes with `Command(resume={"decisions": [decision]})` (`use-interrupted-actions.tsx`). Each `decision` is one of (`types.ts`):

- `{"type": "approve"}`
- `{"type": "reject", "message"?: str}`
- `{"type": "edit", "edited_action": {"name": str, "args": {...}}}` — the owner's modified contract lives in `edited_action["args"]["contract"]`.

`scoping.py:_decision()` normalizes all of this — `{"decisions": [...]}`, a bare list, a bare dict, or a free-text string (the raw-JSON fallback view where the owner types instead of clicking) — into one decision dict. Keep that normalizer if you change the gate; the free-text path is what lets a non-rendering client still approve.

## The Description tab is markdown — emit markdown

The approval card's `StateView` (`agent-inbox/components/state-view.tsx`) has two tabs:

- **State** — renders the LangGraph thread state (`values`) and the message history. We author none of this; it is the UI introspecting the run.
- **Description** — renders `action_requests[0].description` through a markdown component (`<MarkdownText>`).

Because Description is markdown-rendered, plain text breaks: a literal `•` bullet list collapses onto one line (single `\n` becomes a soft break), and raw YAML loses its indentation. So the `description` string must be valid markdown — `- ` list markers (see `report.describe_contract`) and fenced ` ```yaml ` blocks for any YAML. Anything new we put in `description` must follow the same rule.

## Our side of the contract

- `src/dq_agent/agents/scoping.py` — `_approval_node` (emits the interrupt, handles the decision) and `_decision` (resume normalizer).
- `src/dq_agent/report.py` — `describe_contract` (the plain-English, markdown rule summary shown in the Description tab).
- Tests: `tests/test_agents.py` (interrupt shape, approve/edit/reject/text-string resume) and `tests/test_report.py` (markdown list markers).

## Re-verifying against the UI source

When in doubt, read these on the agent-chat-ui `main` branch instead of guessing (raw URLs fetch cleanly):

- `src/lib/agent-inbox-interrupt.ts` — `isAgentInboxInterruptSchema` (the gate).
- `src/components/thread/agent-inbox/types.ts` — `ActionRequest`, `ReviewConfig`, `HITLRequest`, `Decision`, `DecisionType`.
- `src/components/thread/messages/ai.tsx` — widget-vs-`GenericInterruptView` selection.
- `src/components/thread/agent-inbox/components/state-view.tsx` — State/Description tabs.
- `src/components/thread/agent-inbox/hooks/use-interrupted-actions.tsx` — the resume submit shape.

The hosted client at `agentchat.vercel.app` tracks `main`; running agent-chat-ui locally pins it to a known commit if you need certainty.
