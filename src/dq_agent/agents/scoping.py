"""Phase 3 scoping agent: the conversational front half of the tool.

A single LangGraph agent (sub-agent split deferred until it earns its complexity)
drives the scoping conversation: profile the dataset, discuss findings, query the
registry, propose a parameterized rule suite, and route it through the human
approval gate. The approval gate is a LangGraph `interrupt()` whose payload follows
agent-chat-ui's HITL schema (`action_requests` + `review_configs`), so the UI renders
approve/edit/reject controls without custom front-end work (see ACTION_PLAN.md).

The LLM never sees raw cell values (`profiler.redact()`) and never executes rules —
its only product is a draft contract. Approval stamps `approved_at`/`approved_by` and
persists the canonical YAML artifact; from there the deterministic engine takes over.

Serve locally for agent-chat-ui with `uv run langgraph dev` (see langgraph.json).
"""

from __future__ import annotations

import asyncio
import getpass
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Command, interrupt
from pydantic import field_validator

from dq_agent import connectors, profiler
from dq_agent.models import Contract, ContractRule
from dq_agent.registry import Registry
from dq_agent.report import describe_contract

DEFAULT_RULES_DIR = Path("registry/rules")
DEFAULT_CONTRACTS_DIR = Path("contracts")

SYSTEM_PROMPT = (Path(__file__).parent / "scoping_prompt.txt").read_text()


class ScopingState(MessagesState):
    profile: dict[str, Any] | None  # redacted profile report of the dataset under scoping
    draft: dict[str, Any] | None  # unapproved contract awaiting the approval gate
    contract_path: str | None  # set once the approved YAML artifact is persisted


class ProposedRule(ContractRule):
    """A contract rule as proposed by the agent — same shape, separate name so the
    tool schema reads as a proposal, not an approved artifact."""

    @field_validator("params", mode="before")
    @classmethod
    def _coerce_json_string(cls, value: Any) -> Any:
        # Some models (smaller ones especially, e.g. the gemini *-flash-lite tier)
        # serialize the nested params object as a JSON string instead of an object.
        # Parse it here so the first tool call validates, rather than relying on the
        # model to notice and retry.
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value  # let normal validation report the type error
        return value


def _make_tools(registry: Registry) -> list:
    @tool
    def profile_dataset(
        path: str, tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
        """Profile a local CSV or Parquet dataset. Returns a redacted statistical
        report: column types, null rates, uniqueness, distributions, semantic hints.
        Raw value examples (top values) are stripped; numeric/temporal min and max are
        disclosed as bounded aggregates."""
        file = Path(path)
        if not file.exists():
            return Command(update={"messages": [
                ToolMessage(f"error: no file at '{path}'", tool_call_id=tool_call_id)
            ]})
        if file.suffix == ".csv":
            df = connectors.load_csv(file)
        elif file.suffix == ".parquet":
            df = connectors.load_parquet(file)
        else:
            return Command(update={"messages": [
                ToolMessage(
                    f"error: unsupported file type '{file.suffix}' (csv or parquet only)",
                    tool_call_id=tool_call_id,
                )
            ]})

        report = profiler.redact(profiler.profile(df, dataset=file.stem))
        report_json = report.model_dump_json(exclude_none=True)
        return Command(update={
            "profile": report.model_dump(mode="json"),
            "draft": None,  # a new dataset invalidates any pending draft
            "messages": [ToolMessage(report_json, tool_call_id=tool_call_id)],
        })

    @tool
    def list_rules(tags: list[str] | None = None) -> str:
        """List the rules available in the registry, optionally filtered by tags
        (e.g. completeness, uniqueness, validity, freshness, volume). Returns each
        rule's id, description, tags, default severity and parameter specs."""
        lines = []
        for rule_id in registry.rule_ids:
            rule = registry.get(rule_id)
            if tags and not set(tags) & set(rule.tags):
                continue
            params = ", ".join(
                f"{name}: {spec.type}"
                + ("" if spec.required else f" = {spec.default!r}")
                for name, spec in rule.parameters.items()
            )
            lines.append(
                f"- {rule.id} [{', '.join(rule.tags)}] (severity: {rule.severity})\n"
                f"  {rule.description}\n"
                f"  params: {params or 'none'}"
            )
        return "\n".join(lines) or "no rules match those tags"

    @tool
    def propose_contract(
        rules: list[ProposedRule],
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Propose a draft contract for the profiled dataset: the list of rule ids
        with parameters (and optional severity overrides) you recommend. The draft
        is validated against the registry and shown to the owner for discussion —
        it is not approved or executed by this call.

        Each rule's `params` must be a JSON object (dict), not a string. Example:
          {"rule_id": "null_check", "params": {"column": "email"}}
          {"rule_id": "range_check", "params": {"column": "amount", "min_val": 0}}
        """
        report = state.get("profile")
        if report is None:
            return Command(update={"messages": [
                ToolMessage(
                    "error: profile the dataset first", tool_call_id=tool_call_id
                )
            ]})

        errors = _validate_rules(rules, registry)
        if errors:
            return Command(update={"messages": [
                ToolMessage("invalid proposal: " + "; ".join(errors),
                            tool_call_id=tool_call_id)
            ]})

        draft = Contract(
            dataset=report["dataset"],
            columns={c["name"]: c["dtype"] for c in report["columns"]},
            rules=[ContractRule(**r.model_dump()) for r in rules],
        )
        return Command(update={
            "draft": draft.model_dump(mode="json"),
            "messages": [ToolMessage(
                "draft recorded. Summarize it for the owner in plain English:\n"
                + describe_contract(draft, registry),
                tool_call_id=tool_call_id,
            )],
        })

    @tool
    def request_approval() -> None:
        """Send the current draft contract to the human approval gate. Call this
        immediately after propose_contract succeeds — do not wait for a conversational
        confirmation first."""
        # never executed: the graph routes this call to the approval node

    return [profile_dataset, list_rules, propose_contract, request_approval]


def _validate_rules(rules: list[ContractRule], registry: Registry) -> list[str]:
    errors = []
    for rule in rules:
        try:
            errors.extend(registry.validate_params(rule.rule_id, rule.params))
        except KeyError:
            errors.append(f"unknown rule '{rule.rule_id}'")
    return errors


_ACCEPT_WORDS = {"accept", "approve", "approved", "yes", "y", "looks good", "go ahead"}


def _decision(response: Any) -> dict[str, Any]:
    """Normalize a resume payload into a single decision dict.

    agent-chat-ui (main) resumes with {"decisions": [decision]} where decision["type"]
    is approve/edit/reject; a bare list or dict is accepted too. A free-text string —
    the raw-JSON fallback view, where the owner types instead of clicking — becomes an
    approve or reject decision so the text and widget paths converge."""
    if isinstance(response, dict) and "decisions" in response:
        decisions = response["decisions"] or [{}]
        response = decisions[0]
    if isinstance(response, list):
        response = response[0] if response else {}
    if isinstance(response, str):
        if response.strip().lower() in _ACCEPT_WORDS:
            return {"type": "approve"}
        return {"type": "reject", "message": response.strip()}
    return response if isinstance(response, dict) else {}


def _approver(args: dict[str, Any] | None) -> str:
    # localhost demo: single user, no auth — the OS user is the honest identity.
    # A deployed UI must supply approved_by in the interrupt response instead.
    if args and args.get("approved_by"):
        return args["approved_by"]
    return os.environ.get("DQ_AGENT_APPROVER") or getpass.getuser()


def _approval_node(contracts_dir: Path, registry: Registry):
    def approval(state: ScopingState) -> Command:
        # answer every pending tool call: the approval call gets the gate outcome,
        # anything bundled alongside it is refused
        pending = state["messages"][-1].tool_calls
        approval_id = next(
            tc["id"] for tc in pending if tc["name"] == "request_approval"
        )
        replies = [
            ToolMessage("skipped: resolve the approval request first",
                        tool_call_id=tc["id"])
            for tc in pending if tc["id"] != approval_id
        ]

        def respond(text: str, **update: Any) -> Command:
            return Command(
                update={"messages": [*replies, ToolMessage(text, tool_call_id=approval_id)],
                        **update},
                goto="agent",
            )

        draft = state.get("draft")
        if draft is None:
            return respond("error: no draft contract — call propose_contract first")

        contract = Contract.model_validate(draft)
        # The UI markdown-renders this, so the YAML goes in a fenced block to keep its
        # indentation; describe_contract already emits a markdown bullet list.
        description = ("Review the proposed data quality contract:\n\n"
                       + describe_contract(contract, registry)
                       + "\n\nFull definition:\n\n```yaml\n" + contract.to_yaml() + "```")
        # agent-inbox HITL schema (agent-chat-ui main): isAgentInboxInterruptSchema only
        # accepts the plural action_requests + review_configs shape, so this renders as
        # approve/edit/reject controls instead of dumping raw JSON.
        response = interrupt({
            "action_requests": [{
                "name": "approve_contract",
                "args": {"contract": draft},
                "description": description,
            }],
            "review_configs": [{
                "action_name": "approve_contract",
                "allowed_decisions": ["approve", "edit", "reject"],
            }],
        })

        decision = _decision(response)
        kind = decision.get("type")

        if kind == "edit":
            edited = decision.get("edited_action") or {}
            contract = Contract.model_validate(edited.get("args", {}).get("contract", {}))
            errors = _validate_rules(contract.rules, registry)
            if errors:
                return respond("owner's edited contract is invalid, fix and re-propose: "
                               + "; ".join(errors))
            kind = "approve"

        if kind == "approve":
            contract.approved_at = datetime.now(timezone.utc)
            contract.approved_by = _approver(decision.get("args"))
            contracts_dir.mkdir(parents=True, exist_ok=True)
            filename = re.sub(r"[^A-Za-z0-9_-]", "_", contract.dataset) + ".yaml"
            path = contracts_dir / filename
            path.write_text(contract.to_yaml())
            return respond(
                f"contract approved by {contract.approved_by} and saved to {path}",
                draft=None,
                contract_path=str(path),
            )

        # reject / anything non-affirmative: hand the feedback back to the agent so it
        # can iterate on the contract rather than silently dropping the session.
        return respond(f"owner did not approve; feedback: {decision.get('message')}")

    return approval


def build_graph(
    *,
    model: BaseChatModel | None = None,
    registry: Registry | None = None,
    contracts_dir: Path | str = DEFAULT_CONTRACTS_DIR,
    checkpointer: Any = None,
):
    """Compile the scoping graph. All collaborators are injectable for tests;
    defaults serve the localhost demo (model from $DQ_AGENT_MODEL, repo-root
    registry and contracts directories)."""
    if model is None:
        # Branch note: this diverges from main on purpose. The work environment is
        # air-gapped and reaches Bedrock only through the internal bedrock-proxy, so
        # the default model here is DeptBedrockChat rather than init_chat_model. The
        # `model` parameter stays injectable, so tests are unaffected.
        from dq_agent.agents.bedrock_chat import DEFAULT_MODEL_ID, DeptBedrockChat

        model = DeptBedrockChat(
            model_id=os.environ.get("DQ_AGENT_MODEL", DEFAULT_MODEL_ID)
        )
    if registry is None:
        registry = Registry(DEFAULT_RULES_DIR)

    tools = _make_tools(registry)
    model = model.bind_tools(tools)

    def agent(state: ScopingState) -> dict:
        messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [model.invoke(messages)]}

    def route(state: ScopingState) -> str:
        calls = state["messages"][-1].tool_calls
        if not calls:
            return END
        if any(tc["name"] == "request_approval" for tc in calls):
            return "approval"
        return "tools"

    builder = StateGraph(ScopingState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode([t for t in tools if t.name != "request_approval"]))
    builder.add_node("approval", _approval_node(Path(contracts_dir), registry))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, ["tools", "approval", END])
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)


async def make_graph():
    """Zero-arg factory for LangGraph Server (referenced from langgraph.json);
    the server injects its own checkpointer. Async because the server invokes
    factories on its event loop, and build_graph does blocking work (registry
    directory scan, model init) that must run on a worker thread."""
    return await asyncio.to_thread(build_graph)
