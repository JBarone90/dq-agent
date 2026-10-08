"""Phase 3 scoping agent: the conversational front half of the tool.

A single LangGraph agent (sub-agent split deferred until it earns its complexity)
drives the scoping conversation: profile the dataset, discuss findings, query the
registry, propose a parameterized rule suite, and route it through the human
approval gate. The approval gate is a LangGraph `interrupt()` whose payload is the
project's own HITL contract (`action_requests` + `review_configs`), consumed by the
local driver (a CLI or a Streamlit chat panel) that supplies the resume decision.

The LLM never sees raw cell values (`profiler.redact()`) and never executes rules —
its only product is a draft contract: a Great Expectations 0.18.2 suite in the data-flow
envelope. Approval stamps `approval` and persists `<schema>.<table>.yaml`; from there the
data-flow Airflow task (`dataflow_kit/`) runs it.

Drive locally with the CLI in `scripts/scoping_cli.py`. Any driver must compile the
graph with a checkpointer (see `build_graph(checkpointer=...)`): `interrupt()` cannot
pause and resume without one.
"""

from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
from typing import Annotated, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Command, interrupt
from pydantic import field_validator

from dataflow_kit import dq_validation
from dq_agent import connectors, profiler
from dq_agent.models import Contract, Expectation
from dq_agent.registry import Registry
from dq_agent.report import describe_contract

DEFAULT_RULES_DIR = Path("registry/rules")
DEFAULT_CONTRACTS_DIR = Path("contracts")

# Deterministic profiling cap: the code, not the LLM, decides how much to pull. Tables
# under this load whole; larger ones come back as a block-level sample (flagged sampled).
PROFILE_MAX_ROWS = 200_000

SYSTEM_PROMPT = (Path(__file__).parent / "scoping_prompt.txt").read_text()


class ScopingState(MessagesState):
    profile: dict[str, Any] | None  # redacted profile report of the dataset under scoping
    # the Postgres table the profile came from, or None for a local file. Set so the
    # propose step can confirm range bounds against the live table when the profile is
    # a sample; never holds a connection string (the DSN is re-resolved from the env).
    source_table: str | None
    draft: dict[str, Any] | None  # unapproved contract awaiting the approval gate
    contract_path: str | None  # set once the approved YAML artifact is persisted


class ProposedExpectation(Expectation):
    """An expectation as proposed by the agent — same shape, separate name so the
    tool schema reads as a proposal, not an approved artifact."""

    @field_validator("kwargs", mode="before")
    @classmethod
    def _coerce_json_string(cls, value: Any) -> Any:
        # Some models (smaller ones especially, e.g. the gemini *-flash-lite tier)
        # serialize the nested kwargs object as a JSON string instead of an object.
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
            "source_table": None,  # a local file: nothing to confirm bounds against
            "draft": None,  # a new dataset invalidates any pending draft
            "messages": [ToolMessage(report_json, tool_call_id=tool_call_id)],
        })

    @tool
    def profile_table(
        table: str, tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
        """Profile a Postgres table by schema-qualified name ('schema.table'). Loads
        adaptively: a small table whole, a large one as a representative block-level
        sample — the report is then flagged `sampled: true` and its counts, uniqueness
        and ranges are estimates. The connection is read from the environment, never
        from you: supply only the table name. Returns the same redacted report as
        profile_dataset."""
        try:
            uri = connectors.resolve_dsn()
            load = connectors.load_postgres_profiling(
                uri, table=table, max_rows=PROFILE_MAX_ROWS
            )
        except (KeyError, ValueError, ImportError) as exc:
            return Command(update={"messages": [
                ToolMessage(f"error: {exc}", tool_call_id=tool_call_id)
            ]})

        # Same output shape as profile_dataset: the redacted report as pure JSON. The
        # sample's provenance (sampled flag + full-table estimate) lives in the report
        # itself, so no prose note is needed and every consumer parses one shape.
        report = profiler.redact(profiler.profile(
            load.df, dataset=table, sampled=load.sampled,
            estimated_total_rows=load.estimated_rows if load.sampled else None,
        ))
        return Command(update={
            "profile": report.model_dump(mode="json"),
            "source_table": table,
            "draft": None,
            "messages": [ToolMessage(report.model_dump_json(exclude_none=True),
                                     tool_call_id=tool_call_id)],
        })

    @tool
    def list_rules(tags: list[str] | None = None) -> str:
        """List the Great Expectations expectations a contract may use, optionally
        filtered by tags (e.g. completeness, uniqueness, validity, conformance,
        freshness, volume). Returns each expectation_type with its description, tags
        and kwargs."""
        lines = []
        for rule_id in registry.rule_ids:
            rule = registry.get(rule_id)
            if tags and not set(tags) & set(rule.tags):
                continue
            kwargs = "\n".join(
                f"    {name}: {spec.type}{' (required)' if spec.required else ''}"
                + (f" — {spec.description}" if spec.description else "")
                for name, spec in rule.kwargs.items()
            )
            lines.append(
                f"- {rule.id} [{', '.join(rule.tags)}]\n"
                f"  {rule.description}\n"
                f"  kwargs:\n{kwargs}"
            )
        return "\n".join(lines) or "no rules match those tags"

    @tool
    def propose_contract(
        expectations: list[ProposedExpectation],
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        table: str | None = None,
    ) -> Command:
        """Propose a draft contract for the profiled dataset: the Great Expectations
        expectations you recommend, each with its expectation_type, kwargs and an
        optional meta note for the monitoring dashboard. The draft is validated against
        the data-flow allowlist and shown to the owner for discussion — it is not
        approved or executed by this call.

        `table` is the schema-qualified Postgres table the contract will run against
        ('schema.table'). Omit it when you profiled that table with profile_table;
        for a local file, ask the owner which table it is and pass it.

        Each `kwargs` must be a JSON object (dict), not a string. Example:
          {"expectation_type": "expect_column_values_to_not_be_null",
           "kwargs": {"column": "email"},
           "meta": {"notes": {"format": "string", "content": "Every customer needs an email"}}}
          {"expectation_type": "expect_column_values_to_be_between",
           "kwargs": {"column": "amount", "min_value": 0}}
        """
        report = state.get("profile")
        if report is None:
            return Command(update={"messages": [
                ToolMessage(
                    "error: profile the dataset first", tool_call_id=tool_call_id
                )
            ]})

        target = table or state.get("source_table")
        if not target or target.count(".") != 1:
            return Command(update={"messages": [ToolMessage(
                "error: which Postgres table will this contract run against? Ask the "
                "owner and pass it as table='schema.table'",
                tool_call_id=tool_call_id,
            )]})

        schema, name = target.split(".")
        draft = Contract.draft(
            schema, name, [Expectation(**e.model_dump()) for e in expectations]
        )
        errors = draft.problems()
        if errors:
            return Command(update={"messages": [
                ToolMessage("invalid proposal: " + "; ".join(errors),
                            tool_call_id=tool_call_id)
            ]})

        message = ("draft recorded. Summarize it for the owner in plain English:\n"
                   + describe_contract(draft, registry))
        # A sampled profile's min/max are estimates, so confirm any proposed range bound
        # against the live table — but only then, and only for the referenced columns.
        if report.get("sampled") and state.get("source_table"):
            notes = _confirm_range_bounds(draft.suite.expectations, state["source_table"])
            if notes:
                message += "\n\nConfirmed range bounds against the full table:\n- " + "\n- ".join(notes)
        return Command(update={
            "draft": draft.to_dict(),
            "messages": [ToolMessage(message, tool_call_id=tool_call_id)],
        })

    @tool
    def request_approval() -> None:
        """Send the current draft contract to the human approval gate. Call this
        immediately after propose_contract succeeds — do not wait for a conversational
        confirmation first."""
        # never executed: the graph routes this call to the approval node

    return [profile_dataset, profile_table, list_rules, propose_contract, request_approval]


def _confirm_range_bounds(expectations: list[Expectation], table: str) -> list[str]:
    """Confirm proposed range bounds against the full table and flag any that
    existing rows would already violate. Called only when the profile is a *sample*
    (observed min/max are estimates) and the source is a Postgres table, so it runs an
    exact `min/max` query solely for the columns a range expectation references — never
    speculatively, and at most once per column. Returns human-readable notes, empty
    when every bound checks out."""
    range_rules = [e for e in expectations
                   if e.expectation_type == "expect_column_values_to_be_between"]
    if not range_rules:
        return []
    try:
        uri = connectors.resolve_dsn()
    except (KeyError, ImportError) as exc:
        return [f"could not confirm range bounds against the table ({exc})"]

    notes: list[str] = []
    seen: dict[str, tuple] = {}
    for rule in range_rules:
        col = rule.kwargs.get("column")
        if not col:
            continue
        try:
            if col not in seen:
                seen[col] = connectors.column_bounds(uri, table, col)
            lo, hi = seen[col]
            problems = []
            proposed_min, proposed_max = rule.kwargs.get("min_value"), rule.kwargs.get("max_value")
            if proposed_min is not None and lo is not None and lo < proposed_min:
                problems.append(f"actual min {lo} is below the proposed min_value {proposed_min}")
            if proposed_max is not None and hi is not None and hi > proposed_max:
                problems.append(f"actual max {hi} is above the proposed max_value {proposed_max}")
        except (ValueError, ImportError, TypeError) as exc:
            notes.append(f"range on '{col}': could not confirm bounds ({exc})")
            continue
        if problems:
            notes.append(
                f"range on '{col}': " + "; ".join(problems)
                + " — existing rows would violate this bound; confirm the real limit with the owner."
            )
    return notes


_ACCEPT_WORDS = {"accept", "approve", "approved", "yes", "y", "looks good", "go ahead"}


def _decision(response: Any) -> dict[str, Any]:
    """Normalize a resume payload into a single decision dict.

    The driver resumes with {"decisions": [decision]} where decision["type"] is
    approve/edit/reject; a bare list or dict is accepted too. A free-text string —
    a CLI that lets the owner type instead of clicking a button — becomes an approve
    or reject decision so the text and widget paths converge."""
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
        # HITL interrupt contract: the plural action_requests + review_configs shape.
        # The local driver reads action_requests[0].description to show the owner and
        # offers the allowed_decisions (approve/edit/reject) as its resume options.
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
            try:
                contract = Contract.model_validate(edited.get("args", {}).get("contract", {}))
            except ValueError as exc:
                return respond(f"owner's edited contract is malformed, fix and re-propose: {exc}")
            errors = contract.problems()
            if errors:
                return respond("owner's edited contract is invalid, fix and re-propose: "
                               + "; ".join(errors))
            kind = "approve"

        if kind == "approve":
            contract.approve(_approver(decision.get("args")))
            # the data-flow layout: one contract per table, <schema>.<table>.yaml
            path = dq_validation.contract_path(
                contract.dataset.schema_, contract.dataset.table, root=contracts_dir
            )
            dq_validation.validate_contract(contract.to_dict())  # exactly what data-flow loads
            contracts_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(contract.to_yaml(), encoding="utf-8")
            return respond(
                f"contract approved by {contract.approval.approved_by} and saved to {path}",
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
            model_id=os.environ.get("DQ_AGENT_MODEL", DEFAULT_MODEL_ID),
            # Prompt caching is on by default; DQ_AGENT_PROMPT_CACHE=0 turns it
            # off, which is the escape hatch if the proxy ever rejects the
            # cachePoint blocks.
            cache_prompt=os.environ.get("DQ_AGENT_PROMPT_CACHE", "1") != "0",
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
