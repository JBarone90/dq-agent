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
data-flow Airflow task runs it.

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

import polars as pl
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Command, interrupt
from pydantic import field_validator

from dq_agent import connectors, gx_contract, profiler, proposals
from dq_agent.agents.gate import unwrap_decision
from dq_agent.models import Contract, Expectation
from dq_agent.registry import Registry, RuleDefinition
from dq_agent.report import describe_contract

DEFAULT_REGISTRY_FILE = Path("registry/expectations.yaml")
DEFAULT_CONTRACTS_DIR = Path("contracts")
# profile_dataset reads only files under this directory: the path comes from the LLM
DEFAULT_DATA_DIR = Path("data")

# What loading a dataset can raise for reasons outside the agent's control (a missing table,
# a refused connection, a malformed file). Reported back to the agent as a tool error, so the
# conversation carries on; ToolNode would otherwise re-raise and end the turn.
# connectorx reports database errors as RuntimeError.
_LOAD_ERRORS = (KeyError, ValueError, ImportError, OSError, RuntimeError, pl.exceptions.PolarsError)

# Range expectations whose bounds the propose step confirms against the live table
_RANGE_EXPECTATIONS = (
    "expect_column_values_to_be_between",
    "expect_column_min_to_be_between",
    "expect_column_max_to_be_between",
)

# Deterministic profiling cap: the code, not the LLM, decides how much to pull. Tables
# under this load whole; larger ones come back as a block-level sample (flagged sampled).
PROFILE_MAX_ROWS = 200_000

SYSTEM_PROMPT = (Path(__file__).parent / "scoping_prompt.txt").read_text(encoding="utf-8")


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


def _make_tools(
    registry: Registry,
    proposals_dir: Path = proposals.DEFAULT_PROPOSALS_DIR,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> list:
    @tool
    def profile_dataset(
        path: str, tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
        """Profile a local CSV or Parquet dataset in the data directory. Returns a redacted
        statistical report: column types, null rates, uniqueness, distributions, semantic
        hints. Raw value examples (top values) are stripped; numeric/temporal min and max
        are disclosed as bounded aggregates."""
        file = Path(path).resolve()
        if not file.is_relative_to(data_dir.resolve()):
            return Command(update={"messages": [ToolMessage(
                f"error: only files under the data directory '{data_dir}' can be profiled",
                tool_call_id=tool_call_id,
            )]})
        if not file.is_file():
            return Command(update={"messages": [
                ToolMessage(f"error: no file at '{path}'", tool_call_id=tool_call_id)
            ]})
        loaders = {".csv": connectors.load_csv, ".parquet": connectors.load_parquet}
        if file.suffix not in loaders:
            return Command(update={"messages": [
                ToolMessage(
                    f"error: unsupported file type '{file.suffix}' (csv or parquet only)",
                    tool_call_id=tool_call_id,
                )
            ]})
        try:
            df = loaders[file.suffix](file)
        except _LOAD_ERRORS as exc:
            return Command(update={"messages": [
                ToolMessage(f"error: could not read '{path}': {exc}", tool_call_id=tool_call_id)
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
        except _LOAD_ERRORS as exc:
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
        freshness, volume). Returns each expectation_type with its description, usage
        notes and the kwargs it accepts, as documented by Great Expectations."""
        lines = []
        for rule_id in registry.rule_ids:
            rule = registry.get(rule_id)
            if tags and not set(tags) & set(rule.tags):
                continue
            lines.append(_format_rule(rule))
        return "\n".join(lines) or "no rules match those tags"

    @tool
    def propose_contract(
        expectations: list[ProposedExpectation],
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        table: str | None = None,
    ) -> Command:
        """Propose a draft contract for the profiled dataset: the Great Expectations
        expectations you recommend, each with its expectation_type, kwargs and a meta
        with its severity and a note for the monitoring dashboard. The draft is validated against
        the data-flow allowlist and shown to the owner for discussion — it is not
        approved or executed by this call.

        `table` is the schema-qualified Postgres table the contract will run against
        ('schema.table'). Omit it when you profiled that table with profile_table;
        for a local file, ask the owner which table it is and pass it.

        Each `kwargs` must be a JSON object (dict), not a string. Example:
          {"expectation_type": "expect_column_values_to_not_be_null",
           "kwargs": {"column": "email"},
           "meta": {"severity": "error",
                    "notes": {"format": "string", "content": "Every customer needs an email"}}}
          {"expectation_type": "expect_column_values_to_be_between",
           "kwargs": {"column": "amount", "min_value": 0},
           "meta": {"severity": "warning"}}
        """
        report = state.get("profile")
        if report is None:
            return Command(update={"messages": [
                ToolMessage(
                    "error: profile the dataset first", tool_call_id=tool_call_id
                )
            ]})

        target = _target_table(table, state)
        if target is None:
            return Command(update={"messages": [ToolMessage(
                _NO_TARGET, tool_call_id=tool_call_id
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

    @tool
    def list_unlisted_expectations(expectation_type: str | None = None) -> str:
        """Creative mode, step 1. Use only when the owner needs a check that no
        expectation from list_rules can express. Without arguments, lists every Great
        Expectations expectation that is NOT allowed in contracts yet, with a one-line
        description and whether it is known to run on Postgres. Pass one
        expectation_type to see its kwargs, as documented by GX."""
        unlisted = {u.expectation_type: u for u in proposals.unlisted_expectations()}
        if expectation_type is None:
            return "\n".join(
                f"- {u.expectation_type} — {u.description} (Postgres: {u.sql_support})"
                for u in unlisted.values()
            )
        if expectation_type not in unlisted:
            return (f"error: '{expectation_type}' is not an unlisted GX expectation "
                    "(it is either already allowed — see list_rules — or does not exist)")
        rule = registry.describe_unlisted(expectation_type)
        return (_format_rule(rule)
                + f"\n  Postgres: {unlisted[expectation_type].sql_support}")

    @tool
    def propose_new_expectation(
        expectation_type: str,
        rationale: str,
        example_kwargs: dict[str, Any],
        example_pass: str,
        example_fail: str,
        state: Annotated[dict, InjectedState],
        table: str | None = None,
    ) -> str:
        """Creative mode, step 2. Record a proposal for a developer to add an unlisted
        Great Expectations expectation to the allowlist. It is NOT added to the
        contract and never runs until a developer promotes it.

        rationale: the owner's need in plain English. example_kwargs: the kwargs this
        table would use (validated against GX). example_pass / example_fail: short
        invented illustrations of a passing and a failing value — never real data.
        table: 'schema.table', omitted when the dataset was profiled with profile_table.
        """
        target = _target_table(table, state)
        if target is None:
            return _NO_TARGET
        errors = proposals.proposal_errors(expectation_type, example_kwargs)
        if errors:
            return "invalid proposal: " + "; ".join(errors)
        path = proposals.write_proposal(proposals.new_proposal(
            dataset=target,
            expectation_type=expectation_type,
            rationale=rationale,
            example_kwargs=example_kwargs,
            example_pass=example_pass,
            example_fail=example_fail,
        ), proposals_dir)
        return (f"proposal saved to {path} for developer review. Tell the owner this check "
                "is not part of the contract: it will be available once a developer "
                "promotes it. Continue with the contract as it is.")

    return [profile_dataset, profile_table, list_rules, propose_contract,
            list_unlisted_expectations, propose_new_expectation, request_approval]


_NO_TARGET = ("error: which Postgres table is this for? Ask the owner and pass it as "
              "table='schema.table'")


def _target_table(table: str | None, state: dict) -> str | None:
    """The contract's Postgres table: given explicitly, else the one that was profiled."""
    target = table or state.get("source_table")
    return target if target and target.count(".") == 1 else None


def _format_rule(rule: RuleDefinition) -> str:
    kwargs = "\n".join(
        f"    {name}"
        + (f" ({spec.type})" if spec.type else "")
        + (" [required]" if name in rule.required else "")
        + (f" — {spec.description}" if spec.description else "")
        + (f" Default: {spec.default!r}."
           if spec.default is not None and name not in rule.required
           and "default" not in (spec.description or "").lower() else "")
        for name, spec in rule.kwargs.items()
    )
    return (f"- {rule.id} [{', '.join(rule.tags)}]\n"
            f"  {rule.description}\n"
            + (f"  Notes: {rule.notes}\n" if rule.notes else "")
            + f"  kwargs:\n{kwargs}")


def _confirm_range_bounds(expectations: list[Expectation], table: str) -> list[str]:
    """Confirm proposed range bounds against the full table and flag any that
    existing rows would already violate. Called only when the profile is a *sample*
    (observed min/max are estimates) and the source is a Postgres table, so it runs an
    exact `min/max` query solely for the columns a range expectation references — never
    speculatively, and at most once per column. Returns human-readable notes, empty
    when every bound checks out."""
    range_rules = [e for e in expectations if e.expectation_type in _RANGE_EXPECTATIONS]
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
        # a {"$PARAMETER": ...} bound (freshness, stale rows) is evaluated by GX at run time: nothing to compare
        proposed_min, proposed_max = (
            None if isinstance(value, dict) else value
            for value in (rule.kwargs.get("min_value"), rule.kwargs.get("max_value"))
        )
        if not col or (proposed_min is None and proposed_max is None):
            continue
        try:
            if col not in seen:
                seen[col] = connectors.column_bounds(uri, table, col)
            lo, hi = seen[col]
            # which observed value each bound constrains: the column's min, its max, or every value
            low_side, high_side = {
                "expect_column_min_to_be_between": (("min", lo), ("min", lo)),
                "expect_column_max_to_be_between": (("max", hi), ("max", hi)),
            }.get(rule.expectation_type, (("min", lo), ("max", hi)))
            problems = []
            if proposed_min is not None and low_side[1] is not None and low_side[1] < proposed_min:
                problems.append(f"actual {low_side[0]} {low_side[1]} is below the proposed min_value {proposed_min}")
            if proposed_max is not None and high_side[1] is not None and high_side[1] > proposed_max:
                problems.append(f"actual {high_side[0]} {high_side[1]} is above the proposed max_value {proposed_max}")
        except (*_LOAD_ERRORS, TypeError) as exc:
            notes.append(f"range on '{col}': could not confirm bounds ({exc})")
            continue
        if problems:
            notes.append(
                f"range on '{col}': " + "; ".join(problems)
                + " — existing rows would violate this bound; confirm the real limit with the owner."
            )
    return notes


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
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)  # route() sends only an AI tool call here
        pending = last.tool_calls
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
        existing = gx_contract.contract_path(
            contract.dataset.schema_, contract.dataset.table, root=contracts_dir
        )
        # The UI markdown-renders this, so the YAML goes in a fenced block to keep its
        # indentation; describe_contract already emits a markdown bullet list.
        description = ("Review the proposed data quality contract:\n\n"
                       + describe_contract(contract, registry)
                       + "\n\nFull definition:\n\n```yaml\n" + contract.to_yaml() + "```")
        if existing.exists():
            description += (f"\n\n**Approving replaces the existing contract** at `{existing}`.")
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

        decision = unwrap_decision(response)
        kind = decision.get("type")

        if kind == "edit":
            edited = decision.get("edited_action") or {}
            try:
                contract = Contract.model_validate(edited.get("args", {}).get("contract", {}))
            except ValueError as exc:
                return respond(f"owner's edited contract is malformed, fix and re-propose: {exc}")
            if contract.dataset.name != Contract.model_validate(draft).dataset.name:
                # the card the owner reviewed was for the draft's table
                return respond("owner's edited contract targets a different table than the "
                               "draft; re-propose it for that table instead")
            errors = contract.problems()
            if errors:
                return respond("owner's edited contract is invalid, fix and re-propose: "
                               + "; ".join(errors))
            kind = "approve"

        if kind == "approve":
            approval = contract.approve(_approver(decision.get("args")))
            # the data-flow layout: one contract per table, <schema>.<table>.yaml
            path = gx_contract.contract_path(
                contract.dataset.schema_, contract.dataset.table, root=contracts_dir
            )
            gx_contract.validate_contract(contract.to_dict())  # exactly what data-flow loads
            replaced = path.exists()
            contracts_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(contract.to_yaml(), encoding="utf-8")
            return respond(
                f"contract approved by {approval.approved_by} and saved to {path}"
                + (" (replacing the previous contract)" if replaced else ""),
                draft=None,
                contract_path=str(path),
            )

        # reject / anything non-affirmative: hand the feedback back to the agent so it
        # can iterate on the contract rather than silently dropping the session.
        return respond(f"owner did not approve; feedback: {decision.get('message') or 'none given'}")

    return approval


def build_graph(
    *,
    model: BaseChatModel | None = None,
    registry: Registry | None = None,
    contracts_dir: Path | str = DEFAULT_CONTRACTS_DIR,
    proposals_dir: Path | str = proposals.DEFAULT_PROPOSALS_DIR,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    checkpointer: Any = None,
):
    """Compile the scoping graph. All collaborators are injectable for tests;
    defaults serve the localhost demo (model from $DQ_AGENT_MODEL, repo-root
    registry, contracts, proposals and data directories)."""
    if model is None:
        # The work environment is air-gapped and reaches Bedrock only through the
        # internal bedrock-proxy, so the default model is DeptBedrockChat rather than
        # init_chat_model. Inject `model` to run elsewhere (the tests do).
        from dq_agent.agents.bedrock_chat import DEFAULT_MODEL_ID, DeptBedrockChat

        model = DeptBedrockChat(
            model_id=os.environ.get("DQ_AGENT_MODEL", DEFAULT_MODEL_ID),
            # Prompt caching is on by default; DQ_AGENT_PROMPT_CACHE=0 turns it
            # off, which is the escape hatch if the proxy ever rejects the
            # cachePoint blocks.
            cache_prompt=os.environ.get("DQ_AGENT_PROMPT_CACHE", "1") != "0",
        )
    if registry is None:
        registry = Registry(DEFAULT_REGISTRY_FILE)

    tools = _make_tools(registry, Path(proposals_dir), Path(data_dir))
    bound = model.bind_tools(tools)

    def agent(state: ScopingState) -> dict:
        messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [bound.invoke(messages)]}

    def route(state: ScopingState) -> str:
        last = state["messages"][-1]
        calls = last.tool_calls if isinstance(last, AIMessage) else []
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
