"""Rule functions: one module per data-quality category.

Each rule is a pure function — Polars DataFrame in, RuleResult out, no I/O, no LLM.
Rules assume a non-empty DataFrame: the engine is the single gatekeeper for emptiness
(a column rule on zero rows would divide by zero), so functions need not guard against
it themselves. See engine.run() for the pre-flight checks every rule relies on.
"""
