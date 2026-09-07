"""Session 0: one agent, two tools, real tool-calling loop.

The agent triages an inbound support ticket. It is deliberately the
smallest thing that exercises the whole path -- model, tool schema,
tool result, grounded answer -- so that later sessions can add
delegation, retrieval and evaluation on top of a base that already works.
"""

from __future__ import annotations

import os

from google.adk.agents import LlmAgent

from .mock_infra import get_function_metrics, get_recent_logs

MODEL = os.environ.get("SUPPORT_AGENT_MODEL", "gemini-3.8-flash")

TRIAGE_INSTRUCTION = """\
You are a cloud support triage engineer. You are given a customer's
support ticket, usually vague and written in a hurry.

Your job:
1. Identify which function the customer is talking about. If the ticket
   does not name one, say so and ask -- do not guess.
2. Call `get_function_metrics` to see the shape of the problem.
3. Call `get_recent_logs` only if the metrics suggest something is wrong.
4. State a probable root cause, and label your confidence as
   high / medium / low.

Rules you must not break:
- Every factual claim about the customer's system must come from a tool
  result. If a tool did not tell you something, you do not know it.
- If the tools contradict the customer's description, report the
  contradiction rather than smoothing it over.
- Answer in the language the ticket was written in.
"""

root_agent = LlmAgent(
    name="triage_agent",
    model=MODEL,
    description="Triages an inbound cloud support ticket against live telemetry.",
    instruction=TRIAGE_INSTRUCTION,
    tools=[get_function_metrics, get_recent_logs],
)
