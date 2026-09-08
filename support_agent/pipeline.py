"""Intake gate: classify first, and only send the ticket to triage if it should go.

Measurement produced a specific defect. On "notifications aren't going out" the
classifier was correct -- it returned `function_name: null` and
`needs_telemetry: false` -- and the triage agent, run on the raw ticket, guessed
`notify-fanout` and pulled its metrics anyway.

The instruction telling triage not to guess was already there. It lost to the
fact that triage has a tool, a plausible candidate, and a customer who wants an
answer. So the decision moves out of the agent that has the incentive to guess
and into the stage that has already made it correctly.

This is the same lesson as `policy.py`, one level up: the reliable place to put
a constraint is the place the model cannot reach.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .classifier import Triage


@dataclass
class Decision:
    """Whether a ticket should reach triage, and why not if it shouldn't."""

    triage: Triage | None
    proceed: bool
    reply: str | None = None


# Categories that no amount of telemetry will answer.
_NO_TELEMETRY = {"how_to_or_guidance", "insufficient_information"}

_ASK_FOR_RESOURCE = (
    "This ticket does not name a specific resource, so there is nothing to look "
    "up yet. Which function or service is affected? Naming it will let us pull "
    "its metrics directly."
)

_ASK_FOR_RESOURCE_JA = (
    "対象のリソース名が記載されていないため、まだ調査を開始できません。"
    "どの関数・サービスで発生していますか。名前が分かればメトリクスを直接確認できます。"
)

_GUIDANCE = (
    "This is a how-to question rather than an incident report. It should be "
    "answered from the user guide, not from the customer's telemetry."
)


def _is_japanese(text: str) -> bool:
    return any("぀" <= ch <= "ヿ" for ch in text)


def decide(ticket: str, classifier_output: str | dict) -> Decision:
    """Turn a classifier result into a routing decision.

    Args:
        ticket: The original ticket text, used only to pick a reply language.
        classifier_output: The classifier's JSON string, or an already-parsed dict.

    Returns:
        A Decision. When `proceed` is False, `reply` is what the customer
        should get instead of a diagnosis.
    """
    if isinstance(classifier_output, str):
        classifier_output = json.loads(classifier_output)
    triage = Triage.model_validate(classifier_output)

    if triage.category.value in _NO_TELEMETRY or not triage.needs_telemetry:
        if triage.category.value == "how_to_or_guidance":
            return Decision(triage, proceed=False, reply=_GUIDANCE)
        reply = _ASK_FOR_RESOURCE_JA if _is_japanese(ticket) else _ASK_FOR_RESOURCE
        return Decision(triage, proceed=False, reply=reply)

    if not triage.function_name:
        # Belt and braces: needs_telemetry true but nothing to look up.
        reply = _ASK_FOR_RESOURCE_JA if _is_japanese(ticket) else _ASK_FOR_RESOURCE
        return Decision(triage, proceed=False, reply=reply)

    return Decision(triage, proceed=True)
