"""Intake classification: route the ticket before spending telemetry calls on it.

Separate from triage on purpose. Classification reads only the ticket text, so
it is cheap, it has no tools to misuse, and its output is a fixed enum that can
be scored against labels without a judge model. Triage is the expensive stage
and only some categories are worth sending there.

Splitting the two also means a wrong route is visible as a wrong route, rather
than surfacing later as a confidently wrong diagnosis.
"""

from __future__ import annotations

import os
from enum import Enum

from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field

MODEL = os.environ.get("SUPPORT_AGENT_CLASSIFIER_MODEL", "gemini-3.8-flash")


class Category(str, Enum):
    """What kind of ticket this is, decided from the text alone."""

    PERFORMANCE = "performance_degradation"
    ERRORS = "errors_or_failures"
    PERMISSION = "permission_or_access"
    HOW_TO = "how_to_or_guidance"
    INSUFFICIENT_INFO = "insufficient_information"


class Triage(BaseModel):
    """Structured intake decision for one ticket."""

    category: Category = Field(description="The single best-fitting category.")
    function_name: str | None = Field(
        default=None,
        description=(
            "The resource the ticket is about, exactly as the customer wrote it. "
            "Null if the ticket does not name one -- do not infer it."
        ),
    )
    needs_telemetry: bool = Field(
        description=(
            "Whether answering requires looking at the customer's live telemetry. "
            "False for how-to questions and for tickets too vague to act on."
        )
    )
    urgency: int = Field(
        ge=1, le=4, description="1 = highest, 4 = lowest. Judge from the text only."
    )
    rationale: str = Field(
        description="One sentence, citing the words in the ticket that decided it."
    )


CLASSIFIER_INSTRUCTION = """\
You classify inbound cloud support tickets at intake, before anyone has looked
at the customer's systems.

You are working from the ticket text and nothing else. That is the whole point:
you are deciding where this ticket should go, not what is wrong with it.

- `function_name` is only for a resource the customer actually named. If they
  described a symptom without naming a resource ("notifications aren't going
  out"), leave it null and classify as insufficient_information. Inferring
  which system they probably meant is the single most expensive mistake
  available to you here, because everything downstream will trust it.
- `urgency` reflects what the customer is claiming, not what is true. You have
  not checked anything yet.
- Do not diagnose. "Probably a timeout" is not a category.
"""

classifier_agent = LlmAgent(
    name="intake_classifier",
    model=MODEL,
    description="Classifies an inbound support ticket from its text alone.",
    instruction=CLASSIFIER_INSTRUCTION,
    output_schema=Triage,
    output_key="triage",
)
