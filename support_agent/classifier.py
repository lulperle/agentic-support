"""Intake classification: route the ticket before spending telemetry calls on it.

Separate from triage on purpose. Classification reads only the ticket text, so
it is cheap, it has no tools to misuse, and its output is a fixed enum that can
be scored against labels without a judge model. Triage is the expensive stage
and only some categories are worth sending there.

Splitting the two also means a wrong route is visible as a wrong route, rather
than surfacing later as a confidently wrong diagnosis.
"""

from __future__ import annotations

from enum import Enum

from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field

from .models import intake_model


class UrgencyClaim(str, Enum):
    """How fast the customer says they need it.

    An enum rather than a 1-4 integer, for two reasons. Bedrock's structured
    output rejects `minimum`/`maximum` on an integer field, which Gemini
    accepted -- so a constrained int is not portable across the providers a
    government desk actually runs on. And the values carry their own meaning,
    where "2" needs a lookup table to read.

    Values are kept identical to `sla.Urgency` so the AI stage and the policy
    stage need no translation between them; a test asserts they stay in sync.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


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
            "Whether telemetry can be looked up *right now*, from this ticket "
            "alone. False if no resource is named -- there is nothing to query "
            "yet, however useful telemetry would be once the customer answers. "
            "False for how-to questions, which no telemetry can answer."
        )
    )
    urgency: UrgencyClaim = Field(
        description="What the customer is claiming, judged from the text only."
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
- Categories overlap, so prefer the most specific one the ticket supports. A
  permission denial is also a failure; if the ticket indicates a denial, refused
  access or an authorisation problem, it is `permission_or_access` rather than
  the broader `errors_or_failures`. Use `errors_or_failures` when the ticket
  reports something breaking without indicating what kind of break it is.
- `needs_telemetry` asks whether there is something to query *now*, not whether
  telemetry would eventually help. No resource named means nothing to query, so
  it is false. If `function_name` is null, `needs_telemetry` is false.
- `urgency` reflects what the customer is claiming, not what is true. You have
  not checked anything yet.
- Do not diagnose. "Probably a timeout" is not a category.
"""

classifier_agent = LlmAgent(
    name="intake_classifier",
    model=intake_model(),
    description="Classifies an inbound support ticket from its text alone.",
    instruction=CLASSIFIER_INSTRUCTION,
    output_schema=Triage,
    output_key="triage",
)
