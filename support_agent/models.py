"""Model selection, kept in one place so the agents don't name a provider.

The agents originally hardcoded a Gemini model string. That was fine until the
evaluation suite existed, at which point the free tier's 20 requests per day
became the binding constraint on how often the suite could run -- a measurement
harness you can only run once a day is barely a harness.

So the provider is now a choice. It also happens to be the honest shape for this
problem: a government cloud desk runs on more than one provider, and an agent
that only works against one of them is a demo.

    SUPPORT_AGENT_BACKEND=bedrock   # default when AWS credentials are present
    SUPPORT_AGENT_BACKEND=gemini
    SUPPORT_AGENT_MODEL=<explicit model string>   # overrides both

Triage and intake are configured separately because they are different jobs:
intake is a short fixed-output classification, triage is a multi-turn tool loop.
"""

from __future__ import annotations

import os

# Anthropic model ids on Bedrock. The `us.` prefix is a cross-region inference
# profile rather than a plain model id: without it these ids are not invokable
# in most regions.
BEDROCK_TRIAGE = "bedrock/converse/us.anthropic.claude-sonnet-5"
BEDROCK_INTAKE = "bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0"

GEMINI_TRIAGE = "gemini-3.8-flash"
GEMINI_INTAKE = "gemini-3.8-flash"


def _backend() -> str:
    explicit = os.environ.get("SUPPORT_AGENT_BACKEND")
    if explicit:
        return explicit.lower()
    # Credentials being present is a better default signal than a hardcoded
    # provider, and it keeps CI -- which has neither -- from silently
    # attempting real calls.
    if os.environ.get("AWS_PROFILE") or os.environ.get("AWS_REGION"):
        return "bedrock"
    return "gemini"


def _resolve(bedrock: str, gemini: str, override: str | None) -> object:
    if override:
        name = override
    elif _backend() == "bedrock":
        name = bedrock
    else:
        name = gemini

    if name.startswith("bedrock/"):
        # Imported lazily: litellm is a heavy import and is not needed at all
        # when running against Gemini.
        from google.adk.models.lite_llm import LiteLlm

        return LiteLlm(model=name)
    return name


def triage_model() -> object:
    """The model for the tool-using triage agent."""
    return _resolve(BEDROCK_TRIAGE, GEMINI_TRIAGE, os.environ.get("SUPPORT_AGENT_MODEL"))


def intake_model() -> object:
    """The model for the tool-free intake classifier."""
    return _resolve(
        BEDROCK_INTAKE,
        GEMINI_INTAKE,
        os.environ.get("SUPPORT_AGENT_CLASSIFIER_MODEL"),
    )


def describe() -> str:
    """One line naming what will actually be called.

    Worth its own function because of a mistake this cost: a stale
    `SUPPORT_AGENT_MODEL` left in `.env` silently overrode the backend, and a
    full suite run went to the wrong provider and died on its free-tier limit.
    The run had said nothing about which model it was using. A measurement that
    does not record what was measured is not a measurement.
    """
    return f"backend={_backend()} triage={_name(triage_model())} intake={_name(intake_model())}"


def _name(model: object) -> str:
    return model if isinstance(model, str) else getattr(model, "model", str(model))
