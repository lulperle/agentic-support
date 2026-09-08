"""Run one ticket through the agent from the command line.

    uv run python -m scripts.ask "checkout-api is slow this morning"

Prints the intake decision, every tool call the agent made, the final answer, and
the token cost. The trace is the point: an agent you cannot see the intermediate
steps of is an agent you cannot debug in front of a customer.

Both stages run, gate included. This script used to call triage directly on the
raw ticket, which is the same mistake the eval harness made -- it demonstrated a
stage that production never reaches unaided, so the tickets the gate is supposed
to stop got a diagnosis here and a clarifying question in the product.
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv
from google.genai import types

load_dotenv("support_agent/.env")

from google.adk.runners import InMemoryRunner  # noqa: E402  (needs env first)

from support_agent.agent import root_agent  # noqa: E402
from support_agent.classifier import classifier_agent  # noqa: E402
from support_agent.models import describe as describe_models  # noqa: E402
from support_agent.pipeline import decide, triage_prompt  # noqa: E402

APP_NAME = "agentic-support"
USER_ID = "workshop"


async def _run(agent, text: str, *, trace: bool) -> tuple[str, int, int]:
    """Run one agent to completion. Returns (final text, prompt, response tokens)."""
    runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID
    )

    message = types.Content(role="user", parts=[types.Part(text=text)])
    prompt_tokens = response_tokens = 0
    final_text = ""

    async for event in runner.run_async(
        user_id=USER_ID, session_id=session.id, new_message=message
    ):
        if event.usage_metadata:
            prompt_tokens += event.usage_metadata.prompt_token_count or 0
            response_tokens += event.usage_metadata.candidates_token_count or 0

        for part in (event.content.parts if event.content else []) or []:
            if not trace:
                continue
            if part.function_call:
                print(f"  -> tool: {part.function_call.name}({dict(part.function_call.args)})")
            elif part.function_response:
                print(f"  <- result: {part.function_response.name} returned")

        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(p.text or "" for p in event.content.parts)

    return final_text.strip(), prompt_tokens, response_tokens


async def ask(ticket: str) -> None:
    print(f"[models] {describe_models()}")

    intake, prompt_tokens, response_tokens = await _run(
        classifier_agent, ticket, trace=False
    )
    decision = decide(ticket, intake)
    triage = decision.triage
    print(
        f"  intake: {triage.category.value}, resource={triage.function_name!r}, "
        f"urgency={triage.urgency.value}"
    )

    if not decision.proceed:
        print("  gated at intake -- triage not run\n")
        answer = decision.reply or ""
    else:
        answer, p, r = await _run(root_agent, triage_prompt(ticket), trace=True)
        prompt_tokens += p
        response_tokens += r

    print(f"\n{answer}\n")
    print(f"[tokens] prompt={prompt_tokens} response={response_tokens}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: uv run python -m scripts.ask '<ticket text>'")
    asyncio.run(ask(" ".join(sys.argv[1:])))
