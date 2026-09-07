"""Run one ticket through the agent from the command line.

    uv run python -m scripts.ask "checkout-api is slow this morning"

Prints every tool call the agent made, then the final answer, then the
token cost of the run. The tool trace is the point: an agent you cannot
see the intermediate steps of is an agent you cannot debug in front of
a customer.
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv
from google.genai import types

load_dotenv("support_agent/.env")

from google.adk.runners import InMemoryRunner  # noqa: E402  (needs env first)

from support_agent.agent import root_agent  # noqa: E402

APP_NAME = "agentic-support"
USER_ID = "workshop"


async def ask(ticket: str) -> None:
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID
    )

    message = types.Content(role="user", parts=[types.Part(text=ticket)])
    prompt_tokens = response_tokens = 0
    final_text = ""

    async for event in runner.run_async(
        user_id=USER_ID, session_id=session.id, new_message=message
    ):
        if event.usage_metadata:
            prompt_tokens += event.usage_metadata.prompt_token_count or 0
            response_tokens += event.usage_metadata.candidates_token_count or 0

        for part in (event.content.parts if event.content else []) or []:
            if part.function_call:
                print(f"  -> tool: {part.function_call.name}({dict(part.function_call.args)})")
            elif part.function_response:
                print(f"  <- result: {part.function_response.name} returned")

        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(p.text or "" for p in event.content.parts)

    print(f"\n{final_text.strip()}\n")
    print(f"[tokens] prompt={prompt_tokens} response={response_tokens}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: uv run python -m scripts.ask '<ticket text>'")
    asyncio.run(ask(" ".join(sys.argv[1:])))
