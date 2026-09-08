"""Score the agent against labelled tickets and print the numbers.

    uv run python -m evals.run
    uv run python -m evals.run --only perf-en-01 healthy-en-03
    uv run python -m evals.run --json results.json

Two stages are scored separately, because they fail differently. The classifier
either routes a ticket correctly or it doesn't, which is a label comparison.
Triage is scored on behaviour: which tools it reached for, whether it invented
anything, whether it abstained when it should have, and what the run cost.

No judge model is used. Every check here is a deterministic comparison against
an expectation written down in tickets.yaml before the run, so a regression is
a number moving rather than an opinion changing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv
from google.genai import types

load_dotenv("support_agent/.env")

from google.adk.runners import InMemoryRunner  # noqa: E402  (needs env first)

from support_agent.agent import root_agent  # noqa: E402
from support_agent.classifier import classifier_agent  # noqa: E402

TICKETS = Path(__file__).parent / "tickets.yaml"
APP_NAME = "agentic-support-evals"
USER_ID = "evals"

# Free-tier capacity is the binding constraint on how fast the suite can run.
PACING_SECONDS = 1.5

# Hiragana and katakana. Kanji alone is not evidence of Japanese output.
_KANA = re.compile(r"[぀-ヿ]")


@dataclass
class Run:
    """What one agent invocation actually did."""

    text: str = ""
    tools: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    response_tokens: int = 0

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.response_tokens


class QuotaExhausted(RuntimeError):
    """The account's daily request allowance is gone. Retrying cannot help."""


def _classify_upstream(exc: BaseException) -> str | None:
    """Return "retry", "quota", or None if this is not an upstream failure.

    ADK wraps a 429 in its own private error class, so matching on exception
    type alone misses it. The distinction that matters is not the status code
    but whether waiting will help: a per-minute limit clears in seconds, a
    daily limit does not clear today, and burning six backoffs to rediscover
    that wastes the little quota that is left.
    """
    text = f"{type(exc).__name__}: {exc}"
    if "PerDay" in text or "per day" in text:
        return "quota"
    if any(m in text for m in ("429", "RESOURCE_EXHAUSTED", "ResourceExhausted")):
        return "retry"
    if any(m in text for m in ("503", "UNAVAILABLE", "500", "INTERNAL")):
        return "retry"
    return None


async def invoke(agent, message: str, attempts: int = 5) -> Run:
    """Run one ticket, retrying on transient upstream failures.

    A 503 from the model provider is not a finding about the agent. Without a
    retry the suite reports a capacity blip as a behavioural regression, which
    is worse than no suite at all -- you stop trusting the red.

    Backoff is capped rather than doubling forever, and every call is preceded
    by a short pause: running the whole suite back-to-back is itself enough to
    get rate-limited, which would make the suite's own load the thing under
    test.
    """
    for attempt in range(1, attempts + 1):
        await asyncio.sleep(PACING_SECONDS)
        try:
            return await _invoke_once(agent, message)
        except Exception as exc:  # noqa: BLE001 -- re-raised unless upstream
            kind = _classify_upstream(exc)
            if kind is None:
                raise
            if kind == "quota":
                raise QuotaExhausted(
                    "daily request quota exhausted -- the suite cannot finish today"
                ) from exc
            if attempt == attempts:
                raise UpstreamUnavailable(str(exc)[:200]) from exc
            backoff = min(2**attempt, 30)
            print(f"  (upstream error, retry {attempt}/{attempts - 1} in {backoff}s)")
            await asyncio.sleep(backoff)
    raise AssertionError("unreachable")


class UpstreamUnavailable(RuntimeError):
    """Transient upstream failure that outlasted our retries."""


async def _invoke_once(agent, message: str) -> Run:
    runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID
    )
    content = types.Content(role="user", parts=[types.Part(text=message)])
    out = Run()

    async for event in runner.run_async(
        user_id=USER_ID, session_id=session.id, new_message=content
    ):
        if event.usage_metadata:
            out.prompt_tokens += event.usage_metadata.prompt_token_count or 0
            out.response_tokens += event.usage_metadata.candidates_token_count or 0
        for part in (event.content.parts if event.content else []) or []:
            if part.function_call:
                out.tools.append(part.function_call.name)
        if event.is_final_response() and event.content and event.content.parts:
            out.text = "".join(p.text or "" for p in event.content.parts).strip()

    return out


def check(name: str, passed: bool, detail: str = "") -> dict:
    return {"check": name, "passed": passed, "detail": detail}


def score_classification(case: dict, run: Run) -> list[dict]:
    try:
        got = json.loads(run.text)
    except json.JSONDecodeError:
        return [check("classifier_returns_json", False, run.text[:120])]

    results = [check("classifier_returns_json", True)]

    want_category = case["category"]
    got_category = got.get("category")
    results.append(
        check(
            "category",
            got_category == want_category,
            f"want {want_category}, got {got_category}",
        )
    )

    want_fn = case.get("function")
    got_fn = got.get("function_name")
    results.append(
        check(
            "function_extraction",
            got_fn == want_fn,
            f"want {want_fn!r}, got {got_fn!r}",
        )
    )

    want_tel = case.get("needs_telemetry")
    got_tel = got.get("needs_telemetry")
    results.append(
        check("needs_telemetry", got_tel == want_tel, f"want {want_tel}, got {got_tel}")
    )
    return results


def score_triage(case: dict, run: Run) -> list[dict]:
    results = []
    lowered = run.text.lower()
    called = set(run.tools)

    for tool in case.get("expect_tools") or []:
        results.append(
            check(f"calls:{tool}", tool in called, f"called {sorted(called) or 'nothing'}")
        )
    for tool in case.get("forbid_tools") or []:
        results.append(
            check(
                f"avoids:{tool}",
                tool not in called,
                f"called {sorted(called) or 'nothing'}",
            )
        )

    # Groundedness proxy: phrases that would only appear if the agent invented
    # a mechanism the tools never reported.
    for phrase in case.get("must_not_mention") or []:
        results.append(
            check(f"no_invention:{phrase}", phrase.lower() not in lowered, "found it")
        )

    # Any-of groups: the answer must convey each required fact somehow.
    for group in case.get("must_mention") or []:
        alternatives = group if isinstance(group, list) else [group]
        hit = any(alt.lower() in lowered for alt in alternatives)
        results.append(check(f"mentions:{alternatives[0]}", hit, "missing"))

    if case.get("expect_abstain"):
        # Abstaining means asking, or saying plainly that the data is not there.
        asked = "?" in run.text or "？" in run.text
        results.append(check("abstains", asked, "did not ask for anything"))

    want_lang = case.get("expect_language")
    if want_lang == "ja":
        results.append(check("language:ja", bool(_KANA.search(run.text)), "not Japanese"))
    elif want_lang == "en":
        results.append(check("language:en", not _KANA.search(run.text), "not English"))

    return results


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="run only these ticket ids")
    parser.add_argument("--json", type=Path, help="also write full results here")
    args = parser.parse_args()

    cases = yaml.safe_load(TICKETS.read_text())
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]
        if not cases:
            sys.exit(f"no tickets matched {args.only}")

    report = []
    for case in cases:
        print(f"\n[{case['id']}] {case['ticket']}")

        # An upstream outage is not a verdict on the agent, so it is recorded
        # as an error and excluded from the pass rate rather than counted as
        # a failure. A suite that turns red for the wrong reason gets ignored.
        try:
            cls_run = await invoke(classifier_agent, case["ticket"])
            checks = score_classification(case, cls_run)
            tokens = cls_run.tokens
            tool_calls = 0

            if not case.get("skip_triage"):
                tri_run = await invoke(root_agent, case["ticket"])
                checks += score_triage(case, tri_run)
                tokens += tri_run.tokens
                tool_calls = len(tri_run.tools)
                print(f"  tools: {tri_run.tools or 'none'}")
        except QuotaExhausted as exc:
            # Stop rather than marking the rest as errors: a run that only
            # covered the first two tickets should not look like a completed
            # run with seven mystery failures.
            print(f"  STOP   {exc}")
            report.append(
                {"id": case["id"], "checks": [], "tokens": 0, "tool_calls": 0,
                 "error": "daily quota exhausted"}
            )
            break
        except UpstreamUnavailable as exc:
            print(f"  ERROR  upstream unavailable, not scored: {exc}")
            report.append(
                {"id": case["id"], "checks": [], "tokens": 0, "tool_calls": 0,
                 "error": "upstream unavailable"}
            )
            continue

        for c in checks:
            mark = "PASS" if c["passed"] else "FAIL"
            suffix = "" if c["passed"] else f"  <- {c['detail']}"
            print(f"  {mark}  {c['check']}{suffix}")

        report.append(
            {
                "id": case["id"],
                "checks": checks,
                "tokens": tokens,
                "tool_calls": tool_calls,
            }
        )

    errored = [r["id"] for r in report if r.get("error")]
    scored = [r for r in report if not r.get("error")]
    total = sum(len(r["checks"]) for r in scored)
    passed = sum(1 for r in scored for c in r["checks"] if c["passed"])
    tokens = sum(r["tokens"] for r in scored)
    tool_calls = sum(r["tool_calls"] for r in scored)
    n = len(scored) or 1

    print("\n" + "=" * 60)
    print(f"checks     {passed}/{total} passed ({passed / total:.0%})" if total else "no checks scored")
    print(f"tickets    {len(scored)} scored, {len(errored)} errored")
    print(f"tokens     {tokens} total, {tokens // n} per ticket")
    print(f"tool calls {tool_calls} total, {tool_calls / n:.2f} per ticket")

    failed = [r["id"] for r in scored if any(not c["passed"] for c in r["checks"])]
    if failed:
        print(f"tickets with failures: {', '.join(failed)}")
    if errored:
        print(f"tickets not scored:    {', '.join(errored)}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "checks_passed": passed,
                    "checks_total": total,
                    "tokens": tokens,
                    "tool_calls": tool_calls,
                    "tickets": report,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"wrote {args.json}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
