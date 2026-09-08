"""Score the agent against labelled tickets and print the numbers.

    uv run python -m evals.run
    uv run python -m evals.run --only perf-en-01 healthy-en-03
    uv run python -m evals.run --repeat 5 --json results.json

Both stages are scored, and the intake gate between them is honoured. That last
part was a correction: scoring triage on the raw ticket measured a stage that
production never reaches unaided, and reported the ungated agent's guesses as the
product's behaviour. The suite now runs what ships.

`--repeat` exists because a single run is not a measurement. The first two runs
of this suite scored 100% and 99%, and the difference was not a code change --
the agent volunteered an ungrounded cause in one run and not the other. A pass
rate quoted from one run is a sample of size one presented as a property. With
repeats the output separates checks that always pass from checks that sometimes
pass, and the second group is the interesting one.

The classifier either routes a ticket correctly or it doesn't, which is a label
comparison. Triage is scored on behaviour: which tools it reached for, whether it
invented anything, whether it abstained when it should have, and what it cost.

No judge model is used. Every check here is a deterministic comparison against
an expectation written down in tickets.yaml before the run, so a regression is
a number moving rather than an opinion changing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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
from support_agent.models import describe as describe_models  # noqa: E402
from support_agent.pipeline import decide, triage_prompt  # noqa: E402

TICKETS = Path(__file__).parent / "tickets.yaml"
APP_NAME = "agentic-support-evals"
USER_ID = "evals"

# A small gap between calls. Bedrock does not need it, the Gemini free tier
# does, and a suite that only passes on one provider is not measuring the agent.
PACING_SECONDS = float(os.environ.get("EVAL_PACING_SECONDS", "0.5"))

# Hiragana and katakana. Kanji alone is not evidence of Japanese output.
_KANA = re.compile(r"[぀-ヿ]")

# Function words common in English and rare or absent in the other languages a
# model reaches for unprompted. Deliberately excludes words shared with Spanish
# ("no", "es", "a") -- the check exists because an answer in Spanish passed the
# earlier version, which only tested that the text was not Japanese. "Not
# Japanese" was never the same claim as "English", and the gap stayed invisible
# until a run answered an English ticket in Spanish and scored green.
#
# One marker is enough. Requiring two was the first attempt and it failed the
# gate's own English reply, which is three sentences long and happens to contain
# exactly one of them -- the check would have reported a canned constant string
# as the wrong language. A test now pins the shipped replies against this.
_ENGLISH_MARKERS = (
    " the ",
    " and ",
    " for ",
    " with ",
    " that ",
    " you ",
    " is ",
    " it ",
    " to ",
    " will ",
    " does ",
    " not ",
)



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


# Interrogative markers. Punctuation alone was the original test and it was
# wrong: Japanese questions routinely end in "。", so a correct Japanese
# clarifying reply was scored as a failure to ask. The bug was in the check, not
# the agent, which is the failure mode an eval suite has to be most careful
# about -- it sends you off to fix working code.
_ASKS = (
    "?",
    "？",
    "ですか",
    "でしょうか",
    "どの",
    "どちら",
    "教えて",
    "ください",
    "which",
    "what ",
    "could you",
    "can you",
    "would you",
    # A request does not have to be phrased as a question: "I need you to confirm
    # the exact function name" is an abstention, and the marker list missed it
    # because it was looking for interrogatives.
    "need you to",
    "confirm the exact",
    "please confirm",
    "please let",
    "let us know",
)


def _asks_for_information(text: str) -> bool:
    """Whether the reply asks the customer for something rather than diagnosing."""
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in _ASKS)


# Sentence enders, including the Japanese full stop, plus line breaks so a
# markdown bullet or heading counts as its own clause.
_SENTENCE_SPLIT = re.compile(r"[.!?。\n]+")

# Phrases that mark a clause as claiming causation rather than merely naming a
# mechanism.
_CAUSE_MARKERS = ("root cause", "probable cause", "caused by", "due to", "原因", "起因")

# Phrases that rule a mechanism out or defer it. A clause carrying one of these
# is not asserting the mechanism even if it also mentions causation.
_RULED_OUT = (
    "no evidence",
    "not appear",
    "none of",
    "no sign",
    "no indication",
    "not listing",
    "ruled out",
    "ありません",
    "見られません",
    "なし",
    # Japanese negation is clause-final, so it attaches to a predicate some
    # distance from the mention: "...を直接示すログはないため、根本原因の特定は
    # ここまでとし" disclaims the mechanism and names 根本原因 in the same clause.
    # Without these the check reads that as an assertion.
    "はない",
    "できません",
    "不明",
)


def _asserts_as_cause(text: str, phrase: str) -> bool:
    """Whether `phrase` is put forward as a cause, rather than merely appearing.

    A plain substring test was the first version and it was wrong in both
    directions of usefulness. It failed this answer:

        "No evidence of throttling, cold starts, or deployment changes -- none
         of those appeared in the metrics or logs."

    which is the agent doing exactly what it should: naming what it ruled out.
    It also failed a Japanese answer that recommended checking DynamoDB-side
    throttling metrics as a next step, while stating that its own tools could not
    see them. Both were scored as invented mechanisms, so the suite's number went
    down while the answers got better -- the most expensive kind of wrong metric,
    because it argues for reverting an improvement.

    What the check is actually for is the assertion: "probable root cause:
    connectivity or throttling issues", where no tool reported throttling. So the
    unit is the clause, and the clause has to claim causation and not disclaim the
    mechanism.

    This is still a proxy, and it errs on the side of exempting: an invention
    wrapped inside a disclaiming clause would pass. That direction is the right
    one to err in for a check whose failures send a human to read the answer, but
    it is a proxy either way, and no substring rule is going to become a
    groundedness verifier. What it does reliably is catch the specific shape that
    reaches a customer as a false claim -- "probable root cause: X" where no tool
    reported X.
    """
    lowered = phrase.lower()
    for clause in _SENTENCE_SPLIT.split(text):
        low = clause.lower()
        if lowered not in low:
            continue
        if any(r in low for r in _RULED_OUT):
            continue
        if any(c in low for c in _CAUSE_MARKERS):
            return True
    return False


def _looks_english(text: str) -> bool:
    """Positive evidence of English, not merely the absence of Japanese."""
    padded = f" {text.lower()} "
    return any(m in padded for m in _ENGLISH_MARKERS)


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

    # Groundedness proxy: mechanisms no tool reported, asserted as the cause.
    # Naming one to rule it out, or to suggest it as a next step the tools cannot
    # see, is not invention -- see _asserts_as_cause.
    for phrase in case.get("must_not_mention") or []:
        asserted = _asserts_as_cause(run.text, phrase)
        results.append(
            check(f"no_invented_cause:{phrase}", not asserted, "asserted as a cause")
        )

    # Any-of groups: the answer must convey each required fact somehow.
    for group in case.get("must_mention") or []:
        alternatives = group if isinstance(group, list) else [group]
        hit = any(alt.lower() in lowered for alt in alternatives)
        results.append(check(f"mentions:{alternatives[0]}", hit, "missing"))

    if case.get("expect_abstain"):
        results.append(
            check("abstains", _asks_for_information(run.text), "did not ask for anything")
        )

    want_lang = case.get("expect_language")
    if want_lang == "ja":
        results.append(check("language:ja", bool(_KANA.search(run.text)), "not Japanese"))
    elif want_lang == "en":
        english = not _KANA.search(run.text) and _looks_english(run.text)
        results.append(check("language:en", english, f"not English: {run.text[:60]}"))

    return results


async def run_suite(cases: list[dict]) -> list[dict]:
    """Score every case once. Returns one record per ticket."""
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
                # Score the system as it ships, gate included. Running triage
                # on the raw ticket measured a stage that production never
                # reaches on its own, and reported the ungated agent's guess as
                # the product's behaviour.
                decision = decide(case["ticket"], cls_run.text)
                if decision.proceed:
                    tri_run = await invoke(root_agent, triage_prompt(case["ticket"]))
                    tokens += tri_run.tokens
                    tool_calls = len(tri_run.tools)
                    print(f"  tools: {tri_run.tools or 'none'}")
                else:
                    tri_run = Run(text=decision.reply or "")
                    print("  gated at intake, triage not run")
                checks += score_triage(case, tri_run)
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
                # Kept so a failure can be diagnosed from the artifact instead
                # of by re-running a non-deterministic agent and hoping it
                # misbehaves the same way twice.
                "classification": cls_run.text,
                "answer": tri_run.text if not case.get("skip_triage") else None,
            }
        )

    return report


def summarise(report: list[dict]) -> tuple[int, int]:
    """Print one run's numbers. Returns (passed, total)."""
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

    return passed, total


def report_stability(runs: list[list[dict]]) -> list[dict]:
    """Print per-check pass counts across runs and return the unstable ones.

    The split that matters is not pass/fail but always/sometimes. A check that
    fails every run is a defect with a known cause; a check that fails one run in
    three is a defect you will close as "could not reproduce" unless the suite
    tells you it is intermittent. Anything in between 0 and n is reported by name
    so the flakiness is a printed number rather than a thing you remember.
    """
    tally: dict[tuple[str, str], list[bool]] = {}
    for report in runs:
        for ticket in report:
            for c in ticket["checks"]:
                tally.setdefault((ticket["id"], c["check"]), []).append(c["passed"])

    n = len(runs)
    unstable = []
    for (ticket_id, name), outcomes in sorted(tally.items()):
        # Missing observations mean an errored ticket, not a pass.
        passes = sum(outcomes)
        if passes == len(outcomes) == n:
            continue
        record = {"ticket": ticket_id, "check": name, "passes": passes, "runs": n}
        unstable.append(record)

    print("\n" + "=" * 60)
    print(f"stability over {n} runs")
    if not unstable:
        print("  every check passed in every run")
        return unstable
    for r in unstable:
        kind = "always fails" if r["passes"] == 0 else "intermittent"
        print(f"  {r['passes']}/{r['runs']}  {r['ticket']}  {r['check']}  ({kind})")
    return unstable


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="run only these ticket ids")
    parser.add_argument("--json", type=Path, help="also write full results here")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run the suite N times and report which checks are intermittent",
    )
    args = parser.parse_args()

    cases = yaml.safe_load(TICKETS.read_text())
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]
        if not cases:
            sys.exit(f"no tickets matched {args.only}")

    models = describe_models()
    print(models)

    runs = []
    rates = []
    for i in range(args.repeat):
        if args.repeat > 1:
            print(f"\n{'#' * 60}\n# run {i + 1}/{args.repeat}\n{'#' * 60}")
        report = await run_suite(cases)
        passed, total = summarise(report)
        runs.append(report)
        if total:
            rates.append(passed / total)

    unstable = report_stability(runs) if args.repeat > 1 else []
    if len(rates) > 1:
        print(f"pass rate  best {max(rates):.0%}, worst {min(rates):.0%}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "models": models,
                    "repeat": args.repeat,
                    "pass_rates": rates,
                    "unstable_checks": unstable,
                    "runs": runs,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"wrote {args.json}")

    # Intermittent failures still fail the run. A suite that goes green when a
    # groundedness check passes two times in three is worse than no suite.
    return 1 if min(rates, default=1.0) < 1.0 else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
