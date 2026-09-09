# agentic-support

A support-triage agent built on [Google ADK](https://google.github.io/adk-docs/),
running on either Bedrock (Claude) or the Gemini API — ADK is the agent framework
here, not the model provider.

Given a vague inbound ticket, it decides which telemetry to pull, reads it, and
returns a probable root cause with an explicit confidence label — or says it
doesn't know.

The interesting part is not the answer. It's that every tool call is visible and
every factual claim is traceable to a tool result.

There is a second half with no model in it at all. [`queue_triage/`](queue_triage/)
ranks an entire queue by how cheap each ticket looks to close, so *which* ticket to
open first is decided before the agent is involved — deterministically, with every
score explainable as a list of rules that fired. See
[Before the agent](#before-the-agent-which-ticket-first).

## Why

I handle cloud support tickets for a living. The first ten minutes of a ticket are
almost always the same shape: figure out which resource the customer means, look
at its metrics, decide whether the logs are worth reading, form a hypothesis. It's
mechanical, and it's exactly where an agent should be useful.

The failure mode that makes such an agent useless in front of a customer is
confident invention — "your Lambda is being throttled" when nothing in the
telemetry says so. So the design constraint here is stricter than "be helpful":

- Every factual claim about the customer's system must come from a tool result.
- If the tools contradict the ticket, report the contradiction instead of
  smoothing it over.
- If the ticket doesn't name a resource, ask. Don't guess.

That last one matters more than it looks. An agent that guesses which function
the customer meant will eventually diagnose the wrong system, correctly, and
be believed.

## How it works

```
ticket text
    │
    ▼
┌──────────────────────────────┐
│ intake_classifier            │  no tools, structured output:
│  text only, no telemetry     │  category / resource / urgency
└──────────┬───────────────────┘
           ▼
     ┌───────────────┐   nothing to look up
     │ pipeline gate │──────────────────────▶ ask for the resource,
     └───────┬───────┘                        or answer from the guide
             │ resource named
             ▼
┌──────────────────────────────┐
│ triage_agent (LlmAgent)      │
│  grounding rules +           │
│  triage procedure            │
└──────────┬───────────────────┘
           │ tool-calling loop
     ┌─────┴──────┐
     ▼            ▼
get_function   get_recent
_metrics       _logs
     │            │
     └─────┬──────┘
           ▼
     policy.py          ← ordering + not_found rules live here
           │
           ▼
    mock_infra.py
  (stands in for the
   customer's own APIs)
```

Intake runs first and cheaply: text only, no tools to misuse, output a fixed enum
that can be scored against labels. Triage is the expensive stage, and the gate
decides whether a ticket earns it. Splitting them means a misrouted ticket shows
up as a misroute instead of surfacing later as a confidently wrong diagnosis.

The agent is told to reach for metrics first and to pull logs *only* if the
metrics suggest something is actually wrong. Cheap signal before expensive
signal — the same order a human on-call would use, and on the majority of tickets
where nothing is broken it should keep the token cost down. It doesn't yet;
see [Known gap](#known-gap).

`mock_infra.py` is deliberately the only file that knows what the backing systems
are. In a real engagement those two functions would call an observability
backend, a CMDB, a ticketing system — each behind its own auth and network
boundary. When the real integrations land, only that file changes; the agent
layer doesn't.

It carries both shapes of data on purpose:

- **structured** — invocation counts, error rates, p99 latency
- **unstructured** — raw log lines, including the noisy repeated-error case

because the messy one is what retrieval has to cope with later.

## Running it

Needs [uv](https://docs.astral.sh/uv/) and one of: AWS credentials with Bedrock
access, or a Gemini API key.

```bash
uv sync

# either
export AWS_REGION=us-west-2          # uses Bedrock, see Models below
# or
cp support_agent/.env.example support_agent/.env
# put a Gemini key in support_agent/.env -- https://aistudio.google.com/apikey
```

Ask it something:

```bash
uv run python -m scripts.ask "checkout-api is slow this morning"
```

```
  intake: performance_degradation, resource='checkout-api', urgency=high
  -> tool: get_function_metrics({'function_name': 'checkout-api'})
  <- result: get_function_metrics returned
  -> tool: get_recent_logs({'function_name': 'checkout-api'})
  <- result: get_recent_logs returned

## Triage Summary — checkout-api

**Metrics:**
- Invocations: 128,400
- Error rate: 8.1% (elevated)
- p99 latency: 9,850 ms (near the function's 10s timeout ceiling)

**Logs (pulled due to abnormal metrics):**
- Repeated `Task timed out after 10.00 seconds`
- Memory usage nominal (498/512 MB) — not a memory-pressure issue
- `ConnectTimeoutError` connecting to `dynamodb.ap-northeast-1.amazonaws.com`

**Root cause (probable):** The function is timing out while waiting to establish
a connection to DynamoDB in `ap-northeast-1`. This matches the customer's "slow"
report — invocations are running to the full 10s timeout and failing due to a
stalled downstream connection, not application-level slowness.

**Confidence: high** — the timeout duration, near-ceiling p99, and explicit
ConnectTimeoutError to DynamoDB all corroborate the same failure path.

[tokens] prompt=5816 response=684
```

Note the diagnosis is not the one the ticket implied. The customer said "slow";
the function is *timing out*, which is a different problem with a different fix.

The tool trace is printed for every run, along with the token cost. An agent
whose intermediate steps you can't see is an agent you can't debug in front of
a customer.

Or use the ADK web UI, which gives you the same trace in a browser:

```bash
uv run adk web
```

### Things worth trying

| Ticket | What it exercises |
|---|---|
| `"checkout-api is slow this morning"` | metrics → logs → root cause (DynamoDB connect timeout, not slowness) |
| `"notifications aren't going out"` | no resource named — the agent should ask rather than guess |
| `"invoice-batch is throwing errors"` | telemetry is healthy — the agent should push back on the premise |
| `"payment-api is down"` | unknown function — `not_found` handling |

The third one is the real test. A model that wants to be agreeable will find a
problem in `invoice-batch` because it was told there is one.

### The gap that started the measurement

On the `invoice-batch` ticket the agent got the conclusion right — healthy
telemetry, no invented failure — but pulled the logs anyway, which the instruction
told it to do only when the metrics suggested a problem. Grounding held; cost
ordering didn't. Instructions are not a control plane.

That is now enforced in [`policy.py`](support_agent/policy.py), and it's worth
being precise about what the fix buys. The agent still *attempts* the log call and
receives `skipped_by_policy`, then answers from the metrics. Grounding is
protected and the answer is right, but the request still goes out, so the call
count isn't reduced. Cutting cost needed the intake gate, not the tool gate — and
that distinction only became visible because the two were measured separately.

## Models

Two providers, selected by which credentials are present, overridable per stage:

```bash
SUPPORT_AGENT_BACKEND=bedrock     # Claude Sonnet 5 triage, Haiku 4.5 intake
SUPPORT_AGENT_BACKEND=gemini      # gemini-3.8-flash
SUPPORT_AGENT_MODEL=<model>            # triage only
SUPPORT_AGENT_CLASSIFIER_MODEL=<model> # intake only
```

Intake and triage are configured separately on purpose: classification is a
cheap fixed-output task and does not need the model triage needs.

The second provider was not a portability exercise for its own sake. The Gemini
free tier allows 20 requests per day per model and one suite run needs more than
that, so the measurement harness could be run roughly once a day — barely a
harness. Bedrock removed the ceiling and the numbers below became repeatable.

Porting was also the fastest audit of how much of the design had leaked into one
vendor. One thing had: intake declared urgency as an integer with `ge=1, le=4`,
which Gemini accepted and Bedrock rejected outright — `minimum`/`maximum` are
not supported on an integer in its structured-output schema. Replacing it with a
string enum is portable *and* better, because the values now carry their own
meaning and match [`sla.py`](support_agent/sla.py)'s urgency levels exactly, so
the AI stage and the policy stage need no translation table. A test asserts the
two enums stay in sync, because the failure if they drift is a deadline computed
from a value nobody validated.

Every run prints the resolved model ids. That line exists because a stale
`SUPPORT_AGENT_MODEL` left in `.env` silently overrode the backend and sent a
full suite run to the wrong provider, where it died on the free-tier limit — and
nothing in the output said which model had been used.

`GOOGLE_GENAI_USE_VERTEXAI=TRUE` in `.env` runs the same agent against Vertex AI
instead of the Gemini API directly.

## Layout

```
support_agent/
  classifier.py   intake: text-only classification, structured output
  pipeline.py     the gate: does this ticket earn a triage run?
  agent.py        triage: model, instruction, tool wiring
  policy.py       tool-access rules the agent cannot talk its way past
  models.py       provider selection, one place, no vendor in the agents
  sla.py          priority matrix and the business-hours clock
  quality.py      post-hoc review of handled tickets -> scorecard
  mock_infra.py   the only file that knows what the backing systems are
queue_triage/     the step before triage: rank a queue by estimated effort
  query.py        search expression -> AST, no backend knowledge
  emit.py         AST -> predicate or JSON; one function per backend's quirks
  rules.py        ruleset loader; signals are data, not code
  score.py        weighted signals -> score, label, and why
  backends.py     a backend is two methods; the bundled one reads YAML
  rulesets/       cloud_support.yaml -- the weights, editable without a build
scripts/
  ask.py          CLI runner -- prints the tool trace and token cost
evals/
  tickets.yaml    labelled tickets, expectations written before the run
  run.py          harness -- scores both stages, reports cost and stability
  queue.yaml      labelled queue: synthetic tickets with effort labels
  rank.py         ranking harness -- rank agreement, ablation, per-signal checks
tests/            160+ deterministic tests, no API key needed
```

## The parts that are not AI

An agent that diagnoses well and answers late has still missed the commitment,
so two modules are deliberately plain code with no model involved.

**[`sla.py`](support_agent/sla.py)** — impact × urgency → P1–P4, then business-hours
deadlines. Impact is weighted above the customer's self-reported urgency: one
tenant calling something critical is a P2, a shared component degrading for
everyone at merely medium urgency is also a P2. Self-reported urgency alone would
make the loudest ticket the most important one.

The clock is where the bugs live. Elapsed wall-clock time is the wrong measure
for a desk that opens at 09:30 and closes at 18:15: a P1 raised at 17:00 on a
Friday is not breached by Monday morning. So `BusinessCalendar` counts only open
minutes, and stops for weekends, Japanese public holidays and the 12/29–1/3
closure. A phantom breach is worse than no measurement, because someone then
spends a morning explaining that the number is wrong.

Two decisions in there are deliberate and would be worth arguing about in review:

- `escalate()` raises the priority but does **not** recompute deadlines. A
  commitment already made to the customer is not something an internal
  reclassification gets to shorten.
- The holiday list is hardcoded, and the code says in a comment that this is a
  placeholder to be replaced by an authoritative feed. Hardcoded calendar data is
  wrong on a schedule — it just doesn't tell you when.

**[`quality.py`](support_agent/quality.py)** — reviews already-handled tickets and
emits findings by severity: `BREACH` (a target missed), `PROCESS` (escalated
without justification, or a P1 nobody escalated), `HYGIENE` (no resource
identified, no evidence cited). The scorecard aggregates SLA attainment, reopen
rate and evidence rate.

The denominator is the part that matters. First-response attainment counts the
whole batch, including tickets that never got a first response at all —
excluding them would let the worst tickets improve the number, which is how a
metric ends up pointing the wrong way.

## Before the agent: which ticket first

[`queue_triage/`](queue_triage/) answers the question that comes before any of the
above. Forty untouched tickets, one of you, and no signal about which are
twenty-minute answers and which will eat the afternoon. Sorting by age puts the
oldest first; sorting by severity trusts a field the customer filled in.

It is deliberately not an agent. No model, no network, and every score is a list
of rules that fired with a weight attached:

```bash
uv run python -m queue_triage queue --explain
```

```
[+19] quick-win    T-1003  Startup program credits not applied to my January invoice  (+16 effort, +3 waited-24d)
       +5 credit-missing -- 'credits not applied'  (credit did not arrive or vanished)
       +4 credits -- 'credits'  (credit balance or grant question)
       +3 startup-credits -- 'Startup program'
       +2 invoice -- 'invoice'
       +2 severity:low
       +3 waited-24d  (aging, order only)

[ +3] unclear      T-1009  Unauthorised charge of $27.40 on credit card  (+0 effort, +3 waited-21d)
       -5 suspected-fraud -- 'Unauthorised'  (needs identity verification first)
       +3 refund -- 'refund'
       +1 short-subject -- 'Unauthorised charge of $27.40 on credit card'  (terse subject)
       +1 severity:normal
       +3 waited-21d  (aging, order only)
      x   would be excluded: refund-request -- refund workflow, not a support answer
      !   known-hard example: The phrase "credit card" made an earlier version of
          the ruleset score this as a credits question worth +6...
```

The second one is why the explanation is not decoration. `credit card` contains
`credit`, so an early version scored a suspected-fraud dispute as a routine credit
question and sorted it into the quick wins. Reading the per-signal breakdown found
it; a test would only have found it if I had already thought of it.

**Two things sit outside the score, on purpose.** How long a ticket has waited, and
whether it is a shape the queue should not be offering at all.

Age is a fairness constraint, not evidence about effort. A ticket gains a point a
week up to a ceiling of four, which is enough to put a fortnight-old ticket ahead
of a same-day one that scored a point or two higher, and not enough to let age beat
the gap between a template reply and an outage — without the ceiling this collapses
into first-in-first-out, which is the thing the ranker exists to improve on. The
bonus lands in `Verdict.adjustments` rather than `Verdict.score`, so a ticket never
becomes a `quick-win` by sitting in the queue, and the sort falls back to the
creation date before the ticket id, so among equally cheap tickets the one that has
been waiting longest is offered first.

Exclusions are the other layer: refunds and billing adjustments are dropped rather
than penalised. A penalty says "this looks expensive" and leaves the ticket on
screen at the bottom, which is not what "someone else's workflow" means — no score
can express that, whatever the number. The dropped tickets are counted and named on
stderr (`dropped 3: 3x refund-request`) because a ticket that vanishes silently is
indistinguishable from a broken query, and `--keep-excluded` brings them back.

Neither layer touches the measured path. `evals/rank.py` scores every ticket,
including the ones the queue hides, and runs with aging off — and there are
[tests that keep it that way](tests/test_rank_eval.py). That matters in both
directions: two of the adversarial cases are refund requests and are the sharpest
tests of the damping mechanism in the set, and the labelled tickets happen to be
written oldest-easiest, so letting age into `score` would have lifted the agreement
number without the ruleset getting any better. Adding both layers moved the eval by
exactly nothing, which is the point.

**Why a ruleset and not a classifier.** A model would probably rank these better.
It would also cost a call per ticket, take a second per ticket, and answer "why is
this at the top" with a plausible sentence rather than an audit trail. For a
ranking that only has to be roughly right, and that a human overrides for free by
picking a different row, the deterministic version is the better trade. The
[ruleset is a YAML file](queue_triage/rulesets/cloud_support.yaml) so changing a
weight is not a code change.

**The pieces.** The query compiler stops at an AST
([`query.py`](queue_triage/query.py)) and the backend-specific encoding lives in
[`emit.py`](queue_triage/emit.py). That split is the whole design: every real
search API has quirks in how it wants a nested boolean tree encoded, and those
quirks return plausible wrong answers rather than errors. The one worth naming is
that an unquoted multi-word value is often tokenised and OR'd, so `body:missing
credits` matches *more* than `body:credits` — a filter that reads like a narrowing
is a widening. With an AST in between, the parser has one job and each backend's
weirdness is one function. A backend is two methods; the bundled one reads a YAML
file, so everything above runs with no credentials.

### What the ranking eval found

```bash
uv run python -m evals.rank --signals --ablate
```

The ground truth is a hand-set prior, not observed handling time, which bounds
what can be claimed: rank agreement and ablation deltas, not accuracy. The
harness prints no accuracy figure and a
[test asserts it never starts to](tests/test_rank_eval.py).

Current numbers on 32 labelled tickets — Spearman **+0.845**, no quick ticket
ranked below a hard one, top five all genuinely quick. But the interesting output
was the diagnostics, which found three real defects:

**Two signals had never fired, and looked identical to two that simply did not
apply.** `short-subject` and `long-subject` measure how long the subject line is,
using `^.{0,45}$`. They ran against every field joined by newlines, where `.` does
not cross a newline and so the pattern cannot match — ever. Both were dead code
from the day they were written. The fix was to give signals a `scope`, which is
now required for anything measuring a field's shape rather than its content.
`short-subject` went from 0 firings to 16.

**One ticket produced every badly inverted pair on the set.** The credit-card
fraud dispute, still scoring +4 after the false positive above was patched.
Suspected fraud reads as a billing question and is not one — the person writing in
may not own the account. Adding a `suspected-fraud` signal took badly inverted
pairs from 4 to 0 and Spearman from +0.810 to +0.845.

**The ablation on context damping came back at exactly zero, and the mechanism was
not the problem.** Damping halves service-name penalties when a billing signal has
already fired, because a customer disputing a bill names whichever line item
surprised them. It measured as worth nothing because *the dataset had no example
of the case it exists for* — the one ticket I thought covered it named a NAT
gateway, which had already been removed from the networking pattern for the same
reason. A billing ticket naming Direct Connect port hours now covers it. The
honest number is still small (+0.003 Spearman, one ticket's worth), and it is
reported that way rather than rounded up into a claim.

The per-signal report is the check I would keep if I could keep only one: for each
signal, the mean label of the tickets it fires on against the mean of those it
does not. A positive weight whose tickets take *longer* than average has its sign
wrong, and no amount of tuning the magnitude fixes that. It also refuses to make
that call below three firings, because one ticket landing above or below the mean
is a coin flip, not evidence.

## Continuous integration

[CI](.github/workflows/ci.yml) runs the deterministic half on every push: the
priority matrix, the business clock, the scorecard, the eval harness's own scoring
logic, and all of `queue_triage` including its ranking eval. No credentials, no
model calls, no flakiness.

The ranking eval belongs in CI precisely because it has no model in it. It is a
fixed dataset through fixed rules, so any change in the numbers is a change in the
ruleset, and the suite fails on the two things worth failing on: a signal that
never fires, and a signal whose weight disagrees with the labels.

The agent evals are *not* in CI. They cost money, they need keys, and they are
non-deterministic — wiring them to every push would either leak a key or turn the
build red for reasons unrelated to the commit, and a build that is red for the
wrong reason gets ignored. They run on demand with `--repeat` and their numbers
go in this README with the run count attached.

The harness's checks get tests of their own, which is not paranoia: the
abstention check shipped broken once. It looked for `?` and so failed every
correct Japanese clarifying reply, because Japanese questions routinely end in
`。`. The suite was reporting a working agent as broken — the one eval failure
mode you cannot afford, because it sends you off to fix code that was right.

## Measuring it

```bash
uv run python -m evals.run                     # whole suite, once
uv run python -m evals.run --only vague-en-04  # one ticket
uv run python -m evals.run --repeat 5          # and report what is intermittent
uv run python -m evals.run --json out.json     # machine-readable
```

[`evals/tickets.yaml`](evals/tickets.yaml) holds labelled tickets. Each one
records, *before* the run, what a correct answer looks like: the expected intake
category, which tools triage should and should not call, substrings that would
prove the agent invented a mechanism, and whether the correct answer is a
question rather than a diagnosis. Cases whose right answer is "nothing is wrong"
or "I don't know" are first-class — those are the ones a helpful-sounding agent
fails.

No judge model is involved. Every check is a deterministic comparison against a
written-down expectation, so a regression is a number moving rather than an
opinion changing. Scored dimensions:

| Dimension | What it catches |
|---|---|
| intake category / resource extraction | tickets routed to the wrong place |
| tool discipline | calls that shouldn't have happened, and vice versa |
| groundedness proxy | mechanisms asserted that no tool reported |
| abstention | diagnosing when it should have asked |
| answer language | replying in English to a Japanese ticket |
| tokens & tool calls per ticket | what the above costs |

### One run is not a measurement

The agent is non-deterministic, so a pass rate from a single run is a sample of
size one presented as a property. `--repeat` runs the suite N times and splits
the checks into *always passes* and *sometimes passes*, because those two need
different responses: a check that fails every run is a defect with a known cause,
and a check that fails one run in five is a defect you will close as "could not
reproduce" unless the harness tells you it is intermittent.

Intermittent failures still fail the run. A suite that goes green when a
groundedness check passes four times in five is worse than no suite.

### What it found

Three defects, all invisible in ad-hoc testing:

**Guessing the resource.** On `notifications aren't going out` the classifier was
right — `function_name: null`, `needs_telemetry: false`. Triage, run on the raw
ticket, guessed `notify-fanout` and pulled its metrics anyway. The instruction
telling it not to guess was already there; it lost to having a tool, a plausible
candidate, and a customer who wants an answer.

**Substituting a neighbour.** Asked about `payment-api`, which doesn't exist,
the agent received `known_functions: [checkout-api, …]` in the `not_found`
payload, picked `checkout-api`, and reported its DynamoDB timeouts as the
answer — a correct diagnosis of a system nobody asked about, delivered with
confidence. The helpful error message was the cause.

**Answering in the wrong language.** The instruction said "answer in the language
the ticket was written in". Over five runs that held four times out of five: two
runs answered an English ticket about `invoice-batch` in **German**. Nothing in
the ticket was German — the model simply drifted, and a support desk that replies
in a language the customer didn't write in has failed regardless of how good the
diagnosis was.

This one was hidden twice over, because the check that should have caught it
tested only that the reply was not Japanese. German is not Japanese, so it scored
green. Fixing the check surfaced the defect on the very next run.

### What changed as a result

All three fixes move the constraint somewhere the model can't reach:

- [`policy.py`](support_agent/policy.py) gates log retrieval on the metrics
  being out of band, and strips the list of valid names out of `not_found`
  results. The agent is told what it doesn't have, not what it could have had
  instead.
- [`pipeline.py`](support_agent/pipeline.py) makes the classifier's decision
  binding: if intake says there's nothing to look up, triage never runs.
- `pipeline.triage_prompt()` decides the reply language from the ticket's script
  — a two-line function — and states it imperatively in the message. The model no
  longer infers it, because inferring it was where the variance came from.

`mock_infra.py` stays a faithful stand-in for the customer's systems and knows
about none of this.

The same lesson applies to the pattern, not just to each case: every one of these
started as a sentence in an instruction, and the sentence was already there when
the defect was measured. You get compliance in proportion to how much the model
already wanted to comply.

### The measurement that was wrong about the agent

The most useful thing this suite produced was a finding about itself.

After the fixes above, five consecutive runs scored between 97% and 100%, with
four intermittent groundedness failures spread across three tickets. Those
failures looked like the agent occasionally inventing a mechanism. Reading them
instead of counting them showed something else. Verbatim from the runs:

> **No evidence of** throttling, cold starts, or deployment changes — none of
> those appeared in the metrics or logs, so I'm not listing them as contributing
> factors.

> 追加で確認したい点があれば、DynamoDB側のテーブルメトリクス（スロットリングや
> キャパシティ状況など）も合わせて調査することをお勧めします。

The first is the agent stating what it ruled out. The second recommends checking
DynamoDB-side throttling metrics as a next step, while saying its own tools
cannot see them. Both are the behaviour you want from a support engineer, and
both were scored as invention — because the check was a substring test, and
`"throttl" in answer` cannot tell *caused by throttling* from *no evidence of
throttling*.

So the suite's number went down while the answers got better. That is the most
expensive way for a metric to be wrong, because it argues for reverting an
improvement, and it is invisible if you only read the pass rate.

The fix moves the unit of judgement from the answer to the clause: a mechanism
counts as invented only when the clause containing it claims causation and does
not disclaim the mechanism. Japanese needed a second pass, because negation there
is clause-final and lands well after the mention — "…を直接示すログはないため、
根本原因の特定はここまでとし" disclaims the mechanism and says 根本原因 in the same
breath.

Re-scoring the archived runs with the corrected check — no new model calls, the
answers were already saved — turned **95/95** groundedness checks green, and the
real failure found earlier still fails, which is the part that matters:

> **Probable root cause:** Connectivity or throttling issues between
> `checkout-api` and the DynamoDB endpoint in `ap-northeast-1`

Nothing reported throttling. That one is invention, and it is still caught. Both
strings are now test cases in [`tests/test_eval_checks.py`](tests/test_eval_checks.py),
taken verbatim from the runs rather than written to make the test pass.

There is a related result worth recording. When the throttling failure first
appeared, my fix was to add a rule to the triage instruction that named the
example — *"Connectivity or throttling issues" claims throttling*. Measured over
five runs, mentions of throttling went **up**, not down. Naming the term you don't
want makes it available. What the reworded rule did produce was the explicit
"no evidence of throttling, cold starts, or deployment changes" paragraph — an
improvement the broken check then punished.

### Honest status

Everything deterministic — the priority matrix, the business clock, the quality
scorecard, the harness's own scoring logic — is unit-tested and runs in
[CI](.github/workflows/ci.yml) on every push.

The agent numbers below are from five consecutive runs on Bedrock (Claude Sonnet
5 for triage, Haiku 4.5 for intake), with the run count attached because a single
run does not support a claim:

| | 5-run result |
|---|---|
| checks passed | 87/87 in 4 runs, 86/87 in 1 |
| pass rate | 100% best, 99% worst |
| tokens per ticket | ~3,950 (both stages) |
| tool calls per ticket | 1.00 |
| language correctness | 5/5, after the fix; 3/5 before |
| invented causes | 0/95 groundedness checks failed |

The one failure in that set was the abstention check missing a request phrased as
a statement — "I need you to confirm the exact function name" contains no
interrogative. Fixed in the marker list, with the string from that run as the test
case. Re-scoring all five archived runs with the corrected checks gives **195/195**
answer-level checks, which is not the same claim as "the suite passes five times
in a row" and is not written as if it were: the answers were fixed, only the
scoring changed.

The raw results are committed at [`evals/stability.json`](evals/stability.json) —
every answer, every check, every run — so the numbers above can be recomputed
rather than believed.

Two things this does not tell you. The suite is nine tickets against a mock
backend, so it measures the discipline of the agent rather than the difficulty of
real telemetry. And the `must_mention` groups are substring matches, which have
to be widened as models rephrase — one run conveyed "not found" as "no function
with that exact name exists" and was scored as a miss. That is the standing cost
of refusing a judge model, and it is the cheaper of the two costs: a judge forms
fresh opinions every run, and then the suite's own verdicts drift.

## Where this is going

Roughly in the order the complexity is worth it:

- **A real holiday feed** — `sla.py` hardcodes Japanese public holidays with a
  comment saying so. Hardcoded calendar data is wrong on a schedule and doesn't
  announce it.
- **Retrieval over runbooks** — the log lines above are recognisable to someone
  who has seen them before. That knowledge lives in runbooks, not in weights.
- **Delegation** — split triage from remediation, so the agent that proposes a
  fix isn't the one that decided what's broken.
- **FAQ candidates from clusters** — tickets that recur are documentation debt.
  Grouping them by intake category over time is the cheapest way to see which
  guide page is missing.

## License

MIT
