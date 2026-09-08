# agentic-support

A support-triage agent built on [Google ADK](https://google.github.io/adk-docs/).
Given a vague inbound ticket, it decides which telemetry to pull, reads it, and
returns a probable root cause with an explicit confidence label — or says it
doesn't know.

The interesting part is not the answer. It's that every tool call is visible and
every factual claim is traceable to a tool result.

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

Needs [uv](https://docs.astral.sh/uv/) and a Gemini API key.

```bash
uv sync
cp support_agent/.env.example support_agent/.env
# put your key in support_agent/.env -- get one at https://aistudio.google.com/apikey
```

Ask it something:

```bash
uv run python -m scripts.ask "checkout-api is slow this morning"
```

```
  -> tool: get_function_metrics({'function_name': 'checkout-api'})
  <- result: get_function_metrics returned
  -> tool: get_recent_logs({'function_name': 'checkout-api'})
  <- result: get_recent_logs returned

Based on the metrics and recent logs for `checkout-api`:

- **Metrics:** P99 of 9,850 ms and an error rate of 8.1% across 128,400 invocations.
- **Logs:** Task timeouts after 10 seconds and `ConnectTimeoutError` reaching the
  DynamoDB endpoint (`ap-northeast-1`). Memory is also near its ceiling, 498 MB
  of 512 MB.

**Probable Root Cause:** `checkout-api` is timing out while trying to reach
DynamoDB, which causes requests to hang near the 10-second timeout limit.

**Confidence:** High

[tokens] prompt=1869 response=248
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

### Known gap

On the `invoice-batch` ticket the agent gets the conclusion right — it reports
healthy telemetry and declines to invent a failure — but it pulls the logs anyway,
which the instruction tells it to do only when the metrics suggest a problem. So
the grounding rule holds and the cost-ordering rule doesn't, at least not
reliably. Instructions are not a control plane; you get compliance in proportion
to how much the model already wanted to comply.

That gap is what prompted everything in [Measuring it](#measuring-it) below.

## Model

Defaults to `gemini-3.8-flash`. Override either stage without touching the code:

```bash
SUPPORT_AGENT_MODEL=<model> uv run python -m scripts.ask "..."
SUPPORT_AGENT_CLASSIFIER_MODEL=<model> uv run python -m evals.run
```

Intake and triage are configured separately on purpose: classification is a
cheap fixed-output task and does not need the model that triage needs.

Be aware of the free-tier limit — 20 requests per day per model — if you plan to
run the eval suite more than once.

`GOOGLE_GENAI_USE_VERTEXAI=TRUE` in `.env` runs the same agent against Vertex AI
instead of the Gemini API directly.

## Layout

```
support_agent/
  classifier.py   intake: text-only classification, structured output
  pipeline.py     the gate: does this ticket earn a triage run?
  agent.py        triage: model, instruction, tool wiring
  policy.py       tool-access rules the agent cannot talk its way past
  mock_infra.py   the only file that knows what the backing systems are
scripts/
  ask.py          CLI runner -- prints the tool trace and token cost
evals/
  tickets.yaml    labelled tickets, expectations written before the run
  run.py          harness -- scores both stages, reports cost
```

## Measuring it

```bash
uv run python -m evals.run                    # whole suite
uv run python -m evals.run --only vague-en-04 # one ticket
uv run python -m evals.run --json out.json    # machine-readable
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

### What it found

Two defects, both of which had been invisible in ad-hoc testing:

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

### What changed as a result

Both fixes move the constraint somewhere the model can't reach:

- [`policy.py`](support_agent/policy.py) gates log retrieval on the metrics
  being out of band, and strips the list of valid names out of `not_found`
  results. The agent is told what it doesn't have, not what it could have had
  instead.
- [`pipeline.py`](support_agent/pipeline.py) makes the classifier's decision
  binding: if intake says there's nothing to look up, triage never runs.

`mock_infra.py` stays a faithful stand-in for the customer's systems and knows
about none of this.

### Honest status

The routing, the gate, and the `not_found` reshaping are unit-verified and pass.
The full suite has **not** been re-run end to end since those fixes: the Gemini
free tier allows 20 requests per day per model and the suite needs more than
that, so the numbers below are pending rather than claimed.

What is measured so far: the log gate works — on `invoice-batch` the agent still
*attempts* the log call but gets `skipped_by_policy` and answers from the metrics.
Note what that does and doesn't buy. Grounding is protected; the call count isn't
reduced, because the request still goes out. Cutting cost needs the intake gate,
not the tool gate.

## Where this is going

Roughly in the order the complexity is worth it:

- **Finish the baseline** — one complete suite run on a quota that allows it, so
  the pass rate and cost per ticket are numbers rather than intentions.
- **Retrieval over runbooks** — the log lines above are recognisable to someone
  who has seen them before. That knowledge lives in runbooks, not in weights.
- **Delegation** — split triage from remediation, so the agent that proposes a
  fix isn't the one that decided what's broken.
- **FAQ candidates from clusters** — tickets that recur are documentation debt.
  Grouping them by intake category over time is the cheapest way to see which
  guide page is missing.

## License

MIT
