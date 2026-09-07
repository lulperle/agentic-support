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
┌─────────────────────────────┐
│ triage_agent (LlmAgent)     │
│  instruction: grounding     │
│  rules + triage procedure   │
└──────────┬──────────────────┘
           │ tool-calling loop
     ┌─────┴──────┐
     ▼            ▼
get_function   get_recent
_metrics       _logs
     │            │
     └─────┬──────┘
           ▼
    mock_infra.py
  (stands in for the
   customer's own APIs)
```

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

This is the kind of thing the evaluation suite below exists to catch, and the
reason tool-gating eventually belongs in code rather than in a prompt.

## Model

Defaults to `gemini-3.8-flash`. Override without touching the code:

```bash
SUPPORT_AGENT_MODEL=gemini-3.8-pro uv run python -m scripts.ask "..."
```

`GOOGLE_GENAI_USE_VERTEXAI=TRUE` in `.env` runs the same agent against Vertex AI
instead of the Gemini API directly.

## Layout

```
support_agent/
  agent.py        the agent: model, instruction, tool wiring
  mock_infra.py   the only file that knows what the backing systems are
scripts/
  ask.py          CLI runner -- prints the tool trace and token cost
```

## Where this is going

Roughly in the order the complexity is worth it:

- **Delegation** — split triage from remediation, so the agent that proposes a
  fix isn't the one that decided what's broken.
- **Retrieval over runbooks** — the log lines above are recognisable to someone
  who has seen them before. That knowledge lives in runbooks, not in weights.
- **Evaluation** — a fixed set of tickets with known answers, including the
  tickets whose correct answer is "I don't know". Ungrounded confidence should
  fail the suite, not just read badly.

## License

MIT
