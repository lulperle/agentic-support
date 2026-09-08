"""Tool-access policy, enforced in code rather than in the prompt.

The triage instruction tells the agent to read metrics first and to pull logs
only when the metrics suggest something is actually wrong. Measurement showed
it does not reliably obey: on a healthy function it fetched the logs anyway,
which is the expensive call.

Instructions are not a control plane. You get compliance in proportion to how
much the model already wanted to comply, and "skip the step that would let me
answer more confidently" is not something a model wants to do. So the ordering
rule lives here, as a gate the agent cannot talk its way past.

`mock_infra` stays a faithful stand-in for the customer's systems and knows
nothing about our policy. This module is the only place the policy exists.
"""

from __future__ import annotations

from . import mock_infra

# A function is worth reading logs for if either signal is out of band.
# Deliberately loose: the cost of a false positive is one extra tool call,
# the cost of a false negative is a missed outage.
ERROR_RATE_CEILING = 0.01
P99_CEILING_MS = 3_000


def _strip_substitution_bait(result: dict) -> dict:
    """Remove the list of valid names from a not_found result.

    Measured defect: asked about `payment-api`, which does not exist, the agent
    received `known_functions: [checkout-api, invoice-batch, notify-fanout]`,
    picked `checkout-api`, and reported its DynamoDB timeouts as the answer --
    a correct diagnosis of a system nobody asked about.

    The list is a reasonable thing for a real API to return to a developer. It
    is a trap to hand to a model that is trying to be useful, because the
    nearest valid name is always right there. So the agent is told what it does
    not have, and not what it could have had instead.
    """
    if result.get("status") != "not_found":
        return result
    return {
        "status": "not_found",
        "error": result.get("error", "no such function"),
        "guidance": (
            "This resource does not exist in the telemetry backend. Do not answer "
            "about a different resource. Tell the customer the name was not found "
            "and ask them to confirm it."
        ),
    }


def looks_unhealthy(metrics: dict) -> bool:
    """Whether metrics justify the more expensive log retrieval."""
    return (
        metrics.get("error_rate", 0.0) > ERROR_RATE_CEILING
        or metrics.get("p99_ms", 0.0) > P99_CEILING_MS
    )


def get_function_metrics(function_name: str) -> dict:
    """Return invocation metrics for a serverless function.

    Always the first call for a ticket: it is the cheap signal that decides
    whether the expensive one is warranted.

    Args:
        function_name: Name of the function, e.g. "checkout-api".

    Returns:
        A dict with a "status" key. On success it also carries
        "invocations", "error_rate" and "p99_ms".
    """
    return _strip_substitution_bait(mock_infra.get_function_metrics(function_name))


def get_recent_logs(function_name: str) -> dict:
    """Return recent raw log lines, if the function's metrics warrant it.

    Refuses when the metrics are within normal range, so that healthy tickets
    cost one tool call instead of two.

    Args:
        function_name: Name of the function, e.g. "checkout-api".

    Returns:
        A dict with a "status" key: "ok" with a "logs" list, "not_found",
        or "skipped_by_policy" when the metrics look healthy.
    """
    metrics = mock_infra.get_function_metrics(function_name)
    if metrics.get("status") != "ok":
        return metrics

    if not looks_unhealthy(metrics):
        return {
            "status": "skipped_by_policy",
            "function_name": function_name,
            "reason": (
                "Metrics are within normal range "
                f"(error_rate={metrics['error_rate']}, p99_ms={metrics['p99_ms']}), "
                "so logs were not retrieved. Report the function as healthy on the "
                "strength of the metrics. Do not infer a fault from the absence "
                "of logs."
            ),
        }

    return mock_infra.get_recent_logs(function_name)
