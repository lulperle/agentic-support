"""Stand-in for a customer's existing infrastructure.

In a real Forward Deployed engagement these functions would call the
customer's own APIs -- an observability backend, a CMDB, a ticketing
system -- each behind its own auth and network boundary. Keeping them
in one module means the agent layer never changes when the real
integrations land: only this file does.
"""

from __future__ import annotations

# Structured data: the kind of thing that lives in an operational data store.
_INVOCATION_METRICS: dict[str, dict[str, float]] = {
    "checkout-api": {"invocations": 128_400, "error_rate": 0.081, "p99_ms": 9_850},
    "invoice-batch": {"invocations": 1_200, "error_rate": 0.002, "p99_ms": 2_100},
    "notify-fanout": {"invocations": 45_900, "error_rate": 0.311, "p99_ms": 640},
}

# Unstructured data: raw log lines, the messy input RAG has to cope with.
_LOG_LINES: dict[str, list[str]] = {
    "checkout-api": [
        "2026-09-01T02:11:04Z Task timed out after 10.00 seconds",
        "2026-09-01T02:11:04Z REPORT Duration: 10000.12 ms Billed Duration: 10000 ms Memory Size: 512 MB Max Memory Used: 498 MB",
        "2026-09-01T02:12:41Z [ERROR] ConnectTimeoutError: Connect timeout on endpoint URL: https://dynamodb.ap-northeast-1.amazonaws.com/",
    ],
    "invoice-batch": [
        "2026-09-01T18:03:00Z START RequestId: 7c1a Version: $LATEST",
        "2026-09-01T18:03:12Z Processed 4,102 invoices in 12.4s",
    ],
    "notify-fanout": [
        "2026-09-02T09:44:19Z [ERROR] AccessDenied: User is not authorized to perform sns:Publish",
        "2026-09-02T09:44:19Z [ERROR] AccessDenied: User is not authorized to perform sns:Publish",
        "2026-09-02T09:44:20Z Retrying (attempt 3/3)",
    ],
}


def get_function_metrics(function_name: str) -> dict:
    """Return invocation metrics for a serverless function.

    Args:
        function_name: Name of the function, e.g. "checkout-api".

    Returns:
        A dict with a "status" key. On success it also carries
        "invocations", "error_rate" and "p99_ms".
    """
    metrics = _INVOCATION_METRICS.get(function_name)
    if metrics is None:
        return {
            "status": "not_found",
            "error": f"No such function: {function_name}",
            "known_functions": sorted(_INVOCATION_METRICS),
        }
    return {"status": "ok", "function_name": function_name, **metrics}


def get_recent_logs(function_name: str) -> dict:
    """Return recent raw log lines for a serverless function.

    Args:
        function_name: Name of the function, e.g. "checkout-api".

    Returns:
        A dict with a "status" key and, on success, a "logs" list of
        raw log lines in chronological order.
    """
    logs = _LOG_LINES.get(function_name)
    if logs is None:
        return {
            "status": "not_found",
            "error": f"No logs retained for: {function_name}",
            "known_functions": sorted(_LOG_LINES),
        }
    return {"status": "ok", "function_name": function_name, "logs": logs}
