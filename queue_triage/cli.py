"""queue_triage -- rank a support queue by how cheap each ticket looks to close.

    uv run python -m queue_triage queue
    uv run python -m queue_triage queue --explain
    uv run python -m queue_triage queue 'category=credits' --explain
    uv run python -m queue_triage queue --min-score 5 --limit 10
    uv run python -m queue_triage queue --keep-excluded --no-aging
    uv run python -m queue_triage query 'queue=unassigned & -subject:"test case"'

The default backend reads the synthetic queue in `evals/queue.yaml`, so all of
the above runs offline with no credentials.

Two things happen to the queue before it is ranked, both from the ruleset rather
than from here: refunds and billing adjustments are dropped (`--keep-excluded`),
and a ticket that has been waiting gains a capped bonus (`--no-aging`). Neither is
part of the effort score -- see `queue_triage/score.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from . import emit, rules, score
from .backends import Backend, BackendError, FileBackend
from .query import QueryError, parse

DEFAULT_QUERY = 'queue=unassigned & language=en & -subject:"test case"'


def _resolve_ruleset(args) -> rules.Ruleset:
    return rules.load(args.ruleset) if args.ruleset else score.load_default()


def _now(args) -> datetime | None:
    """Resolve --now, which exists so a ranking can be reproduced.

    The bundled queue has fixed creation dates, so without a way to pin the clock
    the age bonuses -- and therefore the order, and therefore any output quoted in
    the README -- drift a little every week.
    """
    if not args.now:
        return None
    try:
        return datetime.fromisoformat(str(args.now).replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit(f"--now: not an ISO 8601 timestamp: {args.now}") from exc


def cmd_queue(args, backend: Backend) -> int:
    node = parse(args.query)
    ruleset = _resolve_ruleset(args)

    tickets = backend.search(node, limit=args.limit, sort_by=args.sort, descending=args.newest_first)
    if not tickets:
        print("no tickets matched", file=sys.stderr)
        return 1

    total = backend.count(node) if hasattr(backend, "count") else len(tickets)

    dropped: list[tuple[dict, object]] = []
    if not args.keep_excluded:
        tickets, dropped = score.split_excluded(tickets, ruleset)

    print(f"# {total} ticket(s) match; ranking {len(tickets)} with ruleset {ruleset.name!r}",
          file=sys.stderr)
    if dropped:
        counts: dict[str, int] = {}
        for _, rule in dropped:
            counts[rule.name] = counts.get(rule.name, 0) + 1
        summary = ", ".join(f"{n}x {name}" for name, n in sorted(counts.items()))
        print(f"# dropped {len(dropped)}: {summary}  (--keep-excluded to see them)",
              file=sys.stderr)
    print(file=sys.stderr)

    if not tickets:
        print("every matching ticket was excluded", file=sys.stderr)
        return 1

    ranked = score.rank(
        tickets,
        ruleset=ruleset,
        damping=not args.no_damping,
        aging=not args.no_aging,
        now=_now(args),
    )

    if args.json:
        print(json.dumps([
            {
                "id": ticket.get("id"),
                "subject": ticket.get("subject"),
                "score": verdict.score,
                "order_score": verdict.order_score,
                "label": verdict.label,
                "signals": {h.name: h.weight for h in verdict.hits},
                "adjustments": {a.name: a.weight for a in verdict.adjustments},
                "excluded_by": getattr(score.excluded_by(ticket, ruleset), "name", None),
            }
            for ticket, verdict in ranked if verdict.score >= args.min_score
        ], indent=2, ensure_ascii=False))
        return 0

    for ticket, verdict in ranked:
        if verdict.score < args.min_score:
            continue
        subject = str(ticket.get("subject") or "(no subject)").replace("\n", " ")
        # The leading number is what the list is sorted by, so the order can be
        # checked by reading down the column. Where age contributed, the effort
        # score is broken out after the subject -- that is the one the label comes
        # from, and the two disagreeing is the point of keeping them apart.
        if verdict.adjustments:
            parts = ", ".join(f"{a.weight:+d} {a.name}" for a in verdict.adjustments)
            tail = f"  ({verdict.score:+d} effort, {parts})"
        else:
            tail = ""
        print(f"[{verdict.order_score:+3d}] {verdict.label:<12} {ticket.get('id')}  "
              f"{subject[:72]}{tail}")
        if args.explain:
            for hit in sorted(verdict.hits, key=lambda h: -abs(h.weight)):
                detail = f" -- {hit.matched!r}" if hit.matched else ""
                note = f"  ({hit.note})" if hit.note else ""
                print(f"      {hit.weight:+3d} {hit.name}{detail}{note}")
            for adj in verdict.adjustments:
                print(f"      {adj.weight:+3d} {adj.name}  ({adj.note})")
            for rule in score.exclusions_for(ticket, ruleset):
                print(f"      x   would be excluded: {rule.name} -- {rule.reason}")
            if ticket.get("adversarial"):
                reason = " ".join(str(ticket.get("adversarial_note") or "").split())
                print(f"      !   known-hard example: {reason}")
        print()
    return 0


def cmd_query(args, backend: Backend) -> int:
    """Show what an expression parses to. The first thing to check when a filter
    silently matches nothing."""
    node = parse(args.query)
    print("parsed:  ", emit.to_text(node))
    print("negation:", emit.to_text(emit.push_not_down(node)))
    print("json:")
    print(json.dumps(emit.to_json(node), indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="queue_triage",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("query", nargs="?", default=DEFAULT_QUERY,
                       help=f"search expression (default: {DEFAULT_QUERY!r})")
        p.add_argument("--tickets", default=None, help="path to a ticket YAML file")
        p.add_argument("--ruleset", default=None, help="ruleset name or path")

    q = sub.add_parser("queue", help="rank the queue by estimated effort")
    common(q)
    q.add_argument("--limit", type=int, default=50,
                   help="tickets to fetch (default 50); exclusions apply after this")
    q.add_argument("--sort", default="created", help="field to page through by")
    q.add_argument("--newest-first", action="store_true",
                   help="default is oldest first, which is the order they age out in")
    q.add_argument("--explain", action="store_true", help="show which signals fired")
    q.add_argument("--min-score", type=int, default=-999,
                   help="hide tickets below this effort score (not the leading "
                        "number, which includes the age bonus)")
    q.add_argument("--no-damping", action="store_true",
                   help="score context-sensitive signals at full weight (see evals/rank.py)")
    q.add_argument("--no-aging", action="store_true",
                   help="rank on effort alone, ignoring how long a ticket has waited")
    q.add_argument("--keep-excluded", action="store_true",
                   help="also rank refunds and billing adjustments, which are dropped by default")
    q.add_argument("--now", default=None,
                   help="pretend it is this ISO 8601 time, for a reproducible age bonus")
    q.add_argument("--json", action="store_true", help="machine-readable output")
    q.set_defaults(func=cmd_queue)

    t = sub.add_parser("query", help="show how an expression parses")
    common(t)
    t.set_defaults(func=cmd_query)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    backend = FileBackend(args.tickets) if args.tickets else FileBackend()
    try:
        return args.func(args, backend)
    except QueryError as exc:
        print(f"bad query: {exc}", file=sys.stderr)
        return 2
    except rules.RulesetError as exc:
        print(f"bad ruleset: {exc}", file=sys.stderr)
        return 3
    except BackendError as exc:
        print(str(exc), file=sys.stderr)
        return 4
