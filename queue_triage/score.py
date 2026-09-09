"""Score a ticket for how cheap it looks to pick up.

The output is a :class:`~queue_triage.rules.Verdict`: a number, a label, and the
list of signals that produced it. The list is the point. A ranker that cannot say
why it ranked something is a ranker nobody will trust twice, and in practice the
per-signal breakdown is what catches bad rules -- every false positive fixed in
the shipped ruleset was found by reading it or by the eval's per-signal report,
not by writing a test first.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import rules
from .rules import Exclusion, Hit, Ruleset, Signal, Verdict

# Fields whose text is worth scoring, in the order they get joined.
SCORED_FIELDS = ("subject", "topic", "category", "item", "correspondence")

# The key under which the joined text lives. A signal with no scope matches this.
ALL = "*"


def fields_of(case: dict, body: str | None = None) -> dict[str, str]:
    """Build the per-field text a ruleset can match against.

    Signals are scoped to a field because some of them measure a field's shape
    rather than its content, and shape does not survive concatenation.
    """
    fields = {f: str(case.get(f) or "") for f in SCORED_FIELDS}
    if body is not None:
        fields["correspondence"] = str(body)
    fields[ALL] = "\n".join(v for v in (fields[f] for f in SCORED_FIELDS) if v)
    return fields


def _apply(fields: dict[str, str], signals, verdict: Verdict, active_groups: set[str]) -> None:
    for signal in signals:
        text = fields.get(signal.scope or ALL, "")
        if not text:
            continue
        match = signal.pattern.search(text)
        if not match:
            continue

        weight = signal.weight
        note = signal.note
        if signal.damped_by and signal.damped_by in active_groups:
            # Halve rather than drop: the signal still carries some information,
            # just much less than its face value in this context.
            weight = round(weight / 2)
            note = f"{note}; damped by {signal.damped_by} context".lstrip("; ")

        verdict.score += weight
        verdict.hits.append(Hit(signal.name, weight, note, match.group(0)[:60]))


def score_fields(
    fields: dict[str, str],
    ruleset: Ruleset | None = None,
    damping: bool = True,
) -> Verdict:
    """Score already-split ticket text.

    Runs in two passes. The first scores every signal that is not damped by
    anything, which establishes which context groups are active; the second scores
    the rest, now knowing the context. One pass cannot do this, because whether
    ``deep-network`` should count depends on whether ``credits`` fired and the
    ruleset does not promise an order.

    ``damping=False`` disables the adjustment. That exists for the eval harness:
    the only honest way to claim the mechanism helps is to measure both arms.
    """
    ruleset = ruleset or load_default()
    verdict = Verdict()

    plain: list[Signal] = [s for s in ruleset.signals if not s.damped_by]
    dampable: list[Signal] = [s for s in ruleset.signals if s.damped_by]

    _apply(fields, plain, verdict, active_groups=set())

    fired = {hit.name for hit in verdict.hits}
    active: set[str] = set()
    if damping:
        active = {
            group for group, members in ruleset.context_groups.items()
            if fired & members
        }
    _apply(fields, dampable, verdict, active_groups=active)

    verdict.label = ruleset.label_for(verdict.score)
    return verdict


def score_text(text: str, ruleset: Ruleset | None = None, damping: bool = True) -> Verdict:
    """Score a bare string -- a subject line, or a subject plus the first message.

    The text is registered as both the subject and the whole ticket, so
    subject-scoped signals still apply. Prefer :func:`score_case` when you have a
    real ticket; this is for tests and for scoring text with no structure.
    """
    return score_fields({ALL: text, "subject": text}, ruleset, damping=damping)


def score_case(
    case: dict,
    body: str | None = None,
    ruleset: Ruleset | None = None,
    damping: bool = True,
) -> Verdict:
    """Score a ticket dict, optionally overriding the customer's opening message.

    Only the first inbound message belongs in ``body``. The whole point is to
    judge a ticket nobody has touched yet, so anything after it does not exist.
    """
    ruleset = ruleset or load_default()
    verdict = score_fields(fields_of(case, body), ruleset, damping=damping)

    severity = str(case.get("severity") or "").lower()
    for name, bonus in ruleset.severity_bonus.items():
        if name and name in severity:
            verdict.score += bonus
            verdict.hits.append(Hit(f"severity:{name}", bonus))
            break

    if case.get("is_vip") and ruleset.vip_penalty:
        verdict.score += ruleset.vip_penalty
        verdict.hits.append(Hit("vip-account", ruleset.vip_penalty, "extra scrutiny"))

    verdict.label = ruleset.label_for(verdict.score)
    return verdict


# --------------------------------------------------------------------------- #
# filtering: what not to rank at all
# --------------------------------------------------------------------------- #

def exclusions_for(
    case: dict,
    ruleset: Ruleset | None = None,
    body: str | None = None,
) -> list[Exclusion]:
    """Every exclusion this ticket trips, in ruleset order.

    Matched against the ticket's metadata as well as its text, so a queue that
    categorises refunds properly gets filtered on the category and one that does
    not still gets filtered on the wording.

    All of them rather than the first, because they overlap: a refund request
    filed as a courtesy adjustment trips both, and reporting only whichever
    happens to be listed first gives the ticket a reason the reader will not
    recognise. It is also the only way to tell an exclusion that has no example in
    the queue from one that is merely shadowed by another.
    """
    ruleset = ruleset or load_default()
    fields = fields_of(case, body)
    hits = []
    for rule in ruleset.exclusions:
        text = fields.get(rule.scope or ALL, "")
        if text and rule.pattern.search(text):
            hits.append(rule)
    return hits


def excluded_by(
    case: dict,
    ruleset: Ruleset | None = None,
    body: str | None = None,
) -> Exclusion | None:
    """The first exclusion this ticket trips, or ``None``."""
    return next(iter(exclusions_for(case, ruleset, body)), None)


def split_excluded(
    cases,
    ruleset: Ruleset | None = None,
) -> tuple[list[dict], list[tuple[dict, Exclusion]]]:
    """Split a queue into what to rank and what to drop, keeping the reason.

    Filtering deliberately does not happen inside :func:`rank`. Scoring is the
    part with a measurable claim attached, and filtering is a preference about
    what the person working the queue wants to see -- so the eval can measure the
    scorer over tickets that never reach the screen, and a change of mind about
    what to hide cannot move a published number.

    The reason travels with the dropped ticket because a ticket that vanishes
    without explanation is indistinguishable from a bug in the query.
    """
    ruleset = ruleset or load_default()
    keep: list[dict] = []
    dropped: list[tuple[dict, Exclusion]] = []
    for case in cases:
        rule = excluded_by(case, ruleset)
        if rule:
            dropped.append((case, rule))
        else:
            keep.append(case)
    return keep, dropped


# --------------------------------------------------------------------------- #
# age
# --------------------------------------------------------------------------- #

def age_days(case: dict, now: datetime | None = None) -> float:
    """Days since the ticket was opened; 0 if it has no usable timestamp.

    An unparseable date reads as brand new rather than raising. Age is a
    tiebreaker, and one malformed field should not stop the queue from ranking.
    """
    raw = str(case.get("created") or "")
    if not raw:
        return 0.0
    try:
        created = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return max((now - created).total_seconds() / 86400.0, 0.0)


# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #

def rank(
    cases,
    ruleset: Ruleset | None = None,
    damping: bool = True,
    aging: bool = True,
    now: datetime | None = None,
) -> list[tuple[dict, Verdict]]:
    """Score every ticket and return them cheapest-looking first.

    Two things push an older ticket up. The aging bonus, capped by the ruleset, is
    the one that can overcome a small difference in effort; and the sort falls back
    to the creation date before the id, so among tickets that look equally cheap
    the one that has been waiting longest is offered first. Age lands in
    ``Verdict.adjustments`` rather than ``Verdict.score``, so the ticket's label
    still describes the work rather than the wait.

    Ties finally break on id, because an unstable ranking makes every eval diff
    unreadable.
    """
    ruleset = ruleset or load_default()
    now = now or datetime.now(timezone.utc)

    rows = []
    for case in cases:
        verdict = score_case(case, ruleset=ruleset, damping=damping)
        days = age_days(case, now)
        if aging:
            bonus = ruleset.aging.bonus(days)
            if bonus:
                verdict.adjustments.append(
                    Hit(f"waited-{int(days)}d", bonus, "aging, order only")
                )
        rows.append((case, verdict, days))

    rows.sort(key=lambda row: (-row[1].order_score, -row[2], str(row[0].get("id") or "")))
    return [(case, verdict) for case, verdict, _ in rows]


_DEFAULT: Ruleset | None = None


def load_default() -> Ruleset:
    """Load and cache the bundled ruleset."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = rules.load()
    return _DEFAULT
