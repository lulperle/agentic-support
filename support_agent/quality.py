"""Ticket quality checks and a vendor scorecard.

First-line tickets are handled by a contracted operator. The desk owner's job is
not to answer them, it is to know whether they were answered well -- and to know
it from the ticket record rather than from a monthly slide.

Every check here is deterministic and reads only fields already present in a
ticket. That is a deliberate limit. "Was the tone appropriate" would need a
judge model and would produce a number nobody can appeal; "was a first response
sent inside the committed window" produces a number the operator can verify and
dispute. Only the second kind belongs in a contract.

The severity split matters as much as the checks. A missed P1 first response and
a missing evidence link are both defects, and treating them as one number tells
the operator to fix whichever is cheaper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .sla import (
    DEFAULT_CALENDAR,
    SLA_TARGETS,
    BusinessCalendar,
    Impact,
    Priority,
    Urgency,
    commit,
)


class Severity(str, Enum):
    """How much a defect costs the customer, not how easy it is to fix."""

    BREACH = "breach"  # a commitment was missed
    PROCESS = "process"  # handled outside the agreed procedure
    HYGIENE = "hygiene"  # record is incomplete; recurrence cannot be studied


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    detail: str


@dataclass
class HandledTicket:
    """One closed ticket as the record leaves it.

    Optional timestamps are genuinely optional: a ticket can be closed without
    ever having had a first response, and that absence is exactly what the
    checks need to see. Defaulting them to `opened_at` would hide the worst
    cases.
    """

    id: str
    impact: Impact
    urgency: Urgency
    opened_at: datetime
    first_response_at: datetime | None = None
    resolved_at: datetime | None = None
    reopened: bool = False
    resource_identified: bool = False
    evidence_cited: bool = False
    escalated: bool = False
    escalation_justified: bool | None = None
    category: str | None = None


@dataclass
class Assessment:
    ticket_id: str
    priority: Priority
    findings: list[Finding] = field(default_factory=list)
    first_response_minutes: int | None = None
    resolution_minutes: int | None = None

    @property
    def clean(self) -> bool:
        return not self.findings

    def has(self, severity: Severity) -> bool:
        return any(f.severity is severity for f in self.findings)


def assess(
    ticket: HandledTicket, calendar: BusinessCalendar = DEFAULT_CALENDAR
) -> Assessment:
    """Check one handled ticket against its commitments and the procedure."""
    commitment = commit(ticket.impact, ticket.urgency, ticket.opened_at, calendar)
    target = SLA_TARGETS[commitment.priority]
    out = Assessment(ticket_id=ticket.id, priority=commitment.priority)

    # --- commitments -------------------------------------------------------
    if ticket.first_response_at is None:
        out.findings.append(
            Finding(
                "no_first_response",
                Severity.BREACH,
                "closed without any response to the customer",
            )
        )
    else:
        if ticket.first_response_at < ticket.opened_at:
            out.findings.append(
                Finding(
                    "timestamps_out_of_order",
                    Severity.HYGIENE,
                    "first response recorded before the ticket was opened",
                )
            )
        elapsed = calendar.minutes_between(ticket.opened_at, ticket.first_response_at)
        out.first_response_minutes = elapsed
        if elapsed > target.first_response_minutes:
            out.findings.append(
                Finding(
                    "first_response_breach",
                    Severity.BREACH,
                    f"{elapsed} business minutes against a "
                    f"{target.first_response_minutes} minute target "
                    f"({commitment.priority.value})",
                )
            )

    if ticket.resolved_at is None:
        out.findings.append(
            Finding("unresolved", Severity.PROCESS, "no resolution recorded")
        )
    else:
        elapsed = calendar.minutes_between(ticket.opened_at, ticket.resolved_at)
        out.resolution_minutes = elapsed
        if elapsed > target.resolution_minutes:
            out.findings.append(
                Finding(
                    "resolution_breach",
                    Severity.BREACH,
                    f"{elapsed} business minutes against a "
                    f"{target.resolution_minutes} minute target "
                    f"({commitment.priority.value})",
                )
            )

    # --- procedure ---------------------------------------------------------
    if ticket.reopened:
        out.findings.append(
            Finding(
                "reopened",
                Severity.PROCESS,
                "customer reopened it: the first resolution did not hold",
            )
        )

    if ticket.escalated and ticket.escalation_justified is False:
        out.findings.append(
            Finding(
                "unjustified_escalation",
                Severity.PROCESS,
                "escalated without meeting the escalation criteria",
            )
        )

    # A P1 or P2 that was never escalated is not automatically wrong, but it is
    # worth surfacing: those are the tickets the desk owner agreed to be told
    # about.
    if commitment.priority in (Priority.P1, Priority.P2) and not ticket.escalated:
        out.findings.append(
            Finding(
                "high_priority_not_escalated",
                Severity.PROCESS,
                f"{commitment.priority.value} handled entirely on the first line",
            )
        )

    # --- record hygiene ----------------------------------------------------
    if not ticket.resource_identified:
        out.findings.append(
            Finding(
                "resource_not_identified",
                Severity.HYGIENE,
                "no affected resource recorded, so recurrence cannot be counted",
            )
        )
    if not ticket.evidence_cited:
        out.findings.append(
            Finding(
                "no_evidence",
                Severity.HYGIENE,
                "resolution cites no metric or log, so it cannot be reviewed",
            )
        )
    if ticket.category is None:
        out.findings.append(
            Finding(
                "uncategorised",
                Severity.HYGIENE,
                "no category recorded, so it cannot be grouped for FAQ candidates",
            )
        )

    return out


@dataclass
class Scorecard:
    """Aggregate quality over a batch of tickets."""

    tickets: int
    by_priority: dict[str, int]
    first_response_attainment: float | None
    resolution_attainment: float | None
    reopen_rate: float
    evidence_rate: float
    findings_by_code: dict[str, int]
    breaching_tickets: list[str]

    def as_dict(self) -> dict:
        return {
            "tickets": self.tickets,
            "by_priority": self.by_priority,
            "first_response_attainment": self.first_response_attainment,
            "resolution_attainment": self.resolution_attainment,
            "reopen_rate": self.reopen_rate,
            "evidence_rate": self.evidence_rate,
            "findings_by_code": self.findings_by_code,
            "breaching_tickets": self.breaching_tickets,
        }


def scorecard(
    tickets: list[HandledTicket], calendar: BusinessCalendar = DEFAULT_CALENDAR
) -> Scorecard:
    """Summarise a batch. Attainment is reported per commitment, not blended.

    Attainment is `None` rather than 100% when nothing was measurable, because
    "we met every target we can find" and "there were no targets" are different
    facts and only one of them is good news.
    """
    if not tickets:
        return Scorecard(0, {}, None, None, 0.0, 0.0, {}, [])

    assessments = [assess(t, calendar) for t in tickets]

    by_priority: dict[str, int] = {}
    findings_by_code: dict[str, int] = {}
    for a in assessments:
        by_priority[a.priority.value] = by_priority.get(a.priority.value, 0) + 1
        for f in a.findings:
            findings_by_code[f.code] = findings_by_code.get(f.code, 0) + 1

    # Every ticket owed a first response, including the ones that never got
    # one, so the denominator is the whole batch. Excluding no_first_response
    # here would let the worst tickets improve the number.
    fr_codes = {"first_response_breach", "no_first_response"}
    fr_met = sum(
        1 for a in assessments if not any(f.code in fr_codes for f in a.findings)
    )

    # Resolution is scored only over tickets that were resolved. An unresolved
    # ticket is already reported as a PROCESS finding; counting it as a
    # resolution breach would double-penalise it and hide how long the
    # resolutions that did happen actually took.
    res_eligible = [a for a in assessments if a.resolution_minutes is not None]
    res_met = sum(
        1
        for a in res_eligible
        if not any(f.code == "resolution_breach" for f in a.findings)
    )

    return Scorecard(
        tickets=len(tickets),
        by_priority=by_priority,
        first_response_attainment=fr_met / len(assessments),
        resolution_attainment=res_met / len(res_eligible) if res_eligible else None,
        reopen_rate=sum(1 for t in tickets if t.reopened) / len(tickets),
        evidence_rate=sum(1 for t in tickets if t.evidence_cited) / len(tickets),
        findings_by_code=dict(sorted(findings_by_code.items())),
        breaching_tickets=[a.ticket_id for a in assessments if a.has(Severity.BREACH)],
    )
