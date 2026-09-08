"""Tests for the ticket quality checks and the vendor scorecard.

The cases that matter are the ones where a naive implementation flatters the
operator: a ticket closed with no response at all, and a batch where excluding
the unanswered tickets would push attainment up.
"""

from __future__ import annotations

from datetime import datetime

from support_agent.quality import (
    HandledTicket,
    Severity,
    assess,
    scorecard,
)
from support_agent.sla import BusinessCalendar, Impact, Priority, Urgency

PLAIN = BusinessCalendar(holidays=frozenset(), closures=frozenset())

OPENED = datetime(2026, 9, 8, 10, 0)  # Tuesday morning


def well_handled(**overrides) -> HandledTicket:
    """A ticket with nothing wrong with it, for tests to spoil one field at a time."""
    defaults = dict(
        id="T-1",
        impact=Impact.SINGLE,
        urgency=Urgency.MEDIUM,  # -> P4
        opened_at=OPENED,
        first_response_at=datetime(2026, 9, 8, 10, 20),
        resolved_at=datetime(2026, 9, 8, 12, 0),
        reopened=False,
        resource_identified=True,
        evidence_cited=True,
        escalated=False,
        escalation_justified=None,
        category="performance_degradation",
    )
    defaults.update(overrides)
    return HandledTicket(**defaults)


class TestAssess:
    def test_clean_ticket_has_no_findings(self):
        result = assess(well_handled(), PLAIN)
        assert result.clean, result.findings
        assert result.priority == Priority.P4
        assert result.first_response_minutes == 20

    def test_first_response_breach_is_reported_with_numbers(self):
        # P4 owes a response in 480 business minutes; this one took two days.
        late = well_handled(first_response_at=datetime(2026, 9, 10, 17, 0))
        result = assess(late, PLAIN)
        codes = {f.code for f in result.findings}
        assert "first_response_breach" in codes
        breach = next(f for f in result.findings if f.code == "first_response_breach")
        assert "480" in breach.detail  # the target is stated, so it can be disputed
        assert result.has(Severity.BREACH)

    def test_missing_first_response_is_a_breach_not_a_gap(self):
        result = assess(well_handled(first_response_at=None), PLAIN)
        codes = {f.code for f in result.findings}
        assert "no_first_response" in codes
        assert result.has(Severity.BREACH)
        assert result.first_response_minutes is None

    def test_unresolved_is_process_not_breach(self):
        result = assess(well_handled(resolved_at=None), PLAIN)
        codes = {f.code for f in result.findings}
        assert "unresolved" in codes
        assert not any(f.code == "resolution_breach" for f in result.findings)

    def test_out_of_order_timestamps_are_flagged(self):
        broken = well_handled(first_response_at=datetime(2026, 9, 8, 9, 0))
        result = assess(broken, PLAIN)
        assert "timestamps_out_of_order" in {f.code for f in result.findings}

    def test_reopen_is_recorded(self):
        result = assess(well_handled(reopened=True), PLAIN)
        assert "reopened" in {f.code for f in result.findings}

    def test_unjustified_escalation_is_flagged(self):
        result = assess(
            well_handled(escalated=True, escalation_justified=False), PLAIN
        )
        assert "unjustified_escalation" in {f.code for f in result.findings}

    def test_justified_escalation_is_not_flagged(self):
        result = assess(well_handled(escalated=True, escalation_justified=True), PLAIN)
        assert "unjustified_escalation" not in {f.code for f in result.findings}

    def test_high_priority_handled_alone_is_surfaced(self):
        p1 = well_handled(
            impact=Impact.WIDESPREAD,
            urgency=Urgency.CRITICAL,
            first_response_at=datetime(2026, 9, 8, 10, 10),
            resolved_at=datetime(2026, 9, 8, 12, 0),
            escalated=False,
        )
        result = assess(p1, PLAIN)
        assert result.priority == Priority.P1
        assert "high_priority_not_escalated" in {f.code for f in result.findings}

    def test_hygiene_gaps_are_separate_from_breaches(self):
        sloppy = well_handled(
            resource_identified=False, evidence_cited=False, category=None
        )
        result = assess(sloppy, PLAIN)
        codes = {f.code for f in result.findings}
        assert {"resource_not_identified", "no_evidence", "uncategorised"} <= codes
        # Untidy is not the same as late.
        assert not result.has(Severity.BREACH)

    def test_weekend_does_not_manufacture_a_breach(self):
        # Raised Friday 17:00, answered Monday 10:00. On a wall clock that is
        # 65 hours; on the business clock it is inside a P4 target.
        friday = well_handled(
            opened_at=datetime(2026, 9, 11, 17, 0),
            first_response_at=datetime(2026, 9, 14, 10, 0),
            resolved_at=datetime(2026, 9, 14, 12, 0),
        )
        result = assess(friday, PLAIN)
        assert not result.has(Severity.BREACH), result.findings


class TestScorecard:
    def test_empty_batch_reports_nothing_rather_than_perfection(self):
        card = scorecard([], PLAIN)
        assert card.tickets == 0
        assert card.first_response_attainment is None
        assert card.resolution_attainment is None

    def test_unanswered_tickets_count_against_first_response_attainment(self):
        # The failure mode this guards: dropping tickets with no response from
        # the denominator, which turns the worst month into a perfect one.
        batch = [
            well_handled(id="T-1"),
            well_handled(id="T-2", first_response_at=None),
        ]
        card = scorecard(batch, PLAIN)
        assert card.first_response_attainment == 0.5
        assert "T-2" in card.breaching_tickets

    def test_resolution_attainment_excludes_unresolved(self):
        batch = [
            well_handled(id="T-1"),
            well_handled(id="T-2", resolved_at=None),
        ]
        card = scorecard(batch, PLAIN)
        # One resolved ticket, met its target.
        assert card.resolution_attainment == 1.0
        assert card.findings_by_code["unresolved"] == 1

    def test_rates_and_grouping(self):
        batch = [
            well_handled(id="T-1", reopened=True, evidence_cited=False),
            well_handled(id="T-2"),
            well_handled(
                id="T-3",
                impact=Impact.WIDESPREAD,
                urgency=Urgency.CRITICAL,
                first_response_at=datetime(2026, 9, 8, 10, 5),
                escalated=True,
                escalation_justified=True,
            ),
        ]
        card = scorecard(batch, PLAIN)
        assert card.tickets == 3
        assert card.reopen_rate == 1 / 3
        assert card.evidence_rate == 2 / 3
        assert card.by_priority == {"P4": 2, "P1": 1}
        assert card.as_dict()["tickets"] == 3
