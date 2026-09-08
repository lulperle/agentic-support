"""Tests for priority and the business-hours clock.

The clock tests are the ones worth writing. The priority matrix is a lookup
table and will fail loudly if it is wrong; an off-by-one in business-minute
arithmetic fails quietly, as a breach report about a ticket that never breached.
"""

from __future__ import annotations

from datetime import date, datetime, time

import pytest

from support_agent.sla import (
    SLA_TARGETS,
    BusinessCalendar,
    Impact,
    Priority,
    Urgency,
    commit,
    escalate,
    prioritise,
    urgency_from_classifier,
)

# A calendar with no holidays, so weekday arithmetic can be tested on its own.
PLAIN = BusinessCalendar(holidays=frozenset(), closures=frozenset())


class TestPriority:
    def test_matrix_is_total(self):
        for impact in Impact:
            for urgency in Urgency:
                assert isinstance(prioritise(impact, urgency), Priority)

    def test_impact_outweighs_self_reported_urgency(self):
        # One tenant shouting is not the same as a shared component degrading.
        assert prioritise(Impact.SINGLE, Urgency.CRITICAL) == Priority.P2
        assert prioritise(Impact.WIDESPREAD, Urgency.MEDIUM) == Priority.P2

    def test_targets_tighten_with_priority(self):
        order = [Priority.P1, Priority.P2, Priority.P3, Priority.P4]
        response = [SLA_TARGETS[p].first_response_minutes for p in order]
        resolution = [SLA_TARGETS[p].resolution_minutes for p in order]
        assert response == sorted(response)
        assert resolution == sorted(resolution)

    def test_classifier_urgency_values_stay_in_sync(self):
        # The AI stage and the policy stage share these strings so no
        # translation table is needed. If either enum drifts, deadlines start
        # being derived from a value nobody validated.
        from support_agent.classifier import UrgencyClaim

        assert {u.value for u in UrgencyClaim} == {u.value for u in Urgency}

    @pytest.mark.parametrize(
        "value", ["critical", "high", "medium", "low"]
    )
    def test_classifier_urgency_casts(self, value):
        assert urgency_from_classifier(value).value == value

    @pytest.mark.parametrize("bad", ["urgent", "", "1", None])
    def test_bad_urgency_raises_rather_than_defaulting(self, bad):
        # A silently defaulted urgency becomes a quietly wrong deadline.
        with pytest.raises(ValueError):
            urgency_from_classifier(bad)


class TestBusinessClock:
    def test_minutes_within_one_open_day(self):
        start = datetime(2026, 9, 8, 10, 0)  # Tuesday
        end = datetime(2026, 9, 8, 11, 30)
        assert PLAIN.minutes_between(start, end) == 90

    def test_time_before_opening_does_not_count(self):
        start = datetime(2026, 9, 8, 7, 0)
        end = datetime(2026, 9, 8, 10, 0)
        assert PLAIN.minutes_between(start, end) == 30  # 09:30 to 10:00

    def test_time_after_closing_does_not_count(self):
        start = datetime(2026, 9, 8, 18, 0)
        end = datetime(2026, 9, 8, 23, 0)
        assert PLAIN.minutes_between(start, end) == 15  # 18:00 to 18:15

    def test_overnight_gap_is_not_elapsed_time(self):
        # 17:00 Tuesday to 10:00 Wednesday is 75 + 30 business minutes, not 17h.
        start = datetime(2026, 9, 8, 17, 0)
        end = datetime(2026, 9, 9, 10, 0)
        assert PLAIN.minutes_between(start, end) == 75 + 30

    def test_weekend_stops_the_clock(self):
        friday_late = datetime(2026, 9, 11, 18, 0)
        monday_early = datetime(2026, 9, 14, 10, 0)
        assert PLAIN.minutes_between(friday_late, monday_early) == 15 + 30

    def test_holiday_stops_the_clock(self):
        # Mechanism, not calendar: an arbitrary Wednesday declared closed.
        closed = BusinessCalendar(
            holidays=frozenset({date(2026, 9, 9)}), closures=frozenset()
        )
        start = datetime(2026, 9, 8, 18, 0)
        end = datetime(2026, 9, 10, 10, 0)
        assert closed.minutes_between(start, end) == 15 + 30
        assert not closed.is_open(date(2026, 9, 9))

    def test_year_end_closure_is_closed(self):
        default = BusinessCalendar()
        assert not default.is_open(date(2025, 12, 30))
        assert not default.is_open(date(2026, 1, 2))

    def test_reversed_pair_is_zero_not_negative(self):
        later = datetime(2026, 9, 8, 12, 0)
        earlier = datetime(2026, 9, 8, 10, 0)
        assert PLAIN.minutes_between(later, earlier) == 0


class TestDeadlines:
    def test_deadline_within_the_same_day(self):
        opened = datetime(2026, 9, 8, 10, 0)
        assert PLAIN.add_minutes(opened, 30) == datetime(2026, 9, 8, 10, 30)

    def test_deadline_rolls_to_next_open_day(self):
        opened = datetime(2026, 9, 8, 18, 0)  # 15 business minutes left
        # 30 minutes owed: 15 today, 15 from tomorrow's opening.
        assert PLAIN.add_minutes(opened, 30) == datetime(2026, 9, 9, 9, 45)

    def test_clock_starts_at_opening_for_out_of_hours_tickets(self):
        opened = datetime(2026, 9, 8, 3, 0)  # raised at 3am
        assert PLAIN.add_minutes(opened, 30) == datetime(2026, 9, 8, 10, 0)

    def test_friday_evening_p1_is_not_due_over_the_weekend(self):
        # The case that produces phantom breaches if the clock is naive.
        opened = datetime(2026, 9, 11, 17, 0)  # Friday
        commitment = commit(Impact.WIDESPREAD, Urgency.CRITICAL, opened, PLAIN)
        assert commitment.priority == Priority.P1
        assert commitment.first_response_due == datetime(2026, 9, 11, 17, 30)
        # Four hours of resolution time cannot fit before 18:15 on Friday.
        assert commitment.resolution_due.date() == date(2026, 9, 14)

    def test_commit_uses_the_matrix(self):
        opened = datetime(2026, 9, 8, 10, 0)
        assert commit(Impact.SINGLE, Urgency.LOW, opened, PLAIN).priority == Priority.P4


class TestEscalation:
    def test_escalation_raises_one_level(self):
        opened = datetime(2026, 9, 8, 10, 0)
        c = commit(Impact.SINGLE, Urgency.MEDIUM, opened, PLAIN)  # P4
        assert escalate(c).priority == Priority.P3

    def test_p1_cannot_escalate_further(self):
        opened = datetime(2026, 9, 8, 10, 0)
        c = commit(Impact.WIDESPREAD, Urgency.CRITICAL, opened, PLAIN)
        assert escalate(c) == c

    def test_escalation_does_not_shorten_promised_deadlines(self):
        opened = datetime(2026, 9, 8, 10, 0)
        c = commit(Impact.SINGLE, Urgency.MEDIUM, opened, PLAIN)
        assert escalate(c).first_response_due == c.first_response_due
