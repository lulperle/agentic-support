"""Priority, business-hours clocks and SLA targets.

The intake classifier decides what a ticket *is*. This module decides what the
helpdesk *owes* the customer as a result: a priority, a first-response deadline
and a resolution deadline, all measured on a clock that stops when the desk is
closed.

There is no model in here. Priority and deadlines are policy, they have to be
identical for two identical tickets, and they have to be explainable to the
person whose ticket was deprioritised. That makes them a lookup table and some
date arithmetic, not an inference.

The clock is the part that is easy to get wrong and expensive to get wrong. A
ticket raised at 17:00 on the Friday before a long weekend has not breached a
four-hour target by Monday morning, and a desk that reports it as breached will
spend its improvement effort on a phantom.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta
from enum import Enum


class Impact(str, Enum):
    """How much of the estate is affected."""

    WIDESPREAD = "widespread"  # many tenants or a shared component
    MULTIPLE = "multiple"  # more than one system, one tenant
    SINGLE = "single"  # one system


class Urgency(str, Enum):
    """How fast the customer needs it, as claimed at intake."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Priority(str, Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


# Impact is deliberately weighted above urgency. A single tenant calling their
# own problem critical does not outrank a shared component degrading quietly:
# self-reported urgency is the least reliable signal at intake.
PRIORITY_MATRIX: dict[tuple[Impact, Urgency], Priority] = {
    (Impact.WIDESPREAD, Urgency.CRITICAL): Priority.P1,
    (Impact.WIDESPREAD, Urgency.HIGH): Priority.P1,
    (Impact.WIDESPREAD, Urgency.MEDIUM): Priority.P2,
    (Impact.WIDESPREAD, Urgency.LOW): Priority.P3,
    (Impact.MULTIPLE, Urgency.CRITICAL): Priority.P1,
    (Impact.MULTIPLE, Urgency.HIGH): Priority.P2,
    (Impact.MULTIPLE, Urgency.MEDIUM): Priority.P3,
    (Impact.MULTIPLE, Urgency.LOW): Priority.P3,
    (Impact.SINGLE, Urgency.CRITICAL): Priority.P2,
    (Impact.SINGLE, Urgency.HIGH): Priority.P3,
    (Impact.SINGLE, Urgency.MEDIUM): Priority.P4,
    (Impact.SINGLE, Urgency.LOW): Priority.P4,
}


def prioritise(impact: Impact, urgency: Urgency) -> Priority:
    """Map impact and urgency onto a priority. Total over the enums by design."""
    return PRIORITY_MATRIX[(impact, urgency)]


def urgency_from_classifier(value: str) -> Urgency:
    """Convert the classifier's urgency claim into an Urgency.

    The values are deliberately identical to `classifier.UrgencyClaim`, so this
    is a validating cast rather than a translation.

    Raises:
        ValueError: on anything unrecognised, rather than silently defaulting.
            A bad urgency should surface at the boundary, not become a quietly
            wrong deadline further down.
    """
    try:
        return Urgency(value)
    except ValueError:
        valid = ", ".join(u.value for u in Urgency)
        raise ValueError(f"urgency must be one of {valid}; got {value!r}") from None


@dataclass(frozen=True)
class Target:
    """What a priority owes, in business minutes."""

    first_response_minutes: int
    resolution_minutes: int


SLA_TARGETS: dict[Priority, Target] = {
    Priority.P1: Target(first_response_minutes=30, resolution_minutes=240),
    Priority.P2: Target(first_response_minutes=60, resolution_minutes=480),
    Priority.P3: Target(first_response_minutes=240, resolution_minutes=1_440),
    Priority.P4: Target(first_response_minutes=480, resolution_minutes=2_400),
}


# National holidays. Hardcoded here so the module has no network dependency and
# the tests are deterministic, but this is the one piece of data in the file
# that must not be trusted: a real deployment reads the published calendar
# rather than a list someone typed. The tests below exercise the mechanism --
# a closed day stops the clock -- not the correctness of any single date.
JP_HOLIDAYS_2026: frozenset[date] = frozenset(
    {
        date(2026, 1, 1),
        date(2026, 1, 12),
        date(2026, 2, 11),
        date(2026, 2, 23),
        date(2026, 3, 20),
        date(2026, 4, 29),
        date(2026, 5, 3),
        date(2026, 5, 4),
        date(2026, 5, 5),
        date(2026, 5, 6),
        date(2026, 7, 20),
        date(2026, 8, 11),
        date(2026, 9, 21),
        date(2026, 9, 22),
        date(2026, 9, 23),
        date(2026, 10, 12),
        date(2026, 11, 3),
        date(2026, 11, 23),
    }
)


def _year_end_closure(year: int) -> set[date]:
    """29 December to 3 January inclusive, the published shutdown."""
    return {date(year, 12, d) for d in (29, 30, 31)} | {
        date(year + 1, 1, d) for d in (1, 2, 3)
    }


@dataclass(frozen=True)
class BusinessCalendar:
    """When the desk is open, and therefore when SLA clocks run."""

    open_time: time = time(9, 30)
    close_time: time = time(18, 15)
    holidays: frozenset[date] = field(default=JP_HOLIDAYS_2026)
    closures: frozenset[date] = field(
        default_factory=lambda: frozenset(_year_end_closure(2025) | _year_end_closure(2026))
    )

    def is_open(self, day: date) -> bool:
        """Whether the desk operates on this calendar day."""
        return (
            day.weekday() < 5
            and day not in self.holidays
            and day not in self.closures
        )

    def _window(self, day: date) -> tuple[datetime, datetime]:
        return (
            datetime.combine(day, self.open_time),
            datetime.combine(day, self.close_time),
        )

    def minutes_between(self, start: datetime, end: datetime) -> int:
        """Business minutes elapsed between two instants.

        Returns 0 rather than a negative number if `end` precedes `start`; an
        out-of-order pair is a data problem for the quality checks to report,
        not something to encode as negative elapsed time.
        """
        if end <= start:
            return 0

        total = timedelta()
        day = start.date()
        while day <= end.date():
            if self.is_open(day):
                opens, closes = self._window(day)
                overlap_start = max(start, opens)
                overlap_end = min(end, closes)
                if overlap_end > overlap_start:
                    total += overlap_end - overlap_start
            day += timedelta(days=1)
        return int(total.total_seconds() // 60)

    def add_minutes(self, start: datetime, minutes: int) -> datetime:
        """The instant `minutes` business minutes after `start`.

        Used for deadlines. A ticket raised while the desk is closed starts its
        clock at the next opening, not immediately.
        """
        remaining = timedelta(minutes=minutes)
        cursor = start
        day = start.date()

        # Walk forward a day at a time, spending the budget in each open window.
        # Bounded so a pathological calendar cannot loop forever.
        for _ in range(400):
            if self.is_open(day):
                opens, closes = self._window(day)
                cursor = max(cursor, opens)
                available = closes - cursor
                if available >= remaining:
                    return cursor + remaining
                if available > timedelta():
                    remaining -= available
            day += timedelta(days=1)
            cursor = datetime.combine(day, self.open_time)
        raise RuntimeError("no open business day found within 400 days")


DEFAULT_CALENDAR = BusinessCalendar()


@dataclass(frozen=True)
class Commitment:
    """The deadlines a ticket is held to."""

    priority: Priority
    opened_at: datetime
    first_response_due: datetime
    resolution_due: datetime


def commit(
    impact: Impact,
    urgency: Urgency,
    opened_at: datetime,
    calendar: BusinessCalendar = DEFAULT_CALENDAR,
) -> Commitment:
    """Derive priority and both deadlines for a newly opened ticket."""
    priority = prioritise(impact, urgency)
    target = SLA_TARGETS[priority]
    return Commitment(
        priority=priority,
        opened_at=opened_at,
        first_response_due=calendar.add_minutes(
            opened_at, target.first_response_minutes
        ),
        resolution_due=calendar.add_minutes(opened_at, target.resolution_minutes),
    )


def escalate(commitment: Commitment) -> Commitment:
    """Raise a commitment one priority level, recomputing nothing else.

    Deadlines are intentionally *not* recomputed from the new priority: the
    clock a customer was promised does not get shorter retroactively. What
    escalation changes is who is looking at it.
    """
    order = [Priority.P1, Priority.P2, Priority.P3, Priority.P4]
    index = order.index(commitment.priority)
    if index == 0:
        return commitment
    return replace(commitment, priority=order[index - 1])
