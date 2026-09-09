"""The ruleset: weighted signals that make a ticket cheap or expensive to pick up.

A ruleset is data, not code -- see ``rulesets/cloud_support.yaml``. That is a
deliberate choice. The weights are guesses until they are measured, so they get
edited far more often than this file does, and a YAML file can be diffed against
an eval result without a code review.

Positive weight means "there is a known playbook and one reply usually finishes
it". Negative means "the answer depends on digging through the customer's
environment". The scoring itself lives in :mod:`queue_triage.score`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

RULESETS = Path(__file__).parent / "rulesets"
DEFAULT_RULESET = RULESETS / "cloud_support.yaml"


class RulesetError(ValueError):
    """Raised when a ruleset file is malformed."""


@dataclass(frozen=True)
class Signal:
    name: str
    weight: int
    pattern: re.Pattern[str]
    note: str = ""
    # Which field to match against; "" means all of the ticket's text joined.
    #
    # Anything measuring the *shape* of a field rather than its content has to say
    # so. `^.{0,45}$` against joined text can never match, because the join puts a
    # newline in the way and `.` does not cross one -- so the signal silently
    # never fires and looks exactly like a signal that simply did not apply. Both
    # structural signals in the shipped ruleset were dead this way until the eval
    # printed "never fired" next to them.
    scope: str = ""
    # Name of a context group whose presence halves this signal's weight, or "".
    #
    # This exists because a bare service name is not evidence of difficulty. A
    # customer disputing a bill names whichever line item surprised them, so
    # "transit gateway" in a refund request means something completely different
    # from "transit gateway" in a routing question. Without damping, a routine
    # refund sorts below a genuinely hard case.
    damped_by: str = ""


@dataclass(frozen=True)
class Exclusion:
    """A ticket shape to drop from the queue instead of ranking it.

    Distinct from a large negative weight on purpose. A penalty says "this looks
    expensive", and an expensive ticket still belongs in the list -- it just sorts
    to the bottom. An exclusion says "not mine to pick up at all", which no score
    can express: whatever the number, the ticket is still on screen.
    """

    name: str
    pattern: re.Pattern[str]
    reason: str = ""
    scope: str = ""


@dataclass(frozen=True)
class Aging:
    """How much a ticket gains for having waited, and the ceiling on that.

    Coarse and capped, both deliberately. Coarse because the point is only to
    break the tie between two similar-looking tickets in favour of the one that
    has been waiting -- an hour of extra age should not move anything. Capped
    because without a ceiling the oldest ticket in the queue eventually outranks
    everything regardless of effort, which is just first-in-first-out with extra
    arithmetic.

    ``days_per_point: 0`` disables aging, which is the default for a ruleset that
    does not mention it.
    """

    days_per_point: int = 0
    max_bonus: int = 0

    @property
    def enabled(self) -> bool:
        return self.days_per_point > 0 and self.max_bonus > 0

    def bonus(self, days: float) -> int:
        if not self.enabled or days <= 0:
            return 0
        return min(int(days // self.days_per_point), self.max_bonus)


@dataclass(frozen=True)
class Ruleset:
    signals: tuple[Signal, ...]
    context_groups: dict[str, frozenset[str]]
    severity_bonus: dict[str, int]
    vip_penalty: int
    thresholds: tuple[tuple[str, int], ...]   # (label, minimum score), best first
    name: str = "unnamed"
    exclusions: tuple[Exclusion, ...] = ()
    aging: Aging = Aging()

    def label_for(self, score: int) -> str:
        for label, minimum in self.thresholds:
            if score >= minimum:
                return label
        return self.thresholds[-1][0] if self.thresholds else "unlabelled"

    def signal(self, name: str) -> Signal:
        for signal in self.signals:
            if signal.name == name:
                return signal
        raise KeyError(name)


@dataclass
class Hit:
    """One signal firing on one case."""

    name: str
    weight: int
    note: str = ""
    matched: str = ""


@dataclass
class Verdict:
    score: int = 0
    hits: list[Hit] = field(default_factory=list)
    label: str = "unlabelled"
    # Adjustments that change where a ticket sorts without claiming anything about
    # how much work it is. Age is the only one so far.
    #
    # Kept out of `score` rather than folded into it, for two reasons. `label`
    # keeps meaning "how cheap this looks", so a ticket does not become a
    # "quick-win" by sitting in the queue for a fortnight. And the eval correlates
    # `score` against the effort labels, so an agreement number can never be
    # inflated by a mechanism that has nothing to do with effort -- which matters
    # here because the labelled set's creation dates run oldest-easiest by
    # accident of how it was written.
    adjustments: list[Hit] = field(default_factory=list)

    @property
    def order_score(self) -> int:
        """What the queue is actually sorted by."""
        return self.score + sum(adj.weight for adj in self.adjustments)

    def named(self, name: str) -> Hit | None:
        for hit in self.hits:
            if hit.name == name:
                return hit
        return None


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def _require(mapping: dict, key: str, where: str):
    if key not in mapping:
        raise RulesetError(f"{where}: missing {key!r}")
    return mapping[key]


def load(path: str | Path = DEFAULT_RULESET) -> Ruleset:
    """Read a ruleset from YAML, compiling every pattern up front.

    Patterns are compiled at load time so a typo in a regex fails immediately
    instead of on whichever case happens to reach it.
    """
    path = Path(path)
    if not path.exists() and not path.suffix:
        path = RULESETS / f"{path.name}.yaml"
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise RulesetError(f"no such ruleset: {path}") from exc
    if not isinstance(raw, dict):
        raise RulesetError(f"{path}: expected a mapping at the top level")

    groups = {
        name: frozenset(members)
        for name, members in (raw.get("context_groups") or {}).items()
    }

    signals: list[Signal] = []
    names: set[str] = set()
    for index, entry in enumerate(raw.get("signals") or []):
        where = f"{path}: signal #{index + 1}"
        name = _require(entry, "name", where)
        if name in names:
            raise RulesetError(f"{where}: duplicate signal name {name!r}")
        names.add(name)

        damped_by = entry.get("damped_by", "") or ""
        if damped_by and damped_by not in groups:
            raise RulesetError(f"{where}: damped_by {damped_by!r} is not a context group")

        try:
            pattern = re.compile(_require(entry, "pattern", where), re.IGNORECASE)
        except re.error as exc:
            raise RulesetError(f"{where}: bad pattern -- {exc}") from exc

        signals.append(Signal(
            name=name,
            weight=int(_require(entry, "weight", where)),
            pattern=pattern,
            note=entry.get("note", "") or "",
            scope=entry.get("scope", "") or "",
            damped_by=damped_by,
        ))

    unknown = {m for members in groups.values() for m in members} - names
    if unknown:
        raise RulesetError(f"{path}: context_groups name unknown signals: {sorted(unknown)}")

    exclusions: list[Exclusion] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw.get("exclusions") or []):
        where = f"{path}: exclusion #{index + 1}"
        name = _require(entry, "name", where)
        if name in seen:
            raise RulesetError(f"{where}: duplicate exclusion name {name!r}")
        seen.add(name)
        try:
            pattern = re.compile(_require(entry, "pattern", where), re.IGNORECASE)
        except re.error as exc:
            raise RulesetError(f"{where}: bad pattern -- {exc}") from exc
        exclusions.append(Exclusion(
            name=name,
            pattern=pattern,
            reason=entry.get("reason", "") or "",
            scope=entry.get("scope", "") or "",
        ))

    aging_raw = raw.get("aging") or {}
    if not isinstance(aging_raw, dict):
        raise RulesetError(f"{path}: aging must be a mapping")
    aging = Aging(
        days_per_point=int(aging_raw.get("days_per_point", 0) or 0),
        max_bonus=int(aging_raw.get("max_bonus", 0) or 0),
    )
    if aging.days_per_point < 0 or aging.max_bonus < 0:
        raise RulesetError(f"{path}: aging values cannot be negative")
    # Half-configured aging is worse than none: it reads as enabled everywhere it
    # is discussed and does nothing at all, which is exactly how the two
    # structurally-dead signals hid for as long as they did.
    if bool(aging.days_per_point) != bool(aging.max_bonus):
        raise RulesetError(
            f"{path}: aging needs both days_per_point and max_bonus, or neither"
        )

    thresholds = tuple(
        (str(_require(t, "label", f"{path}: thresholds")), int(_require(t, "min", f"{path}: thresholds")))
        for t in (raw.get("thresholds") or [])
    )
    if sorted(thresholds, key=lambda t: -t[1]) != list(thresholds):
        raise RulesetError(f"{path}: thresholds must be listed highest score first")

    return Ruleset(
        signals=tuple(signals),
        context_groups=groups,
        severity_bonus={str(k).lower(): int(v) for k, v in (raw.get("severity_bonus") or {}).items()},
        vip_penalty=int(raw.get("vip_penalty", 0)),
        thresholds=thresholds,
        name=raw.get("name") or path.stem,
        exclusions=tuple(exclusions),
        aging=aging,
    )
