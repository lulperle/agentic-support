"""Tests for the ruleset loader and the effort scoring.

The subject lines here are written by hand. None of them are copied from a real
ticket, and the fixtures in `evals/queue.yaml` are invented too -- see the note at
the top of that file.
"""

import textwrap
from datetime import datetime, timedelta, timezone

import pytest

from queue_triage import rules, score


@pytest.fixture(scope="module")
def ruleset():
    return rules.load()


def ruleset_from(body: str, tmp_path) -> rules.Ruleset:
    path = tmp_path / "ruleset.yaml"
    path.write_text(textwrap.dedent(body))
    return rules.load(path)


# --------------------------------------------------------------------------- #
# the loader
# --------------------------------------------------------------------------- #

def test_the_bundled_ruleset_loads(ruleset):
    assert ruleset.name == "cloud-support"
    assert ruleset.signals


def test_patterns_are_compiled_at_load_time(tmp_path):
    # A typo in a regex should fail on load, not on whichever ticket reaches it.
    with pytest.raises(rules.RulesetError, match="bad pattern"):
        ruleset_from(
            """
            signals:
              - {name: broken, weight: 1, pattern: '([unclosed'}
            """,
            tmp_path,
        )


def test_duplicate_signal_names_are_rejected(tmp_path):
    with pytest.raises(rules.RulesetError, match="duplicate"):
        ruleset_from(
            """
            signals:
              - {name: same, weight: 1, pattern: 'a'}
              - {name: same, weight: 2, pattern: 'b'}
            """,
            tmp_path,
        )


def test_damped_by_must_name_a_real_context_group(tmp_path):
    with pytest.raises(rules.RulesetError, match="not a context group"):
        ruleset_from(
            """
            signals:
              - {name: one, weight: -1, pattern: 'a', damped_by: nope}
            """,
            tmp_path,
        )


def test_a_context_group_cannot_name_an_unknown_signal(tmp_path):
    with pytest.raises(rules.RulesetError, match="unknown signals"):
        ruleset_from(
            """
            context_groups:
              billing: [ghost]
            signals:
              - {name: real, weight: 1, pattern: 'a'}
            """,
            tmp_path,
        )


def test_thresholds_must_be_listed_highest_first(tmp_path):
    with pytest.raises(rules.RulesetError, match="highest score first"):
        ruleset_from(
            """
            thresholds:
              - {label: low, min: 0}
              - {label: high, min: 6}
            signals: []
            """,
            tmp_path,
        )


def test_a_missing_ruleset_is_a_clear_error():
    with pytest.raises(rules.RulesetError, match="no such ruleset"):
        rules.load("/nonexistent/ruleset.yaml")


# --------------------------------------------------------------------------- #
# credit signals
# --------------------------------------------------------------------------- #

def test_missing_credits_ranks_as_a_quick_win():
    assert score.score_text("Promotional credits disappeared from my account").label == "quick-win"


def test_inflected_forms_match():
    # An earlier pattern ended in `disappear\b`, which cannot match the most
    # common phrasing of the exact thing it was written to catch.
    for phrasing in ("credits disappeared", "credits have vanished", "credits expired"):
        verdict = score.score_text(phrasing)
        assert verdict.named("credit-missing"), phrasing


def test_the_words_may_come_in_either_order():
    assert score.score_text("missing credits").named("credit-missing-reversed")


def test_credit_card_is_not_a_credits_question():
    # The single biggest false positive in the ruleset: a fraud dispute names the
    # card that was charged, which has nothing to do with promotional credit.
    verdict = score.score_text("Unauthorised charge on my credit card")
    assert verdict.named("credits") is None


def test_suspected_fraud_is_penalised():
    verdict = score.score_text("Unauthorised charge of $27.40 on credit card")
    assert verdict.named("suspected-fraud")
    assert verdict.score < 3


# --------------------------------------------------------------------------- #
# hard signals and damping
# --------------------------------------------------------------------------- #

def test_an_outage_is_a_time_sink():
    assert score.score_text("Production is down, urgent, complete outage").label == "time-sink"


def test_a_service_name_penalty_is_damped_inside_a_billing_ticket():
    # A customer disputing a bill names whichever line item surprised them. That
    # is not evidence the ticket is hard.
    billing = score.score_text("Refund request for Direct Connect port hours we never used")
    technical = score.score_text("Direct Connect route propagation broken")

    damped = billing.named("deep-network")
    raw = technical.named("deep-network")
    assert damped and raw
    assert damped.weight > raw.weight        # less negative
    assert "damped" in damped.note


def test_damping_can_be_switched_off():
    with_damping = score.score_text("Refund for Direct Connect port hours")
    without = score.score_text("Refund for Direct Connect port hours", damping=False)
    assert with_damping.score > without.score


def test_an_undamped_penalty_still_applies_without_billing_context():
    assert score.score_text("BGP session flapping on Direct Connect").label == "time-sink"


# --------------------------------------------------------------------------- #
# scoping
# --------------------------------------------------------------------------- #

def test_a_subject_scoped_signal_ignores_the_body():
    # `short-subject` measures the subject's length. Before signals had a scope it
    # ran against every field joined by newlines, where `^.{0,45}$` can never
    # match -- so it was dead code that looked like a rule which did not apply.
    ticket = {"subject": "Charge wrong", "correspondence": "x" * 400}
    assert score.score_case(ticket).named("short-subject")


def test_a_long_subject_is_penalised_however_short_the_body():
    ticket = {"subject": "y" * 200, "correspondence": "short"}
    assert score.score_case(ticket).named("long-subject")


def test_an_unscoped_signal_reads_the_body():
    ticket = {"subject": "Billing question", "correspondence": "my promo code will not redeem"}
    assert score.score_case(ticket).named("promo-code")


def test_body_can_be_supplied_separately():
    ticket = {"subject": "Billing question"}
    plain = score.score_case(ticket)
    with_body = score.score_case(ticket, body="My promotional credits are gone")
    assert with_body.score > plain.score


# --------------------------------------------------------------------------- #
# case-level adjustments
# --------------------------------------------------------------------------- #

def test_low_severity_adds_a_bonus():
    verdict = score.score_case({"subject": "How do I close my account", "severity": "low"})
    assert verdict.named("severity:low")


def test_only_one_severity_bonus_applies():
    verdict = score.score_case({"subject": "anything", "severity": "low"})
    assert len([h for h in verdict.hits if h.name.startswith("severity:")]) == 1


def test_a_named_account_is_penalised():
    plain = score.score_case({"subject": "Missing credits", "severity": "low"})
    vip = score.score_case({"subject": "Missing credits", "severity": "low", "is_vip": True})
    assert vip.score < plain.score


# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #

def test_rank_puts_the_cheapest_first():
    tickets = [
        {"id": "hard", "subject": "Production is down, complete outage", "severity": "critical"},
        {"id": "easy", "subject": "Promotional credits disappeared", "severity": "low"},
    ]
    assert [t["id"] for t, _ in score.rank(tickets)] == ["easy", "hard"]


def test_rank_is_stable_for_equal_scores():
    tickets = [{"id": "b", "subject": "same"}, {"id": "a", "subject": "same"}]
    assert [t["id"] for t, _ in score.rank(tickets)] == ["a", "b"]


def test_every_hit_is_explainable():
    verdict = score.score_case({"subject": "Promotional credits disappeared", "severity": "low"})
    assert verdict.score == sum(hit.weight for hit in verdict.hits)


# --------------------------------------------------------------------------- #
# age
# --------------------------------------------------------------------------- #

NOW = datetime(2026, 9, 9, tzinfo=timezone.utc)


def aged(ticket_id: str, days: int, subject: str = "Billing question") -> dict:
    created = NOW - timedelta(days=days)
    return {"id": ticket_id, "subject": subject, "created": created.isoformat()}


def test_age_is_measured_in_days():
    assert score.age_days(aged("x", 10), now=NOW) == pytest.approx(10.0)


def test_a_missing_or_unparseable_date_reads_as_new():
    # Age is a tiebreaker. One malformed field should not stop the queue ranking.
    assert score.age_days({"id": "x"}, now=NOW) == 0.0
    assert score.age_days({"id": "x", "created": "last tuesday"}, now=NOW) == 0.0


def test_a_future_date_does_not_earn_a_penalty():
    assert score.age_days({"created": "2027-01-01T00:00:00Z"}, now=NOW) == 0.0


def test_the_older_of_two_equal_tickets_comes_first():
    ranked = score.rank([aged("new", 1), aged("old", 3)], now=NOW)
    assert [t["id"] for t, _ in ranked] == ["old", "new"]


def test_age_can_overcome_a_small_effort_gap():
    fresh = {"id": "fresh", "subject": "Cannot redeem promo code",
             "created": NOW.isoformat()}
    stale = aged("stale", 21, subject="Need a tax receipt showing our VAT number")

    on_effort = score.rank([fresh, stale], aging=False, now=NOW)
    with_age = score.rank([fresh, stale], now=NOW)

    assert [t["id"] for t, _ in on_effort] == ["fresh", "stale"]
    assert [t["id"] for t, _ in with_age] == ["stale", "fresh"]


def test_age_cannot_overcome_a_large_one():
    # The whole reason the bonus is capped: waiting must not turn an outage into
    # the next thing to pick up.
    outage = aged("outage", 400, subject="Production is down, complete outage")
    outage["severity"] = "critical"
    easy = {"id": "easy", "subject": "Promotional credits disappeared",
            "severity": "low", "created": NOW.isoformat()}
    assert [t["id"] for t, _ in score.rank([outage, easy], now=NOW)] == ["easy", "outage"]


def test_the_bonus_is_capped(ruleset):
    forever = score.rank([aged("x", 10_000)], now=NOW)[0][1]
    assert forever.adjustments[0].weight == ruleset.aging.max_bonus


def test_age_stays_out_of_the_effort_score():
    # If the bonus landed in `score`, a ticket would become a "quick-win" by
    # sitting in the queue, and the eval's agreement number could be inflated by a
    # mechanism that says nothing about effort.
    old = aged("old", 90)
    with_age = score.rank([old], now=NOW)[0][1]
    without = score.rank([old], aging=False, now=NOW)[0][1]

    assert with_age.score == without.score
    assert with_age.label == without.label
    assert with_age.order_score > with_age.score
    assert without.adjustments == []


def test_half_configured_aging_is_rejected(tmp_path):
    # A ruleset that names days_per_point but no ceiling reads as enabled and does
    # nothing, which is how the dead structural signals hid for so long.
    with pytest.raises(rules.RulesetError, match="days_per_point and max_bonus"):
        ruleset_from(
            """
            aging: {days_per_point: 7}
            signals: []
            """,
            tmp_path,
        )


def test_aging_is_off_for_a_ruleset_that_does_not_mention_it(tmp_path):
    quiet = ruleset_from("signals: []\n", tmp_path)
    assert not quiet.aging.enabled
    assert quiet.aging.bonus(500) == 0


# --------------------------------------------------------------------------- #
# exclusions
# --------------------------------------------------------------------------- #

def test_a_refund_request_is_excluded(ruleset):
    case = {"id": "r", "subject": "Refund request for instance left running by accident"}
    assert score.excluded_by(case, ruleset).name == "refund-request"


def test_a_billing_adjustment_is_excluded(ruleset):
    case = {"id": "a", "subject": "Please make a one-time adjustment to my invoice"}
    assert score.excluded_by(case, ruleset).name == "billing-adjustment"


def test_the_exclusion_reads_metadata_as_well_as_text(ruleset):
    case = {"id": "a", "subject": "Charge looks wrong", "item": "courtesy-adjustment"}
    assert score.excluded_by(case, ruleset).name == "billing-adjustment"


def test_reimbursement_is_not_a_refund_request(ruleset):
    # An invoice download question, filtered out for a word the customer used
    # about their own expense claim. The pattern lost `reimburse\w*` over this.
    case = {"id": "i", "subject": "Where do I download my invoice",
            "correspondence": "Our finance team needs it for reimbursement."}
    assert score.excluded_by(case, ruleset) is None


def test_an_ordinary_credits_ticket_survives(ruleset):
    case = {"id": "c", "subject": "Promotional credits disappeared from my account"}
    assert score.excluded_by(case, ruleset) is None


def test_split_keeps_the_reason_with_the_dropped_ticket(ruleset):
    cases = [
        {"id": "keep", "subject": "Promotional credits disappeared"},
        {"id": "drop", "subject": "I want a refund"},
    ]
    keep, dropped = score.split_excluded(cases, ruleset)
    assert [c["id"] for c in keep] == ["keep"]
    assert [(c["id"], rule.name) for c, rule in dropped] == [("drop", "refund-request")]
    assert dropped[0][1].reason


def test_ranking_does_not_filter(ruleset):
    # Filtering is a preference about what to look at; scoring is the part with a
    # measurable claim. Keeping them apart is what lets the eval measure the
    # scorer on tickets the operator never sees.
    cases = [{"id": "drop", "subject": "I want a refund"}]
    assert len(score.rank(cases, ruleset=ruleset)) == 1


def test_duplicate_exclusion_names_are_rejected(tmp_path):
    with pytest.raises(rules.RulesetError, match="duplicate exclusion"):
        ruleset_from(
            """
            exclusions:
              - {name: same, pattern: 'a'}
              - {name: same, pattern: 'b'}
            signals: []
            """,
            tmp_path,
        )


def test_an_exclusion_pattern_is_compiled_at_load_time(tmp_path):
    with pytest.raises(rules.RulesetError, match="bad pattern"):
        ruleset_from(
            """
            exclusions:
              - {name: broken, pattern: '([unclosed'}
            signals: []
            """,
            tmp_path,
        )
