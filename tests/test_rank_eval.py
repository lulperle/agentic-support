"""Tests for the ranking eval harness.

Two kinds of test here. The statistics are checked against cases with a known
answer, because a wrong correlation is not visible in the output -- it just prints
a number that looks like a result. The rest are guards on the dataset and on the
claims the harness makes, so that "the eval passes" keeps meaning something.
"""

import pytest

from evals import rank
from queue_triage import rules, score


@pytest.fixture(scope="module")
def tickets():
    return rank.load_tickets()


@pytest.fixture(scope="module")
def rows(tickets):
    return rank.run(tickets, rules.load())


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

def test_spearman_is_one_for_identical_order():
    assert rank.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_spearman_is_minus_one_for_reversed_order():
    assert rank.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_uses_ranks_not_values():
    # Monotone but wildly non-linear: a rank correlation should not care.
    assert rank.spearman([1, 2, 3, 4], [1, 2, 3, 4000]) == pytest.approx(1.0)


def test_spearman_handles_ties_without_dividing_by_zero():
    assert rank.spearman([1, 1, 1], [1, 2, 3]) == 0.0


def test_spearman_of_a_single_point_is_zero_not_an_error():
    assert rank.spearman([1], [1]) == 0.0


def test_average_ranks_are_shared_between_tied_values():
    assert rank._ranks([5, 5, 9]) == [1.5, 1.5, 3.0]


def test_buckets_split_on_the_documented_boundaries():
    assert rank.bucket_of(30) == "quick"
    assert rank.bucket_of(31) == "easy"
    assert rank.bucket_of(60) == "easy"
    assert rank.bucket_of(151) == "hard"


# --------------------------------------------------------------------------- #
# the dataset
# --------------------------------------------------------------------------- #

def test_the_default_query_excludes_what_it_should(tickets):
    ids = {t["id"] for t in tickets}
    assert "T-1031" not in ids, "a ticket with 'test case' in the subject"
    assert "T-1032" not in ids, "a non-English ticket"
    assert "T-1033" not in ids, "a ticket that is already assigned"


def test_every_ticket_carries_a_label(tickets):
    missing = [t["id"] for t in tickets if not t.get("minutes")]
    assert not missing, f"unlabelled: {missing}"


def test_every_ticket_has_a_unique_id(tickets):
    ids = [t["id"] for t in tickets]
    assert len(ids) == len(set(ids))


def test_the_set_covers_every_bucket(rows):
    covered = {row.bucket for row in rows}
    assert covered == {"quick", "easy", "medium", "hard"}


def test_adversarial_tickets_explain_themselves(tickets):
    for ticket in tickets:
        if ticket.get("adversarial"):
            assert ticket.get("adversarial_note"), ticket["id"]


# --------------------------------------------------------------------------- #
# the claims the harness makes
# --------------------------------------------------------------------------- #

def test_every_signal_fires_at_least_once(rows):
    """Coverage guard.

    A signal that never fires is either untested or structurally incapable of
    matching, and the two look identical in the report. Adding a rule means adding
    a ticket that exercises it.
    """
    report = rank.signal_report(rows, rules.load())
    never = [e["signal"] for e in report if not e["fired_on"]]
    assert not never, f"no ticket exercises: {never}"


def test_every_exclusion_has_an_example(tickets):
    """The same coverage guard, for the filtering layer.

    Checked against all of a ticket's matches rather than its first, or an
    exclusion shadowed by one listed above it would look exactly like one no
    ticket exercises.
    """
    ruleset = rules.load()
    matched = {
        rule.name
        for ticket in tickets
        for rule in score.exclusions_for(ticket, ruleset)
    }
    missing = [r.name for r in ruleset.exclusions if r.name not in matched]
    assert not missing, f"no ticket exercises: {missing}"


def test_the_eval_ranks_tickets_the_queue_hides(tickets):
    """Exclusions are a preference, so they must not shrink the measured set.

    Two of the adversarial cases are refund requests, and they are the sharpest
    tests of the damping mechanism in the set. If filtering leaked into the eval,
    the ablation would go quiet and look like the mechanism had stopped mattering.
    """
    kept, dropped = score.split_excluded(tickets, rules.load())
    assert dropped, "the fixture no longer exercises exclusion at all"
    assert len(rank.run(tickets, rules.load())) == len(tickets) > len(kept)


def test_the_eval_keeps_age_out_of_the_numbers(tickets):
    """Age must not be able to move an agreement number.

    The fixture was written easiest-first, so creation date correlates with the
    labels by accident. Aging is a fairness mechanism; letting it touch `score`
    would flatter the metrics for a reason that has nothing to do with the ruleset.
    """
    for row in rank.run(tickets, rules.load()):
        assert row.verdict.adjustments == []
        assert row.verdict.order_score == row.verdict.score


def test_no_signal_points_the_wrong_way(rows):
    report = rank.signal_report(rows, rules.load())
    wrong = [e["signal"] for e in report if e["verdict"] == "WRONG WAY"]
    assert not wrong, f"weight sign disagrees with the labels: {wrong}"


def test_a_thin_disagreement_is_not_reported_as_a_failure():
    """One firing cannot tell you a sign is wrong, and must not claim to."""
    assert rank.MIN_N_FOR_SIGN >= 3


def test_no_quick_ticket_is_ranked_below_a_hard_one(rows):
    assert rank.wrong_pairs(rows) == []


def test_ranking_beats_the_queue_order(rows, tickets):
    """The ranker has to be better than opening tickets oldest-first.

    Without this the metrics have no baseline, and a Spearman of +0.8 could just be
    the dataset happening to be ordered that way.
    """
    ranked = rank.metrics(rows)["spearman"]

    as_filed = [rank.Row(t, score.score_case(t)) for t in tickets]
    baseline = -rank.spearman(
        [float(i) for i in range(len(as_filed))],
        [r.minutes for r in as_filed],
    )
    assert ranked > baseline + 0.2


def test_damping_does_not_make_the_ranking_worse(tickets):
    ruleset = rules.load()
    on = rank.metrics(rank.run(tickets, ruleset, damping=True))["spearman"]
    off = rank.metrics(rank.run(tickets, ruleset, damping=False))["spearman"]
    assert on >= off


def test_the_harness_runs_end_to_end(capsys):
    assert rank.main(["--signals", "--ablate"]) == 0
    out = capsys.readouterr().out
    assert "rank agreement" in out
    # The labels are invented, so an accuracy claim would be a lie. Guard against
    # one creeping into the output.
    assert "accuracy" not in out.lower()
