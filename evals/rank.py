"""Score the queue ranker against labelled tickets and print the numbers.

    uv run python -m evals.rank
    uv run python -m evals.rank --signals
    uv run python -m evals.rank --ablate
    uv run python -m evals.rank --json results.json

No model is involved, so unlike `evals/run.py` this suite is deterministic, free,
and finishes instantly. Running it twice gives the same answer; if a number moved,
something in the ruleset moved.

WHAT IS BEING MEASURED, AND WHAT IS NOT

The ground truth is `minutes` in `evals/queue.yaml`, which is a hand-set prior
rather than observed handling time. That bounds what can honestly be claimed:

  - Claimable: the ruleset ranks these tickets in roughly the order I would, a
    change moved that agreement up or down, and mechanism X contributes Y to it.
  - Not claimable: any statement about accuracy on a real queue.

So this file reports rank agreement and ablation deltas, and deliberately prints
no accuracy percentage. Swap `minutes` for real handling times and the same
harness becomes a real measurement -- that is the intended upgrade, and no code
here has to change for it.

THE DIAGNOSTIC THAT MATTERS MOST

`--signals` is the one to run after editing a weight. For each signal it prints
the mean label of the tickets it fires on against the mean of those it does not.
A positive-weighted signal whose tickets take longer than average has its sign
wrong, and no amount of tuning the magnitude will fix that. It is the cheapest
check available on a hand-set number.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from queue_triage import rules, score

QUEUE = Path(__file__).parent / "queue.yaml"

# The default queue: untouched, English, real. Matches the CLI default so the
# eval measures the ranking people actually see.
EVAL_QUERY = 'queue=unassigned & language=en & -subject:"test case"'

# minutes -> bucket. Boundaries are where the handling changes in kind, not round
# numbers: a ticket you answer from a template, one that needs a lookup, one that
# needs the customer's environment, one that needs a call.
BUCKETS = (
    ("quick", 30),
    ("easy", 60),
    ("medium", 150),
    ("hard", 10 ** 9),
)

# Which predicted label we would like each bucket to get. Used for the confusion
# table only -- the ranking metrics do not depend on this mapping.
IDEAL_LABEL = {
    "quick": "quick-win",
    "easy": "likely-easy",
    "medium": "unclear",
    "hard": "time-sink",
}
LABEL_ORDER = ("quick-win", "likely-easy", "unclear", "time-sink")


def bucket_of(minutes: float) -> str:
    for name, upper in BUCKETS:
        if minutes <= upper:
            return name
    return BUCKETS[-1][0]


# --------------------------------------------------------------------------- #
# statistics, stdlib only
# --------------------------------------------------------------------------- #

def _ranks(values: list[float]) -> list[float]:
    """Average ranks, so ties do not distort the correlation."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            result[order[k]] = shared
        i = j + 1
    return result


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation. Pearson applied to average ranks, which handles ties."""
    if len(xs) < 2:
        return 0.0
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #

@dataclass
class Row:
    ticket: dict
    verdict: object

    @property
    def id(self) -> str:
        return str(self.ticket.get("id"))

    @property
    def minutes(self) -> float:
        return float(self.ticket.get("minutes", 0))

    @property
    def bucket(self) -> str:
        return bucket_of(self.minutes)

    @property
    def fired(self) -> set[str]:
        return {h.name for h in self.verdict.hits}


def load_tickets(path: Path = QUEUE) -> list[dict]:
    from queue_triage.backends import FileBackend
    from queue_triage.query import parse

    backend = FileBackend(path)
    return backend.search(parse(EVAL_QUERY), limit=10 ** 6)


def run(tickets: list[dict], ruleset: rules.Ruleset, damping: bool = True) -> list[Row]:
    """Rank the labelled set, with the two policy layers switched off.

    `aging=False`: the age bonus is not a claim about effort, and on this set it
    would flatter the numbers for the wrong reason -- the tickets were written
    easiest-first, so creation date correlates with the labels by accident. A
    mechanism that reorders the queue for fairness should not be able to move an
    accuracy-shaped number at all.

    Exclusions are likewise not applied: `score.split_excluded` is never called
    here. What the operator has chosen to hide is a preference, and the scorer
    should still be measured on those tickets -- two of the adversarial cases are
    refunds, and they are the sharpest tests of the damping mechanism in the set.
    """
    ranked = score.rank(tickets, ruleset=ruleset, damping=damping, aging=False)
    return [Row(ticket, verdict) for ticket, verdict in ranked]


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def metrics(rows: list[Row]) -> dict:
    scores = [float(r.verdict.score) for r in rows]
    minutes = [r.minutes for r in rows]

    # Negated because a high score should mean few minutes; this way +1 is
    # perfect agreement and the sign is not a thing to remember.
    rho = -spearman(scores, minutes)

    def precision_at(k: int, of: set[str]) -> float:
        top = rows[:k]
        return mean(1.0 if r.bucket in of else 0.0 for r in top)

    inversions = wrong_pairs(rows)
    pairs = len(rows) * (len(rows) - 1) // 2

    return {
        "tickets": len(rows),
        "spearman": round(rho, 3),
        "precision_at_5_quick": round(precision_at(5, {"quick"}), 3),
        "precision_at_10_quick_or_easy": round(precision_at(10, {"quick", "easy"}), 3),
        "hard_in_top_10": sum(1 for r in rows[:10] if r.bucket == "hard"),
        "quick_in_bottom_10": sum(1 for r in rows[-10:] if r.bucket == "quick"),
        "inverted_pairs": len(inversions),
        "inverted_pair_rate": round(len(inversions) / pairs, 3) if pairs else 0.0,
    }


def wrong_pairs(rows: list[Row]) -> list[tuple[str, str]]:
    """Pairs ranked in the opposite order to their labels, two buckets apart.

    Adjacent-bucket disagreement is noise given hand-set labels. A `quick` ticket
    ranked below a `hard` one is not.
    """
    index = {name: i for i, (name, _) in enumerate(BUCKETS)}
    out = []
    for i, above in enumerate(rows):
        for below in rows[i + 1 :]:
            gap = index[below.bucket] - index[above.bucket]
            if gap <= -2:
                out.append((above.id, below.id))
    return out


def confusion(rows: list[Row]) -> dict:
    table: dict[str, dict[str, int]] = {}
    for row in rows:
        table.setdefault(row.bucket, {}).setdefault(row.verdict.label, 0)
        table[row.bucket][row.verdict.label] += 1
    return table


# Below this many firings, the sign check is not a check. One ticket either
# happens to sit above the mean or happens to sit below it, and calling that
# "the weight points the wrong way" is reading a coin flip as evidence.
MIN_N_FOR_SIGN = 3


def signal_report(rows: list[Row], ruleset: rules.Ruleset) -> list[dict]:
    """Per-signal sanity check: do the tickets it fires on take longer or less?"""
    overall = mean(r.minutes for r in rows)
    report = []
    for signal in ruleset.signals:
        on = [r for r in rows if signal.name in r.fired]
        if not on:
            report.append({
                "signal": signal.name, "weight": signal.weight, "fired_on": 0,
                "mean_minutes": None, "delta_vs_overall": None, "sign_ok": None,
                "verdict": "never fired",
            })
            continue
        on_mean = mean(r.minutes for r in on)
        delta = on_mean - overall
        # A positive weight claims "faster than average", i.e. a negative delta.
        sign_ok = (signal.weight > 0) == (delta < 0)
        if len(on) < MIN_N_FOR_SIGN:
            verdict = "ok" if sign_ok else f"n<{MIN_N_FOR_SIGN}, no call"
        else:
            verdict = "ok" if sign_ok else "WRONG WAY"
        report.append({
            "signal": signal.name,
            "weight": signal.weight,
            "fired_on": len(on),
            "mean_minutes": round(on_mean, 1),
            "delta_vs_overall": round(delta, 1),
            "sign_ok": sign_ok,
            "verdict": verdict,
        })
    return report


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #

def print_metrics(name: str, m: dict) -> None:
    print(f"\n## {name}")
    print(f"  tickets ranked                {m['tickets']}")
    print(f"  rank agreement (Spearman)     {m['spearman']:+.3f}   (+1 = my order exactly)")
    print(f"  top 5 that are quick          {m['precision_at_5_quick']:.0%}")
    print(f"  top 10 that are quick or easy {m['precision_at_10_quick_or_easy']:.0%}")
    print(f"  hard tickets in the top 10    {m['hard_in_top_10']}")
    print(f"  quick tickets in the bottom 10 {m['quick_in_bottom_10']}")
    print(f"  badly inverted pairs          {m['inverted_pairs']} ({m['inverted_pair_rate']:.1%} of all pairs)")


def print_confusion(rows: list[Row]) -> None:
    table = confusion(rows)
    width = max(len(l) for l in LABEL_ORDER) + 2
    print("\n## predicted label by actual bucket")
    print("  " + "actual".ljust(8) + "".join(l.ljust(width) for l in LABEL_ORDER))
    for bucket, _ in BUCKETS:
        row = table.get(bucket, {})
        cells = "".join(str(row.get(l, 0)).ljust(width) for l in LABEL_ORDER)
        want = IDEAL_LABEL[bucket]
        print(f"  {bucket.ljust(8)}{cells}  <- would like {want}")


def print_signals(report: list[dict]) -> None:
    print("\n## per-signal check (does the label agree with the sign of the weight?)")
    print(f"  {'signal':<26}{'wt':>4}{'n':>4}{'mean min':>10}{'vs all':>9}  sign")
    for entry in sorted(report, key=lambda e: (e["fired_on"] == 0, -(e["delta_vs_overall"] or 0))):
        if not entry["fired_on"]:
            print(f"  {entry['signal']:<26}{entry['weight']:>4}{0:>4}{'-':>10}{'-':>9}  never fired")
            continue
        print(f"  {entry['signal']:<26}{entry['weight']:>+4}{entry['fired_on']:>4}"
              f"{entry['mean_minutes']:>10.1f}{entry['delta_vs_overall']:>+9.1f}  {entry['verdict']}")

    wrong = [e for e in report if e["verdict"] == "WRONG WAY"]
    if wrong:
        print(f"\n  {len(wrong)} signal(s) point the wrong way on this set: "
              f"{', '.join(e['signal'] for e in wrong)}")
        print("  Either the weight's sign is wrong, or the pattern is matching")
        print("  tickets it was not written for. Both are worth looking at.")

    thin = [e for e in report if e["verdict"].startswith("n<")]
    if thin:
        print(f"\n  {len(thin)} signal(s) disagree but fired on fewer than {MIN_N_FOR_SIGN} "
              f"tickets: {', '.join(e['signal'] for e in thin)}")
        print("  Not reported as failures. The fix is more labelled tickets, not a")
        print("  different weight -- a single ticket cannot tell you a sign is wrong.")

    never = [e for e in report if not e["fired_on"]]
    if never:
        print(f"\n  {len(never)} signal(s) never fired: {', '.join(e['signal'] for e in never)}")
        print("  Either the set has no example, or the pattern cannot match. Check which:")
        print("  a pattern that is structurally incapable of matching looks identical here.")


def print_adversarial(rows: list[Row]) -> None:
    flagged = [(i, r) for i, r in enumerate(rows, 1) if r.ticket.get("adversarial")]
    if not flagged:
        return
    print("\n## known-hard examples, and where they landed")
    for position, row in flagged:
        print(f"  #{position:<3} [{row.verdict.score:+3d}] {row.verdict.label:<12} "
              f"{row.id}  {str(row.ticket.get('subject'))[:52]}")
        print(f"        actually {row.minutes:.0f} min ({row.bucket}); "
              f"{' '.join(str(row.ticket.get('adversarial_note') or '').split())[:150]}")


def print_ablation(tickets: list[dict], ruleset: rules.Ruleset) -> dict:
    print("\n## ablation: context damping")
    print("  Halves service-name penalties when a billing signal has already fired.")
    print("  Both arms use the same labels, so the delta is meaningful even though")
    print("  the labels are hand-set.\n")

    arms = {}
    for label, damping in (("damping on", True), ("damping off", False)):
        m = metrics(run(tickets, ruleset, damping=damping))
        arms[label] = m
        print(f"  {label:<14} spearman {m['spearman']:+.3f}   "
              f"inverted pairs {m['inverted_pairs']:>2}   "
              f"top-5 quick {m['precision_at_5_quick']:.0%}")

    delta = arms["damping on"]["spearman"] - arms["damping off"]["spearman"]
    verdict = "helps" if delta > 0.001 else "hurts" if delta < -0.001 else "makes no difference"
    print(f"\n  -> damping {verdict} here (spearman {delta:+.3f})")
    if abs(delta) < 0.001:
        print("     On 30 tickets that is entirely possible; it means the mechanism is")
        print("     not earning its complexity on this set, not that it is wrong.")
    return arms


# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the queue ranker against labelled tickets.")
    parser.add_argument("--tickets", default=QUEUE, type=Path)
    parser.add_argument("--ruleset", default=None, help="ruleset name or path")
    parser.add_argument("--signals", action="store_true", help="per-signal sanity check")
    parser.add_argument("--ablate", action="store_true", help="measure the damping mechanism")
    parser.add_argument("--json", default=None, help="also write results to this path")
    args = parser.parse_args(argv)

    ruleset = rules.load(args.ruleset) if args.ruleset else rules.load()
    tickets = load_tickets(args.tickets)
    if not tickets:
        print(f"no tickets matched {EVAL_QUERY!r} in {args.tickets}", file=sys.stderr)
        return 1

    rows = run(tickets, ruleset)

    print(f"# queue_triage ranking eval -- ruleset {ruleset.name!r}, {len(rows)} tickets")
    print(f"# ground truth is a hand-set prior, not observed handling time.")
    m = metrics(rows)
    print_metrics("ranking", m)
    print_confusion(rows)
    print_adversarial(rows)

    results = {"ruleset": ruleset.name, "metrics": m, "confusion": confusion(rows)}

    if args.signals:
        report = signal_report(rows, ruleset)
        print_signals(report)
        results["signals"] = report

    if args.ablate:
        results["ablation"] = print_ablation(tickets, ruleset)

    bad = wrong_pairs(rows)
    if bad:
        print(f"\n## {len(bad)} badly inverted pair(s) -- two buckets apart, ranked backwards")
        for above, below in bad[:10]:
            print(f"  {above} ranked above {below}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
