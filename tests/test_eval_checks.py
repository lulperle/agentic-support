"""Tests for the harness's own scoring logic.

The suite's checks are code, and code that scores other code has to be right or
it sends you off to fix things that work. The abstention check shipped broken
once -- it looked for "?" and so failed every correct Japanese clarifying reply,
which end in "。" -- so it gets tests.
"""

from __future__ import annotations

import pytest

from evals.run import (
    _asks_for_information,
    _asserts_as_cause,
    _classify_upstream,
    _looks_english,
    report_stability,
)


class TestAsksForInformation:
    @pytest.mark.parametrize(
        "text",
        [
            "Which function or service is affected?",
            "Could you confirm the resource name",
            "Please let us know which system",
            # The case that was scored wrong: a question ending in "。".
            "どの関数・サービスで発生していますか。名前が分かれば確認できます。",
            "対象のリソース名を教えてください。",
            # Verbatim from a run. A request stated rather than asked -- no
            # interrogative anywhere, and the first marker list missed it.
            "Before I can triage this, I need you to confirm the exact function "
            "name (it may be spelled differently).",
        ],
    )
    def test_recognises_a_request_for_information(self, text):
        assert _asks_for_information(text)

    @pytest.mark.parametrize(
        "text",
        [
            "The root cause is a DynamoDB connect timeout.",
            "notify-fanout lacks sns:Publish permission.",
            "根本原因はDynamoDBへの接続タイムアウトです。",
        ],
    )
    def test_does_not_mistake_a_diagnosis_for_a_question(self, text):
        assert not _asks_for_information(text)


class TestUpstreamClassification:
    def test_daily_quota_is_not_retried(self):
        exc = RuntimeError(
            "429 RESOURCE_EXHAUSTED quotaId: "
            "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
        )
        assert _classify_upstream(exc) == "quota"

    def test_per_minute_limit_is_retried(self):
        assert _classify_upstream(RuntimeError("429 RESOURCE_EXHAUSTED")) == "retry"

    def test_capacity_error_is_retried(self):
        assert _classify_upstream(RuntimeError("503 UNAVAILABLE")) == "retry"

    def test_ordinary_bug_is_not_swallowed(self):
        # The retry wrapper catches broadly, so this is what stops a genuine
        # TypeError from being reported as a flaky upstream.
        assert _classify_upstream(TypeError("bad argument")) is None


class TestInventedCause:
    """All strings here are verbatim from real runs, not invented for the test."""

    def test_asserted_as_a_probable_cause_is_invention(self):
        assert _asserts_as_cause(
            "**Probable root cause:** Connectivity or throttling issues between "
            "`checkout-api` and the DynamoDB endpoint in `ap-northeast-1`, leading "
            "to connection timeouts and task timeouts.",
            "throttl",
        )

    def test_naming_what_was_ruled_out_is_not_invention(self):
        assert not _asserts_as_cause(
            "**No evidence of** throttling, cold starts, or deployment changes -- "
            "none of those appeared in the metrics or logs, so I'm not listing "
            "them as contributing factors.",
            "throttl",
        )
        assert not _asserts_as_cause(
            "**No evidence of** throttling, cold starts, or deployment changes.",
            "cold start",
        )

    def test_suggesting_a_next_step_is_not_invention(self):
        assert not _asserts_as_cause(
            "追加で確認したい点があれば、DynamoDB側のテーブルメトリクス"
            "（スロットリングやキャパシティ状況など）も合わせて調査することを"
            "お勧めします。",
            "スロットリング",
        )

    def test_deferring_to_a_system_the_tools_cannot_see_is_not_invention(self):
        assert not _asserts_as_cause(
            "DynamoDB側の状態(スロットリング設定・リージョン間ネットワーク・VPC設定など)"
            "を確認してください。ツール側からはDynamoDBそのものの状態は取得できません。",
            "スロットリング",
        )

    def test_japanese_clause_final_negation_is_not_an_assertion(self):
        # Verbatim. The negation ("ログはないため") attaches to a predicate well
        # after the mention, and 根本原因 appears in the same clause, so an
        # earlier version of the check read this as a causal claim. It is the
        # opposite: the agent is saying it cannot see DynamoDB's side and is
        # therefore capping its confidence.
        assert not _asserts_as_cause(
            "DynamoDBへの接続タイムアウトが記録されている点は明確な証拠ですが、"
            "DynamoDB側で何が起きているか(スロットリング、ネットワーク側の問題、"
            "DynamoDB自体の障害など)を直接示すログはないため、根本原因の特定は"
            "ここまでとし、確信度は中程度としています。",
            "スロットリング",
        )

    def test_japanese_causal_assertion_is_still_caught(self):
        # The rule-out and next-step exemptions must not become a loophole: the
        # same word inside an actual causal claim still fails.
        assert _asserts_as_cause(
            "原因はDynamoDBのスロットリングです。", "スロットリング"
        )

    def test_a_mention_with_no_causal_claim_at_all_passes(self):
        assert not _asserts_as_cause(
            "The logs show task timeouts. Throttling is a separate subject.",
            "throttl",
        )


class TestLanguageCheck:
    def test_accepts_english(self):
        assert _looks_english(
            "The p99 is 9,850 ms and the error rate is 8.1% for checkout-api."
        )

    def test_rejects_spanish(self):
        # The answer that exposed the gap: an English ticket answered in Spanish
        # scored green, because the check only asked whether the text was
        # Japanese. "Not Japanese" is not "English".
        assert not _looks_english(
            "Gracias por el reporte, pero no pude encontrar una función llamada "
            "payment-api en el sistema de telemetría."
        )

    def test_rejects_japanese(self):
        assert not _looks_english("DynamoDBへの接続がタイムアウトしています。")

    def test_the_shipped_gate_replies_pass_their_own_checks(self):
        # These are constant strings, so a suite that scores them as the wrong
        # language is measuring itself. The first version of the English check
        # did exactly that, and the failure looked like an agent defect.
        from support_agent.pipeline import (
            _ASK_FOR_RESOURCE,
            _ASK_FOR_RESOURCE_JA,
            _GUIDANCE,
        )

        assert _looks_english(_ASK_FOR_RESOURCE)
        assert _asks_for_information(_ASK_FOR_RESOURCE)
        assert _asks_for_information(_ASK_FOR_RESOURCE_JA)
        assert not _looks_english(_ASK_FOR_RESOURCE_JA)
        assert _looks_english(_GUIDANCE)


def _run(*outcomes: bool) -> list[dict]:
    """One run's report for a single ticket with the given check outcomes."""
    return [
        {
            "id": "t-01",
            "checks": [
                {"check": f"c{i}", "passed": p} for i, p in enumerate(outcomes)
            ],
        }
    ]


class TestStabilityReport:
    def test_consistent_passes_are_not_reported(self):
        assert report_stability([_run(True, True), _run(True, True)]) == []

    def test_intermittent_check_is_named_with_its_rate(self):
        unstable = report_stability([_run(True, True), _run(True, False)])
        assert unstable == [
            {"ticket": "t-01", "check": "c1", "passes": 1, "runs": 2}
        ]

    def test_consistent_failure_is_still_reported(self):
        unstable = report_stability([_run(False), _run(False)])
        assert unstable[0]["passes"] == 0

    def test_missing_observation_is_not_counted_as_a_pass(self):
        # An errored ticket contributes no checks. Without the len() guard a
        # check seen once and passed would look like 1/1 and be called stable,
        # hiding that four of five runs never got far enough to score it.
        runs = [_run(True), _run(True), [{"id": "t-01", "checks": []}]]
        assert report_stability(runs) == [
            {"ticket": "t-01", "check": "c0", "passes": 2, "runs": 3}
        ]
