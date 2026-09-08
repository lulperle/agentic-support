"""Tests for the intake gate and the output-language decision.

Both are code paths that exist because the model was unreliable at the same job:
triage guessed a resource it had been told not to guess, and drifted out of the
ticket's language two runs in five. So both are now decided here, deterministically,
and the tests are about the decision -- no model involved.
"""

from __future__ import annotations

import json

import pytest

from support_agent.pipeline import decide, reply_language, triage_prompt


def _intake(**overrides) -> str:
    base = {
        "category": "performance_degradation",
        "function_name": "checkout-api",
        "needs_telemetry": True,
        "urgency": "high",
        "rationale": "the ticket says responses are slow",
    }
    return json.dumps({**base, **overrides})


class TestGate:
    def test_named_resource_proceeds(self):
        assert decide("checkout-api is slow", _intake()).proceed

    def test_no_resource_named_does_not_reach_triage(self):
        # The defect this gate exists for: triage guessed `notify-fanout` from
        # "notifications aren't going out" and pulled its metrics.
        decision = decide(
            "notifications aren't going out",
            _intake(
                category="insufficient_information",
                function_name=None,
                needs_telemetry=False,
            ),
        )
        assert not decision.proceed
        assert decision.reply

    def test_telemetry_claimed_but_nothing_to_query_still_stops(self):
        # needs_telemetry true with no resource is incoherent; the gate must not
        # take the model's word for it and send an empty lookup downstream.
        decision = decide("everything is broken", _intake(function_name=None))
        assert not decision.proceed

    def test_how_to_gets_guidance_not_a_question(self):
        decision = decide(
            "how do I rotate a key?",
            _intake(category="how_to_or_guidance", function_name=None,
                    needs_telemetry=False),
        )
        assert not decision.proceed
        assert "user guide" in decision.reply

    def test_reply_matches_the_ticket_language(self):
        decision = decide(
            "通知が届きません",
            _intake(category="insufficient_information", function_name=None,
                    needs_telemetry=False),
        )
        assert any("぀" <= ch <= "ヿ" for ch in decision.reply)


class TestReplyLanguage:
    @pytest.mark.parametrize(
        "ticket,expected",
        [
            ("checkout-api is slow this morning", "English"),
            ("今朝からcheckout-apiの応答が異常に遅いです", "Japanese"),
            ("invoice-batch 504", "English"),
        ],
    )
    def test_language_is_decided_from_the_script(self, ticket, expected):
        assert reply_language(ticket) == expected

    def test_kanji_only_text_is_read_as_english_and_that_is_a_known_limit(self):
        # Pinning the limitation rather than pretending it isn't there. The
        # detector requires kana, because kanji alone does not distinguish
        # Japanese from Chinese. A kanji-only Japanese ticket -- rare in a real
        # helpdesk, since particles are kana -- would therefore be answered in
        # English. Fixing it properly means a real language identifier, not a
        # wider character range, and this test is where that change would land.
        assert reply_language("請求書処理") == "English"

    def test_prompt_states_the_language_and_keeps_the_ticket(self):
        prompt = triage_prompt("checkout-api is slow")
        assert "checkout-api is slow" in prompt
        assert "English" in prompt

    def test_japanese_prompt_asks_for_japanese(self):
        assert "Japanese" in triage_prompt("応答が遅いです")
