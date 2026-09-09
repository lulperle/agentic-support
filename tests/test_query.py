"""Tests for the query expression compiler.

These pin the behaviours that fail quietly rather than loudly. A parser bug in a
search filter does not raise -- it returns a plausible number of wrong rows -- so
the cases worth writing down are the ones where a reasonable implementation would
silently disagree.
"""

import pytest

from queue_triage.emit import (
    push_not_down,
    terms_of,
    to_json,
    to_predicate,
    to_text,
)
from queue_triage.query import And, Not, Or, QueryError, Term, parse, tokenize


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #

def test_a_term_may_contain_spaces():
    # `subject:test case` is one criterion, not a term followed by a stray word.
    assert tokenize("subject:test case") == [("TERM", "subject:test case")]


def test_hyphen_inside_a_word_is_not_a_negation():
    # Otherwise every region and product name parses as NOT.
    node = parse("queue=free-tier")
    assert node == Term(field="queue", op="=", value="free-tier")


def test_hyphen_starting_a_term_is_a_negation():
    node = parse("-queue=free-tier")
    assert node == Not(Term(field="queue", op="=", value="free-tier"))


def test_quotes_survive_tokenising():
    node = parse('subject:"test case"')
    assert node.quoted is True
    assert node.value == "test case"


def test_unterminated_quote_is_an_error():
    with pytest.raises(QueryError):
        parse('subject:"test case')


def test_escaped_character_is_taken_literally():
    assert parse(r"queue=free\-tier").value == "free-tier"


# --------------------------------------------------------------------------- #
# precedence and grouping
# --------------------------------------------------------------------------- #

def test_and_binds_tighter_than_or():
    node = parse("a & b | c")
    assert isinstance(node, Or)
    assert isinstance(node.children[0], And)


def test_parentheses_override_precedence():
    node = parse("a & (b | c)")
    assert isinstance(node, And)
    assert isinstance(node.children[1], Or)


def test_same_operator_flattens():
    # Three ANDs should be one node with three children, not a nested chain.
    node = parse("a & b & c")
    assert isinstance(node, And)
    assert len(node.children) == 3


def test_negation_can_apply_to_a_group():
    node = parse("-(a & b)")
    assert isinstance(node, Not)
    assert isinstance(node.child, And)


def test_unbalanced_parentheses_are_errors():
    with pytest.raises(QueryError):
        parse("(a & b")
    with pytest.raises(QueryError):
        parse("a & b)")


def test_empty_query_is_an_error():
    with pytest.raises(QueryError):
        parse("   ")


# --------------------------------------------------------------------------- #
# leaves
# --------------------------------------------------------------------------- #

def test_field_alias_is_applied():
    assert parse("sev=low").field == "severity"


def test_a_value_with_no_field_is_free_text():
    assert parse("credits").is_free_text is True


def test_a_colon_inside_a_quoted_value_does_not_split_a_field():
    node = parse('"note: see attachment"')
    assert node.is_free_text is True
    assert node.value == "note: see attachment"


def test_equals_implies_phrase_matching():
    # Equality on a multi-word value is meaningless if the backend may split it.
    assert parse("subject=test case").is_phrase is True


def test_an_unquoted_multiword_value_is_not_a_phrase():
    assert parse("subject:test case").is_phrase is False


def test_a_field_with_no_value_is_an_error():
    with pytest.raises(QueryError):
        parse("subject:")


# --------------------------------------------------------------------------- #
# negation push-down
# --------------------------------------------------------------------------- #

def test_not_over_and_becomes_or_over_nots():
    pushed = push_not_down(parse("-(a & b)"))
    assert isinstance(pushed, Or)
    assert all(isinstance(c, Not) for c in pushed.children)


def test_not_over_or_becomes_and_over_nots():
    pushed = push_not_down(parse("-(a | b)"))
    assert isinstance(pushed, And)


def test_double_negation_cancels():
    assert push_not_down(Not(Not(parse("a")))) == parse("a")


def test_push_down_leaves_a_negated_leaf_alone():
    node = parse("-a")
    assert push_not_down(node) == node


# --------------------------------------------------------------------------- #
# the predicate emitter
# --------------------------------------------------------------------------- #

TICKET = {
    "id": "T-1",
    "subject": "Promotional credits disappeared",
    "correspondence": "My credit balance shows zero this morning.",
    "queue": "unassigned",
    "language": "en",
    "severity": "low",
    "minutes": 15,
}


def matches(expression: str, ticket: dict = TICKET) -> bool:
    return to_predicate(parse(expression))(ticket)


def test_contains_is_case_insensitive():
    assert matches("subject:CREDITS")


def test_equals_requires_the_whole_field():
    assert matches("queue=unassigned")
    assert not matches("queue=unass")


def test_contains_does_not_require_the_whole_field():
    assert matches("queue:unass")


def test_an_unquoted_multiword_value_requires_all_words():
    # This is the reading that makes adding a word narrow the results. The
    # opposite reading -- OR over the words -- means `body:missing credits`
    # matches more tickets than `body:credits`, so a filter that looks like a
    # narrowing is a widening.
    assert matches("subject:credits disappeared")
    assert not matches("subject:credits refunded")


def test_a_quoted_value_must_appear_in_order():
    assert matches('subject:"credits disappeared"')
    assert not matches('subject:"disappeared credits"')


def test_free_text_searches_the_long_form_fields():
    assert matches("balance")          # only in correspondence
    assert not matches("unassigned")   # only in a metadata field


def test_and_or_not_compose():
    assert matches("queue=unassigned & language=en")
    assert matches("queue=nonsense | language=en")
    assert not matches("queue=unassigned & -language=en")


def test_range_comparison_on_a_number():
    assert matches("minutes<30")
    assert not matches("minutes>30")


def test_range_comparison_falls_back_to_string_order():
    ticket = {"created": "2026-08-15T00:00:00Z"}
    assert to_predicate(parse("created<2026-09-01"))(ticket)


def test_a_missing_field_never_matches():
    assert not matches("nosuchfield:anything")


# --------------------------------------------------------------------------- #
# the JSON emitter and the debug helpers
# --------------------------------------------------------------------------- #

def test_json_emitter_labels_the_match_kind():
    leaf = to_json(parse('subject:"two words"'))
    assert leaf == {"field": "subject", "match": "phrase", "value": "two words"}
    assert to_json(parse("subject:two words"))["match"] == "words"
    assert to_json(parse("subject=exact"))["match"] == "exact"


def test_json_emitter_nests_branches():
    tree = to_json(parse("a & (b | c)"))
    assert tree["op"] == "AND"
    assert tree["terms"][1]["op"] == "OR"


def test_json_emitter_marks_range_direction():
    leaf = to_json(parse("minutes>30"))
    assert leaf["match"] == "range"
    assert leaf["direction"] == ">"


def test_round_trip_through_text_reparses_to_the_same_tree():
    for expression in ("a & b", "a | b", "-a", "queue=unassigned & -subject:\"test case\""):
        node = parse(expression)
        assert parse(to_text(node)) == node


def test_terms_of_lists_every_leaf():
    names = [t.field for t in terms_of(parse("queue=unassigned & (sev=low | sev=normal)"))]
    assert names == ["queue", "severity", "severity"]
