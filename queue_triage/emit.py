"""Turn a parsed query AST into something that can actually run.

Two emitters ship here:

``to_predicate``
    Compiles to a Python callable that tests one case dict. This is what the
    bundled file backend uses, so the same query syntax works offline with no
    service behind it.

``to_json``
    Compiles to a nested boolean JSON tree, the shape most hosted search APIs
    accept. It is the template for writing an emitter against a real one.

Writing the second kind against a live API is where the surprises live, and they
are worth naming because none of them announce themselves -- each one returns a
plausible wrong answer rather than an error:

- **Field names in the console are not always field names in the API.** A filter
  that quietly matches nothing looks identical to a filter that matches nothing.
  Keep a mapping (:data:`queue_triage.query.FIELD_ALIASES`) instead of editing
  every saved query.
- **An unquoted multi-word value may be tokenised and OR'd.** If so,
  ``body:missing credits`` matches *more* than ``body:credits`` -- a filter that
  reads like a narrowing is a widening. :attr:`~queue_triage.query.Term.is_phrase`
  exists so an emitter can force the grouping instead of hoping.
- **NOT often does not exist.** Backends that lack it expect negation pushed down
  onto the leaves, which is why :func:`push_not_down` is a separate pass.

The reason all of this is in the emitter and not the parser is that it is
per-backend. The parser stays honest and testable; a new backend is one function.
"""

from __future__ import annotations

import re
from typing import Callable

from .query import And, Not, Or, Term

# Fields that hold long-form text (the customer's message), as opposed to the
# short metadata fields. Backends usually store these separately.
TEXT_FIELDS = {"correspondence", "subject", "description"}


# --------------------------------------------------------------------------- #
# negation as a rewrite pass
# --------------------------------------------------------------------------- #

def push_not_down(node):
    """Rewrite the tree so ``Not`` only ever wraps a :class:`Term`.

    De Morgan, applied until the negations reach the leaves. Backends without a
    NOT operator need this; ones with a real NOT can skip it.

    Note the honest limitation: a backend whose only negation is a ``-`` prefix on
    a leaf value cannot express "not (A or B)" without also supporting AND over
    the negated leaves. This pass produces the correct tree; whether the backend
    can run it is a separate question.
    """
    if isinstance(node, Term):
        return node
    if isinstance(node, (And, Or)):
        return type(node)(tuple(push_not_down(c) for c in node.children))
    if isinstance(node, Not):
        inner = node.child
        if isinstance(inner, Term):
            return node
        if isinstance(inner, Not):
            return push_not_down(inner.child)          # double negative
        flipped = Or if isinstance(inner, And) else And
        return flipped(tuple(push_not_down(Not(c)) for c in inner.children))
    raise TypeError(f"not a query node: {node!r}")


# --------------------------------------------------------------------------- #
# emitter: Python predicate
# --------------------------------------------------------------------------- #

def _value_matches(term: Term, haystack: str) -> bool:
    needle = term.value.lower()
    haystack = haystack.lower()

    if term.op == "=":
        return haystack.strip() == needle
    if term.op == ">":
        return _compare(haystack, needle, greater=True)
    if term.op == "<":
        return _compare(haystack, needle, greater=False)

    if term.is_phrase:
        return needle in haystack
    # An unquoted multi-word value means "all of these words", which is the
    # reading that makes adding a word narrow the result set.
    return all(word in haystack for word in needle.split())


def _compare(left: str, right: str, greater: bool) -> bool:
    try:
        lhs: object = float(left)
        rhs: object = float(right)
    except ValueError:
        lhs, rhs = left, right      # ISO dates compare correctly as strings
    return lhs > rhs if greater else lhs < rhs


def _searchable(case: dict, field: str) -> str:
    value = case.get(field)
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


def _free_text_haystack(case: dict) -> str:
    return " ".join(_searchable(case, f) for f in TEXT_FIELDS if case.get(f))


def to_predicate(node) -> Callable[[dict], bool]:
    """Compile the AST into ``predicate(case) -> bool``."""
    if isinstance(node, Term):
        if node.is_free_text:
            return lambda case: _value_matches(node, _free_text_haystack(case))
        return lambda case: _value_matches(node, _searchable(case, node.field))

    if isinstance(node, Not):
        inner = to_predicate(node.child)
        return lambda case: not inner(case)

    if isinstance(node, And):
        parts = [to_predicate(c) for c in node.children]
        return lambda case: all(p(case) for p in parts)

    if isinstance(node, Or):
        parts = [to_predicate(c) for c in node.children]
        return lambda case: any(p(case) for p in parts)

    raise TypeError(f"not a query node: {node!r}")


# --------------------------------------------------------------------------- #
# emitter: nested JSON
# --------------------------------------------------------------------------- #

def to_json(node) -> dict:
    """Compile the AST into a nested boolean JSON tree.

    The shape is the common denominator: ``{"op": "AND"|"OR"|"NOT", "terms": [...]}``
    for branches and ``{"field": ..., "match": "phrase"|"words"|"exact", "value": ...}``
    for leaves. Adapt the leaf and branch keys to whatever your API wants; the
    traversal does not change.
    """
    if isinstance(node, Term):
        match = "exact" if node.op == "=" else "phrase" if node.is_phrase else "words"
        if node.op in ("<", ">"):
            match = "range"
        leaf = {
            "field": node.field or "*",
            "match": match,
            "value": node.value,
        }
        if node.op in ("<", ">"):
            leaf["direction"] = node.op
        return leaf

    if isinstance(node, Not):
        return {"op": "NOT", "terms": [to_json(node.child)]}
    if isinstance(node, (And, Or)):
        name = "AND" if isinstance(node, And) else "OR"
        return {"op": name, "terms": [to_json(c) for c in node.children]}

    raise TypeError(f"not a query node: {node!r}")


# --------------------------------------------------------------------------- #
# handy for debugging a query
# --------------------------------------------------------------------------- #

def to_text(node) -> str:
    """Render the AST back as an expression, for eyeballing what was parsed."""
    if isinstance(node, Term):
        value = f'"{node.value}"' if node.quoted or " " in node.value else node.value
        return f"{node.field}{node.op}{value}" if node.field else value
    if isinstance(node, Not):
        return f"-{to_text(node.child)}"
    joiner = " & " if isinstance(node, And) else " | "
    return "(" + joiner.join(to_text(c) for c in node.children) + ")"


_WORD = re.compile(r"\w+")


def terms_of(node) -> list[Term]:
    """Every leaf in the tree, for reporting which filters a query applies."""
    if isinstance(node, Term):
        return [node]
    if isinstance(node, Not):
        return terms_of(node.child)
    return [t for c in node.children for t in terms_of(c)]
