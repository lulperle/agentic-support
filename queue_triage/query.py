"""Compile a compact search expression into an abstract syntax tree.

The syntax is the Lucene-flavoured one that most ticketing consoles expose:

    (queue=unassigned) & (language=en) & -subject:"test case"

    &   AND
    |   OR
    -   NOT
    :   field contains value
    =   field equals value
    ""  phrase -- match the words together, in order
    ()  grouping

Precedence is ``-`` > ``&`` > ``|``.

This module deliberately knows nothing about where the query will run. It stops
at the AST; turning that into a backend's wire format is :mod:`queue_triage.emit`.
The split is not decoration. Every real search API has quirks in how it wants a
nested boolean tree encoded -- which side of a pair gets wrapped in an envelope,
whether NOT exists at all, what an unquoted multi-word value means -- and those
quirks are numerous enough that mixing them into the parser makes both halves
untestable. With the AST in between, the parser has one job and each backend's
weirdness is contained in one function.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class QueryError(ValueError):
    """Raised when an expression cannot be parsed."""


# --------------------------------------------------------------------------- #
# AST
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Term:
    """A single field test, or free text when ``field`` is empty."""

    field: str
    op: str          # ":" contains, "=" equals, ">"/"<" range, "" free text
    value: str
    quoted: bool = False   # the user wrote the value in double quotes

    @property
    def is_free_text(self) -> bool:
        return not self.field

    @property
    def is_phrase(self) -> bool:
        """Whether the value must be matched as one unit.

        An explicit quote says so. So does ``=``: equality on a multi-word value
        is meaningless if the backend is free to split it into words.
        """
        return self.quoted or self.op == "=" or " " not in self.value


@dataclass(frozen=True)
class Not:
    child: object


@dataclass(frozen=True)
class And:
    children: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class Or:
    children: tuple = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #

_OPS = {"&": And, "|": Or}
_PRECEDENCE = {"-": 3, "&": 2, "|": 1}


def tokenize(expression: str) -> list[tuple[str, str]]:
    """Split an expression into ``(kind, text)`` pairs.

    ``kind`` is ``(``, ``)``, ``&``, ``|``, ``-`` or ``TERM``.

    Two things here are easy to get wrong. Terms may contain spaces, because
    ``subject:test case`` is a single criterion, so a term accumulates until the
    next operator rather than the next space. And ``-`` is only NOT when it opens
    a term -- otherwise ``free-tier`` and ``us-east-1`` would parse as negations.
    """
    tokens: list[tuple[str, str]] = []
    buf = ""
    i = 0
    while i < len(expression):
        ch = expression[i]

        if ch == "\\" and i + 1 < len(expression):
            buf += expression[i + 1]
            i += 2
            continue

        if ch == '"':
            end = expression.find('"', i + 1)
            if end == -1:
                raise QueryError(f"unterminated quote at offset {i}")
            buf += expression[i : end + 1]   # keep the quotes; they carry meaning
            i = end + 1
            continue

        if ch in "()&|":
            if buf.strip():
                tokens.append(("TERM", buf.strip()))
                buf = ""
            tokens.append((ch, ch))
            i += 1
            continue

        if ch == "-" and not buf.strip():
            tokens.append(("-", "-"))
            i += 1
            continue

        buf += ch
        i += 1

    if buf.strip():
        tokens.append(("TERM", buf.strip()))
    return tokens


# --------------------------------------------------------------------------- #
# shunting-yard
# --------------------------------------------------------------------------- #

def to_rpn(tokens: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Reorder infix tokens into postfix, honouring precedence and parentheses."""
    out: list[tuple[str, str]] = []
    stack: list[tuple[str, str]] = []
    for kind, text in tokens:
        if kind == "TERM":
            out.append((kind, text))
        elif kind == "-":
            stack.append((kind, text))
        elif kind in _OPS:
            while (
                stack
                and stack[-1][0] in _PRECEDENCE
                and _PRECEDENCE[stack[-1][0]] >= _PRECEDENCE[kind]
            ):
                out.append(stack.pop())
            stack.append((kind, text))
        elif kind == "(":
            stack.append((kind, text))
        elif kind == ")":
            while stack and stack[-1][0] != "(":
                out.append(stack.pop())
            if not stack:
                raise QueryError("unbalanced ')'")
            stack.pop()
            # A NOT sitting immediately before the group negates the whole group.
            if stack and stack[-1][0] == "-":
                out.append(stack.pop())

    while stack:
        kind, text = stack.pop()
        if kind == "(":
            raise QueryError("unbalanced '('")
        out.append((kind, text))
    return out


# --------------------------------------------------------------------------- #
# leaves
# --------------------------------------------------------------------------- #

_FIELD_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")

# Consoles tend to show short friendly names for longer stored ones. Extend this
# rather than rewriting your saved queries.
FIELD_ALIASES = {
    "sev": "severity",
    "lang": "language",
    "body": "correspondence",
    "content": "correspondence",
}


def _split_field(term: str) -> tuple[str, str, str] | None:
    """Find the first ``:``, ``=``, ``>`` or ``<`` that follows a bare field name."""
    for index, ch in enumerate(term):
        if ch in ":=><":
            name = term[:index].strip()
            if name and all(c in _FIELD_CHARS for c in name) and name[0].isalpha():
                return name, ch, term[index + 1 :].strip()
            return None
        if ch == '"':
            return None
    return None


def _unquote(value: str) -> tuple[str, bool]:
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1], True
    return value, False


def parse_term(term: str) -> Term:
    """Turn one leaf token into a :class:`Term`."""
    split = _split_field(term)
    if split is None:
        value, quoted = _unquote(term)
        return Term(field="", op="", value=value, quoted=quoted)

    name, op, raw = split
    value, quoted = _unquote(raw)
    if not value:
        raise QueryError(f"field {name!r} has no value")
    return Term(
        field=FIELD_ALIASES.get(name.lower(), name),
        op=op,
        value=value,
        quoted=quoted,
    )


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #

def _combine(node_type, left, right):
    """Flatten same-operator nesting, so ``a & b & c`` is one node, not two."""
    children: list = []
    for side in (left, right):
        if isinstance(side, node_type):
            children.extend(side.children)
        else:
            children.append(side)
    return node_type(tuple(children))


def build(rpn: list[tuple[str, str]]):
    stack: list = []
    for kind, text in rpn:
        if kind == "TERM":
            stack.append(parse_term(text))
        elif kind == "-":
            if not stack:
                raise QueryError("'-' with nothing to negate")
            stack.append(Not(stack.pop()))
        else:
            if len(stack) < 2:
                raise QueryError(f"operator {text!r} needs two operands")
            right = stack.pop()
            left = stack.pop()
            stack.append(_combine(_OPS[kind], left, right))

    if len(stack) != 1:
        raise QueryError("could not reduce the expression to a single query")
    return stack[0]


def parse(expression: str):
    """Expression string -> AST."""
    tokens = tokenize(expression)
    if not tokens:
        raise QueryError("empty query")
    return build(to_rpn(tokens))


def conjoin(node, *extra):
    """AND more nodes onto an existing tree."""
    result = node
    for fragment in extra:
        result = _combine(And, result, fragment)
    return result
