"""Where tickets come from.

A backend is two methods. That is the whole contract:

    search(node, limit, ...) -> list[dict]
    body(ticket_id)          -> str

``node`` is a parsed query AST (:mod:`queue_triage.query`), not a string, so a
backend translates it with its own emitter and nothing about one backend's query
dialect leaks into the ranker.

:class:`FileBackend` ships here and reads a YAML file, which makes the tool
runnable with no service and no credentials. Point it at a real ticketing system
by writing a second class with the same two methods -- the CLI takes whatever it
is handed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml

from .emit import to_predicate

DATASET = Path(__file__).resolve().parent.parent / "evals" / "queue.yaml"


class BackendError(RuntimeError):
    pass


@runtime_checkable
class Backend(Protocol):
    name: str

    def search(self, node, limit: int = 50, sort_by: str = "created", descending: bool = False) -> list[dict]:
        ...

    def body(self, ticket_id: str) -> str:
        ...

    def url(self, ticket_id: str) -> str:
        ...


class FileBackend:
    """Serve tickets from a YAML file, filtering in-process.

    The same query syntax works here as against a hosted search API, because the
    filtering goes through the same AST -- :func:`queue_triage.emit.to_predicate`
    instead of a JSON emitter. That is worth more than it sounds: it means the
    query language has test coverage that does not depend on a network.
    """

    name = "file"

    def __init__(self, path: str | Path = DATASET):
        self.path = Path(path)
        self._tickets: list[dict] | None = None

    # -- loading ---------------------------------------------------------- #

    @property
    def tickets(self) -> list[dict]:
        if self._tickets is None:
            self._tickets = self._load()
        return self._tickets

    def _load(self) -> list[dict]:
        try:
            raw = yaml.safe_load(self.path.read_text())
        except FileNotFoundError as exc:
            raise BackendError(f"no ticket file at {self.path}") from exc
        tickets = raw.get("tickets") if isinstance(raw, dict) else raw
        if not isinstance(tickets, list):
            raise BackendError(f"{self.path}: expected a list of tickets")
        return tickets

    # -- the contract ----------------------------------------------------- #

    def search(self, node, limit: int = 50, sort_by: str = "created", descending: bool = False) -> list[dict]:
        matches = [t for t in self.tickets if to_predicate(node)(t)]
        matches.sort(key=lambda t: str(t.get(sort_by) or ""), reverse=descending)
        return matches[:limit]

    def count(self, node) -> int:
        """How many tickets match in total, before ``limit`` is applied."""
        predicate = to_predicate(node)
        return sum(1 for t in self.tickets if predicate(t))

    def body(self, ticket_id: str) -> str:
        for ticket in self.tickets:
            if str(ticket.get("id")) == str(ticket_id):
                return str(ticket.get("correspondence") or "")
        return ""

    def url(self, ticket_id: str) -> str:
        return f"{self.path}#{ticket_id}"
