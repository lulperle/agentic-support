"""Rank a support queue by how cheap each ticket looks to close.

The deterministic half of this repository: no model, no network, every score
explainable as a list of rules that fired. It answers the question that comes
before the triage agent gets involved -- which of these forty untouched tickets
should I open first.
"""
