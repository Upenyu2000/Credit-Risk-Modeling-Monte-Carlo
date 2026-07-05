# Author: Upenyu Hlangabeza
"""Formatting helpers for UI labels and metrics."""

from __future__ import annotations


def compact_number(value: float | int) -> str:
    """Format large values for metric cards."""
    numeric = float(value)
    if abs(numeric) >= 1_000_000:
        return f"{numeric / 1_000_000:.1f}M"
    if abs(numeric) >= 1_000:
        return f"{numeric / 1_000:.1f}K"
    return f"{numeric:,.0f}"


def percent(value: float, digits: int = 1) -> str:
    """Format a decimal as a percentage."""
    return f"{value * 100:.{digits}f}%"


def titleize(name: str) -> str:
    """Convert a feature name into a readable label."""
    return name.replace("_", " ").title()


def risk_category(probability: float) -> str:
    """Map default probability to a business-friendly risk category."""
    if probability < 0.08:
        return "Low"
    if probability < 0.18:
        return "Moderate"
    if probability < 0.35:
        return "Elevated"
    return "High"
