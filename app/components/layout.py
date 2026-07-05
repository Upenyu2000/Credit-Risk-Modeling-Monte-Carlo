# Author: Upenyu Hlangabeza
"""Layout primitives and visual shell for the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.config import APP_SUBTITLE, APP_TITLE, PAGE_NAMES, THEME_PATH


def inject_theme(theme_path: Path = THEME_PATH) -> None:
    """Inject the JetBrains-inspired CSS theme."""
    if theme_path.exists():
        st.markdown(f"<style>{theme_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_sidebar() -> str:
    """Render global navigation and return the selected page."""
    with st.sidebar:
        st.markdown(
            f"""
            <div class="cr-hero">
                <div class="cr-kicker">Credit Risk Studio</div>
                <div class="cr-title">{APP_TITLE}</div>
                <p class="cr-subtitle">{APP_SUBTITLE}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected = st.radio("Navigation", PAGE_NAMES, label_visibility="collapsed")
        st.divider()
        active_model = st.session_state.get("active_model_name") or "Not trained"
        source = st.session_state.get("dataset_source", "Generated demo portfolio")
        st.caption("Workspace")
        st.markdown(f"<span class='cr-badge'>Data: {source}</span>", unsafe_allow_html=True)
        st.markdown(f"<span class='cr-badge'>Model: {active_model}</span>", unsafe_allow_html=True)
        st.caption("Shortcuts: use the sidebar search with / in the browser, then Enter to jump.")
        return selected


def page_header(kicker: str, title: str, subtitle: str) -> None:
    """Render a consistent page heading."""
    st.markdown(
        f"""
        <div class="cr-hero">
            <div class="cr-kicker">{kicker}</div>
            <div class="cr-title">{title}</div>
            <p class="cr-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str | None = None) -> None:
    """Render a compact section title."""
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


def render_step(name: str, details: str, before_rows: int, after_rows: int) -> None:
    """Display one preprocessing step summary."""
    delta = after_rows - before_rows
    sign = "+" if delta >= 0 else ""
    st.markdown(
        f"""
        <div class="cr-step">
            <strong>{name}</strong><br>
            <span class="cr-muted">{details}</span><br>
            <span class="cr-muted">Rows: {before_rows:,} -> {after_rows:,} ({sign}{delta:,})</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
