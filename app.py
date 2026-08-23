"""
app.py — TIGA Hunt Admin Panel (Streamlit).

Admin-only. Accessed from the settings drawer in the main UI.
Password: admin / admin

Run via: python tiga.py ui
"""

from __future__ import annotations

from typing import Any

import requests
import streamlit as st

from config import cfg

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TIGA Admin",
    page_icon=" ",
    layout="wide",
)

# Shared brand mark (slender cat — matches search portal)
CAT_SVG = """
<svg class="tiga-cat-svg" viewBox="0 0 24 32" fill="none" aria-hidden="true">
  <g>
    <path d="M6.2 28.8 C1.8 26.2 2.2 20.8 4.4 17.2 C5.2 21.5 5.8 25.8 7.4 27.6" stroke="#C96442" stroke-width="1.65" fill="none" stroke-linecap="round"/>
    <path d="M12 13.8 C9.4 13.8 8.8 16.8 8.8 21.8 C8.8 27.2 9.8 29.8 12 29.8 C14.2 29.8 15.2 27.2 15.2 21.8 C15.2 16.8 14.6 13.8 12 13.8 Z" fill="#C96442"/>
    <circle cx="12" cy="9.2" r="4.9" fill="#C96442"/>
    <path d="M8.4 6.2 L6.8 0.8 L10.6 5.4 Z" fill="#C96442"/>
    <path d="M15.6 6.2 L17.2 0.8 L13.4 5.4 Z" fill="#C96442"/>
    <ellipse cx="10.35" cy="9.5" rx=".95" ry="1.65" fill="white" opacity=".92"/>
    <ellipse cx="13.65" cy="9.5" rx=".95" ry="1.65" fill="white" opacity=".92"/>
    <ellipse cx="10.35" cy="9.5" rx=".38" ry="1.35" fill="#1A1410" opacity=".88"/>
    <ellipse cx="13.65" cy="9.5" rx=".38" ry="1.35" fill="#1A1410" opacity=".88"/>
  </g>
</svg>
"""

# ── Design system: match main UI fonts + coral accents ──────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,300;8..60,400&display=swap');

:root {
  --tiga-coral: #C96442;
  --tiga-coral-dim: rgba(201,100,66,0.12);
  --tiga-coral-glow: rgba(201,100,66,0.18);
  --tiga-ink: #1A1410;
  --tiga-ink-60: rgba(26,20,16,0.52);
  --tiga-ink-12: rgba(26,20,16,0.09);
  --tiga-page: #FCFAF7;
  --tiga-surface: rgba(255,255,255,0.78);
  --tiga-border: rgba(26,20,16,0.09);
  --tiga-glass-shadow: 0 8px 32px rgba(0,0,0,0.06), 0 1.5px 4px rgba(0,0,0,0.03);
}

html, body, [class*="css"], .stMarkdown, .stText, .stCaption,
.stDataFrame, div[data-testid="stMetricLabel"], div[data-testid="stMetricValue"] {
    font-family: 'Source Serif 4', Georgia, serif !important;
}
h1, h2, h3, h4, .stSubheader, div[data-testid="stHeading"] {
    font-family: 'Lora', Georgia, serif !important;
    font-weight: 400 !important;
    letter-spacing: -0.01em;
}
#MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden;}
.block-container {padding-top: 0.75rem; padding-bottom: 3rem; max-width: 1080px;}
.stApp {background: var(--tiga-page) !important; color: var(--tiga-ink) !important;}
.tiga-blobs {position: fixed; inset: 0; pointer-events: none; z-index: 0; overflow: hidden;}
.tiga-blob {position: absolute; border-radius: 50%; filter: blur(90px);}
.tiga-blob-1 {width: 420px; height: 420px; background: rgba(201,100,66,0.09); top: -100px; left: -60px;}
.tiga-blob-2 {width: 340px; height: 340px; background: rgba(255,200,170,0.10); bottom: -60px; right: -40px;}
.tiga-admin-header {position: relative; z-index: 1; display: flex; align-items: center; justify-content: space-between; padding: 8px 0 20px; margin-bottom: 4px; border-bottom: 1px solid var(--tiga-border);}
.tiga-brand {display: flex; align-items: center; gap: 10px; color: var(--tiga-ink);}
.tiga-cat-wrap {width: 26px; height: 34px; flex-shrink: 0;}
.tiga-cat-svg {width: 26px; height: 34px; display: block;}
.tiga-brand-name {font-family: 'Lora', Georgia, serif; font-size: 1.35rem; font-weight: 500;}
.tiga-brand-sub {font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--tiga-coral); margin-top: 1px;}
.tiga-portal-link {font-size: 13px; color: var(--tiga-ink-60); text-decoration: none; padding: 7px 14px; border: 1px solid var(--tiga-border); border-radius: 100px; background: var(--tiga-surface);}
.stButton > button[kind="primary"] {
    background-color: var(--tiga-coral) !important;
    border-color: var(--tiga-coral) !important;
    color: white !important;
    font-family: 'Source Serif 4', Georgia, serif !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 8px rgba(201,100,66,0.22) !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #a8522f !important;
    border-color: #a8522f !important;
}
.stButton > button {
    font-family: 'Source Serif 4', Georgia, serif !important;
    border-radius: 10px !important;
    border-color: var(--tiga-border) !important;
    background: var(--tiga-surface) !important;
}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    border-bottom-color: var(--tiga-coral) !important;
    color: var(--tiga-coral) !important;
    font-family: 'Source Serif 4', Georgia, serif !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Source Serif 4', Georgia, serif !important;
    font-size: 14px !important;
    color: var(--tiga-ink-60) !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'Lora', Georgia, serif !important;
    font-size: 1.45rem !important;
}
.stProgress > div > div { background-color: var(--tiga-coral) !important; }
.stTextInput input, .stTextArea textarea, .stNumberInput input {
    font-family: 'Source Serif 4', Georgia, serif !important;
    border-radius: 10px !important;
    border-color: var(--tiga-border) !important;
    background: var(--tiga-surface) !important;
}
section[data-testid="stSidebar"] {
    background: rgba(252,250,247,0.97) !important;
    border-right: 1px solid var(--tiga-border);
}
.stDivider {border-color: var(--tiga-border) !important; margin: 1.25rem 0 !important;}
.tiga-login-logo {display: flex; justify-content: center; margin-bottom: 14px;}
.tiga-login-title {
    font-family: 'Lora', Georgia, serif;
    font-size: 2rem;
    font-weight: 400;
    text-align: center;
    margin-bottom: 4px;
}
.tiga-login-sub {
    text-align: center;
    color: var(--tiga-ink-60);
    font-size: 14px;
    margin-bottom: 28px;
}
.tiga-status-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 20px;
}
.tiga-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 13px;
    border-radius: 100px;
    border: 1px solid var(--tiga-border);
    font-size: 12.5px;
    background: var(--tiga-surface);
    box-shadow: var(--tiga-glass-shadow);
}
.tiga-status-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.tiga-dot-ok { background: #34a853; }
.tiga-dot-warn { background: #fbbc04; }
.tiga-dot-err { background: #ea4335; }
.tiga-empty {
    text-align: center;
    padding: 36px 20px;
    color: var(--tiga-ink-60);
    border: 1px dashed var(--tiga-border);
    border-radius: 14px;
    font-size: 14px;
    background: rgba(255,255,255,0.35);
}
.tiga-login-wrap {
    max-width: 380px;
    margin: 3rem auto 0;
    padding: 36px 30px;
    background: var(--tiga-surface);
    border: 1px solid var(--tiga-border);
    border-radius: 18px;
    box-shadow: var(--tiga-glass-shadow);
    position: relative;
    z-index: 1;
}
.tiga-roadmap-grid {display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px;}
@media (max-width: 768px) { .tiga-roadmap-grid { grid-template-columns: 1fr; } }
.tiga-roadmap-card {background: var(--tiga-surface); border: 1px solid var(--tiga-border); border-radius: 14px; padding: 16px; box-shadow: var(--tiga-glass-shadow);}
.tiga-roadmap-card.live {border-top: 2px solid var(--tiga-coral);}
.tiga-roadmap-card.build {border-top: 2px solid #fbbc04;}
.tiga-roadmap-card.plan {border-top: 2px solid var(--tiga-ink-12);}
.tiga-roadmap-badge {font-size: 10px; font-weight: 500; letter-spacing: 0.11em; text-transform: uppercase; color: var(--tiga-coral); margin-bottom: 6px;}
.tiga-roadmap-title {font-family: 'Lora', Georgia, serif; font-size: 1.15rem; margin-bottom: 4px;}
.tiga-roadmap-detail {font-size: 13px; color: var(--tiga-ink-60); line-height: 1.5; margin-bottom: 12px; min-height: 2.6em;}
.tiga-roadmap-stat {font-family: 'Lora', Georgia, serif; font-size: 1.55rem;}
.tiga-roadmap-stat-label {font-size: 11px; color: var(--tiga-ink-60); margin-top: 2px;}
.tiga-layer-grid {display: grid; grid-template-columns: 1fr 1fr 1fr; border: 1px solid var(--tiga-border); border-radius: 12px; overflow: hidden; background: var(--tiga-surface); margin-bottom: 8px; box-shadow: var(--tiga-glass-shadow);}
.tiga-layer-cell {padding: 12px 14px; font-size: 13px; border-bottom: 1px solid var(--tiga-border); border-right: 1px solid var(--tiga-border);}
.tiga-layer-cell:nth-child(3n) {border-right: none;}
.tiga-layer-head {font-size: 10px; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; color: var(--tiga-coral); background: rgba(201,100,66,0.04);}
.tiga-layer-name {font-family: 'Lora', Georgia, serif;}
.tiga-metric-grid {display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 8px;}
.tiga-metric-cell {background: var(--tiga-surface); border: 1px solid var(--tiga-border); border-radius: 12px; padding: 12px 14px; box-shadow: var(--tiga-glass-shadow);}
.tiga-metric-value {font-family: 'Lora', Georgia, serif; font-size: 1.35rem;}
.tiga-metric-label {font-size: 11px; color: var(--tiga-ink-60); margin-top: 2px;}
.tiga-panel {background: var(--tiga-surface); border: 1px solid var(--tiga-border); border-radius: 14px; padding: 16px 18px; margin-bottom: 14px; box-shadow: var(--tiga-glass-shadow);}
.tiga-live-panel {background: var(--tiga-surface); border: 1px solid var(--tiga-border); border-radius: 14px; padding: 16px 18px; margin-top: 8px; box-shadow: var(--tiga-glass-shadow);}
.tiga-dir-meta {display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--tiga-ink-60); margin-top: 6px;}
.tiga-atlas-card {background: var(--tiga-surface); border: 1px solid var(--tiga-border); border-radius: 14px; padding: 14px 16px; margin-bottom: 10px; box-shadow: var(--tiga-glass-shadow);}
.tiga-atlas-head {display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 8px;}
.tiga-atlas-code {font-family: 'Lora', Georgia, serif; font-size: 1.05rem;}
.tiga-atlas-pct {font-size: 12px; color: var(--tiga-coral); white-space: nowrap;}
.tiga-atlas-bar {height: 4px; border-radius: 2px; background: var(--tiga-ink-12); overflow: hidden; margin-bottom: 10px;}
.tiga-atlas-bar-fill {height: 100%; background: var(--tiga-coral); border-radius: 2px;}
.tiga-atlas-fields {font-size: 13px; color: var(--tiga-ink-60); line-height: 1.55;}
.tiga-atlas-missing {font-size: 12px; color: #c0392b; margin-top: 8px;}
.tiga-check-row {display: flex; gap: 10px; align-items: flex-start; padding: 10px 0; border-bottom: 1px solid var(--tiga-ink-12); font-size: 13px;}
.tiga-check-dot {width: 8px; height: 8px; border-radius: 50%; margin-top: 5px; flex-shrink: 0;}
.tiga-check-ok {background: #34a853;}
.tiga-check-fail {background: #ea4335;}
.tiga-audit-row {font-size: 13px; padding: 8px 0; border-bottom: 1px solid var(--tiga-ink-12); line-height: 1.45;}
.tiga-section-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--tiga-coral);
    margin: 0 0 10px 0;
}
.tiga-tab-intro {
    color: var(--tiga-ink-60);
    font-size: 14px;
    margin: -6px 0 20px 0;
    line-height: 1.58;
}
.main div[data-testid="stMetric"] {
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 0;
}
.tiga-dir-card {
    background: var(--tiga-surface);
    border: 1px solid var(--tiga-border);
    border-radius: 14px;
    padding: 14px 16px 10px;
    margin-bottom: 8px;
    box-shadow: var(--tiga-glass-shadow);
}
.tiga-dir-path {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 14px;
    word-break: break-all;
}
.tiga-danger-note {
    font-size: 13px;
    color: #ea4335;
    margin-bottom: 8px;
}
@media (prefers-color-scheme: dark) {
    :root {
        --tiga-ink: #EDE8E0;
        --tiga-ink-60: rgba(237,232,224,0.52);
        --tiga-ink-12: rgba(237,232,224,0.08);
        --tiga-page: #0F0E0C;
        --tiga-surface: rgba(30,26,22,0.88);
        --tiga-border: rgba(237,232,224,0.10);
        --tiga-glass-shadow: 0 8px 32px rgba(0,0,0,0.28);
    }
    .stApp { background: var(--tiga-page) !important; color: var(--tiga-ink) !important; }
    section[data-testid="stSidebar"] {
        background: rgba(22,18,14,0.98) !important;
    }
    .tiga-blob-1 { background: rgba(201,100,66,0.14); }
    .tiga-blob-2 { background: rgba(140,80,50,0.08); }
}
</style>
<div class="tiga-blobs" aria-hidden="true">
  <div class="tiga-blob tiga-blob-1"></div>
  <div class="tiga-blob tiga-blob-2"></div>
</div>
""", unsafe_allow_html=True)

_API = f"http://localhost:{cfg.server_port}"
_SEARCH_UI = f"http://localhost:{cfg.server_port}"


def empty_state(msg: str) -> None:
    st.markdown(f'<div class="tiga-empty">{msg}</div>', unsafe_allow_html=True)


def status_pill(label: str, ok: bool | None) -> str:
    dot = "tiga-dot-ok" if ok else ("tiga-dot-warn" if ok is None else "tiga-dot-err")
    return f'<span class="tiga-status-pill"><span class="tiga-status-dot {dot}"></span>{label}</span>'


def tab_intro(text: str) -> None:
    st.markdown(f'<p class="tiga-tab-intro">{text}</p>', unsafe_allow_html=True)


def section_label(text: str) -> None:
    st.markdown(f'<p class="tiga-section-label">{text}</p>', unsafe_allow_html=True)


def brand_header(portal_url: str) -> None:
    st.markdown(
        f'<div class="tiga-admin-header">'
        f'<div class="tiga-brand">'
        f'<div class="tiga-cat-wrap">{CAT_SVG}</div>'
        f'<div><div class="tiga-brand-name">TIGA</div>'
        f'<div class="tiga-brand-sub">Administration</div></div></div>'
        f'<a class="tiga-portal-link" href="{portal_url}" target="_self">← Search portal</a>'
        f'</div>',
        unsafe_allow_html=True,
    )


def roadmap_cards(hunt: dict, atlas: dict, ein: dict, status: dict) -> None:
    indexed = hunt.get("files_indexed") or status.get("files_indexed") or 0
    cards_n = atlas.get("project_cards") or 0
    ein_label = "Enabled" if ein.get("enabled") else "Phase 2"
    st.markdown(
        f'<div class="tiga-roadmap-grid">'
        f'<div class="tiga-roadmap-card live">'
        f'<div class="tiga-roadmap-badge">Live</div>'
        f'<div class="tiga-roadmap-title">Hunt</div>'
        f'<div class="tiga-roadmap-detail">{hunt.get("detail", "Search + indexing")}</div>'
        f'<div class="tiga-roadmap-stat">{indexed:,}</div>'
        f'<div class="tiga-roadmap-stat-label">Indexed files</div></div>'
        f'<div class="tiga-roadmap-card build">'
        f'<div class="tiga-roadmap-badge">Building</div>'
        f'<div class="tiga-roadmap-title">Atlas</div>'
        f'<div class="tiga-roadmap-detail">{atlas.get("detail", "Project memory")}</div>'
        f'<div class="tiga-roadmap-stat">{cards_n:,}</div>'
        f'<div class="tiga-roadmap-stat-label">Project cards</div></div>'
        f'<div class="tiga-roadmap-card plan">'
        f'<div class="tiga-roadmap-badge">Planned</div>'
        f'<div class="tiga-roadmap-title">Einstein</div>'
        f'<div class="tiga-roadmap-detail">{ein.get("detail", "Expert reasoning — Phase 2")}</div>'
        f'<div class="tiga-roadmap-stat">{ein_label}</div>'
        f'<div class="tiga-roadmap-stat-label">Status</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def layer_grid() -> None:
    st.markdown(
        '<div class="tiga-layer-grid">'
        '<div class="tiga-layer-cell tiga-layer-head">Layer</div>'
        '<div class="tiga-layer-cell tiga-layer-head">Status</div>'
        '<div class="tiga-layer-cell tiga-layer-head">Admin controls</div>'
        '<div class="tiga-layer-cell tiga-layer-name">Hunt</div>'
        '<div class="tiga-layer-cell">Live</div>'
        '<div class="tiga-layer-cell">Pipeline, Archives, Schedule, Quality</div>'
        '<div class="tiga-layer-cell tiga-layer-name">Atlas</div>'
        '<div class="tiga-layer-cell">Foundation</div>'
        '<div class="tiga-layer-cell">Project cards (this panel)</div>'
        '<div class="tiga-layer-cell tiga-layer-name">Einstein</div>'
        '<div class="tiga-layer-cell">Planned</div>'
        '<div class="tiga-layer-cell">Config flag only — not shipping yet</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def metric_row(items: list[tuple[str, str]]) -> None:
    cells = "".join(
        f'<div class="tiga-metric-cell"><div class="tiga-metric-value">{val}</div>'
        f'<div class="tiga-metric-label">{label}</div></div>'
        for label, val in items
    )
    st.markdown(f'<div class="tiga-metric-grid">{cells}</div>', unsafe_allow_html=True)


def atlas_card_html(card: dict) -> str:
    code = card.get("project_code", "?")
    name = card.get("name") or "Untitled"
    miss = card.get("missing_fields") or []
    pct = card.get("completeness", 0)
    fields = (
        f"Typology: {card.get('typology_primary') or '—'} · "
        f"Location: {card.get('location') or '—'} · "
        f"Stage: {card.get('stage') or '—'} · "
        f"Client: {card.get('client') or '—'}"
    )
    missing_html = (
        f'<div class="tiga-atlas-missing">Missing: {", ".join(miss)}</div>'
        if miss else '<div class="tiga-atlas-fields" style="color:#34a853">All required fields filled</div>'
    )
    return (
        f'<div class="tiga-atlas-card">'
        f'<div class="tiga-atlas-head"><div class="tiga-atlas-code">{code} · {name}</div>'
        f'<div class="tiga-atlas-pct">{pct}% complete</div></div>'
        f'<div class="tiga-atlas-bar"><div class="tiga-atlas-bar-fill" style="width:{pct}%"></div></div>'
        f'<div class="tiga-atlas-fields">{fields}</div>{missing_html}'
        f'<div class="tiga-atlas-fields" style="margin-top:8px;font-size:12px">'
        f'Edit: <code>python tiga.py card {code}</code></div></div>'
    )


def pipeline_trigger(label: str, path: str, audit_action: str, *, key: str) -> None:
    if st.button(label, use_container_width=True, key=key):
        r = api("post", path)
        if r:
            api("post", "/api/audit/log", json={"action": audit_action})
            st.toast(f"{label}: {r.get('status', 'ok')}")


def confirm_and_run(
    label: str,
    confirm_key: str,
    warning: str,
    path: str,
    audit_action: str,
    *,
    btn_key: str,
) -> None:
    if st.session_state.get(confirm_key):
        st.markdown(f'<p class="tiga-danger-note">{warning}</p>', unsafe_allow_html=True)
        yes, no = st.columns(2)
        with yes:
            if st.button("Confirm", type="primary", key=f"{btn_key}_yes", use_container_width=True):
                r = api("post", path)
                if r:
                    api("post", "/api/audit/log", json={"action": audit_action})
                    st.toast(f"{label}: {r.get('status', 'ok')}")
                st.session_state.pop(confirm_key, None)
                st.rerun()
        with no:
            if st.button("Cancel", key=f"{btn_key}_no", use_container_width=True):
                st.session_state.pop(confirm_key, None)
                st.rerun()
    elif st.button(label, key=btn_key, use_container_width=True):
        st.session_state[confirm_key] = True
        st.rerun()


# ---------------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------------

def api(method: str, path: str, **kwargs) -> Any:
    url = _API + path
    try:
        resp = getattr(requests, method)(url, timeout=kwargs.pop("timeout", 30), **kwargs)
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        if "json" in ct:
            return resp.json()
        return resp.content
    except requests.RequestException as e:
        st.error(f"API error ({path}): {e}")
        return None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "admin_authed" not in st.session_state:
    st.session_state.admin_authed = False

# ---------------------------------------------------------------------------
# Login gate
# ---------------------------------------------------------------------------

if not st.session_state.admin_authed:
    st.markdown('<div class="tiga-login-wrap">', unsafe_allow_html=True)
    st.markdown(f'<div class="tiga-login-logo"><div class="tiga-cat-wrap">{CAT_SVG}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="tiga-login-title">TIGA Admin</div>', unsafe_allow_html=True)
    st.markdown('<div class="tiga-login-sub">Administration panel for TIGA Hunt</div>', unsafe_allow_html=True)

    username = st.text_input("Username", key="login_user", placeholder="admin")
    password = st.text_input("Password", type="password", key="login_pass", placeholder="••••••")

    if st.button("Sign in", type="primary", use_container_width=True):
        if username == "admin" and password == "admin":
            st.session_state.admin_authed = True
            st.rerun()
        else:
            st.error("Invalid credentials.")

    st.caption(f"Search portal: [{_SEARCH_UI}]({_SEARCH_UI})")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------------------------
# Admin panel (only reached when authenticated)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f'<div class="tiga-cat-wrap" style="margin:0 auto 8px">{CAT_SVG}</div>', unsafe_allow_html=True)
    st.markdown("### TIGA")
    st.caption("Hunt · Atlas · Einstein")
    st.caption(f"API `{_API}`")
    st.markdown(f"[← Search portal]({_SEARCH_UI})")
    st.divider()
    try:
        h = requests.get(_API + "/health", timeout=3).json()
        if h.get("ollama"):
            st.success("Ollama online")
        else:
            st.warning("Ollama offline")
    except Exception:
        st.error("Server offline")
    st.divider()
    if st.button("Sign out", use_container_width=True):
        st.session_state.admin_authed = False
        st.rerun()

brand_header(_SEARCH_UI)

# ── Product status ─────────────────────────────────────────────────────────
_product = api("get", "/api/product/status") or {}
_status = api("get", "/api/status") or {}
_sched = api("get", "/api/schedule/status") or {}
_hunt = _product.get("hunt") or {}
_atlas = _product.get("atlas") or {}
_ein = _product.get("einstein") or {}
_health = {}
try:
    _health = requests.get(_API + "/health", timeout=3).json()
except Exception:
    pass

st.markdown(
    '<div class="tiga-status-bar">'
    + status_pill("Server online" if _status else "Server offline", bool(_status))
    + status_pill("Ollama online" if _health.get("ollama") else "Ollama offline", _health.get("ollama"))
    + status_pill(
        f"Schedule: {_sched.get('mode', '?')}",
        True if _sched.get("mode") else None,
    )
    + status_pill(f"{(_status.get('files_indexed') or 0):,} indexed", True if _status.get("files_indexed") else None)
    + "</div>",
    unsafe_allow_html=True,
)

roadmap_cards(_hunt, _atlas, _ein, _status)

tabs = st.tabs([
    "Overview",
    "Hunt",
    "Archives",
    "Schedule",
    "Atlas",
    "Quality",
    "System",
])

# ── TAB: Overview ──────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Product roadmap")
    tab_intro(
        "TIGA is local-first archive intelligence: Hunt indexes and answers, "
        "Atlas builds project memory, Einstein (Phase 2) reasons like a senior architect."
    )

    section_label("What you operate today")
    layer_grid()

    section_label("Active schedule")
    if _sched:
        metric_row([
            ("Mode", str(_sched.get("mode", "—"))),
            ("Extract workers", str(_sched.get("extract_workers", "—"))),
            ("Embed batch", str(_sched.get("embed_batch_size", "—"))),
        ])
        st.caption(_sched.get("description") or f"Source: {_sched.get('source', '—')}")
        if not _sched.get("run_indexing"):
            st.info("Day mode: heavy indexing paused so the search portal stays responsive. Switch to night in Schedule to run full indexing.")
    else:
        empty_state("Schedule API unavailable — is the server running?")

    section_label("Quick actions")
    qa1, qa2, qa3 = st.columns(3)
    with qa1:
        pipeline_trigger("Run full Hunt pipeline", "/api/pipeline/full", "Triggered: Run Full Pipeline", key="ov_full")
    with qa2:
        if st.button("Force night mode", use_container_width=True, key="ov_night"):
            api("post", "/api/schedule/mode", json={"mode": "night"})
            st.toast("Night mode set")
            st.rerun()
    with qa3:
        if st.button("Open Atlas cards", use_container_width=True, key="ov_atlas"):
            st.info("Use the Atlas tab to review project cards.")


# ── TAB: Hunt (pipeline) ───────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Hunt pipeline")
    tab_intro(
        "Discover → extract → embed/index. Extract uses the Schedule worker count. "
        "OCR is Atlas Tier C — planned, not enabled."
    )

    section_label("Run stages")
    run_cols = st.columns(3)
    for idx, (label, path, audit) in enumerate([
        ("Discover", "/api/pipeline/discover", "Triggered: Run Discover"),
        ("Extract", "/api/pipeline/extract", "Triggered: Run Extract"),
        ("Index / embed", "/api/pipeline/index", "Triggered: Run Index"),
    ]):
        with run_cols[idx]:
            pipeline_trigger(label, path, audit, key=f"hunt_run_{idx}")

    full_cols = st.columns(2)
    with full_cols[0]:
        pipeline_trigger("Run full pipeline", "/api/pipeline/full", "Triggered: Run Full Pipeline", key="hunt_full")
    with full_cols[1]:
        if st.button("Refresh status", use_container_width=True, key="hunt_refresh"):
            st.rerun()

    st.caption("OCR (selective, on-demand) is on the Atlas roadmap — not available as a Hunt stage yet.")

    section_label("Control")
    ctrl_cols = st.columns(3)
    with ctrl_cols[0]:
        pipeline_trigger("Pause / Resume", "/api/pipeline/pause", "Pipeline: Pause/Resume", key="hunt_pause")
    with ctrl_cols[1]:
        confirm_and_run(
            "Cancel pipeline",
            "confirm_cancel",
            "Stop the currently running pipeline?",
            "/api/pipeline/cancel",
            "Pipeline: Cancel",
            btn_key="hunt_cancel",
        )
    with ctrl_cols[2]:
        if st.button("Reset stuck UI", use_container_width=True, key="hunt_reset"):
            api("post", "/api/pipeline/reset", json={"force": True})
            api("post", "/api/audit/log", json={"action": "Pipeline: Reset stuck state"})
            st.toast("Pipeline state cleared.")
            st.rerun()

    section_label("Danger zone")
    confirm_and_run(
        "Rebuild index",
        "confirm_rebuild",
        "Rebuild re-processes the entire archive. Slow on large NAS volumes.",
        "/api/pipeline/rebuild",
        "Triggered: Rebuild",
        btn_key="hunt_rebuild",
    )

    st.subheader("Live progress")
    ps = api("get", "/api/pipeline/status") or {}
    if ps.get("running"):
        st.markdown('<div class="tiga-live-panel">', unsafe_allow_html=True)
        st.markdown(f'<div class="tiga-live-stage"><strong>Stage:</strong> {ps.get("stage", "—")}</div>', unsafe_allow_html=True)
        p, t = ps.get("processed", 0), ps.get("total", 0)
        if t > 0:
            st.progress(min(p / t, 1.0), text=f"{p:,} / {t:,} files")
        if ps.get("eta"):
            st.caption(f"ETA: {ps['eta']}s  ·  {ps.get('throughput', 0):.1f} files/s")
        for err in ps.get("errors", []):
            st.warning(err)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        empty_state("No pipeline running. Prefer night mode + Run full pipeline for large archives.")

    with st.expander("Live output", expanded=bool(ps.get("running"))):
        lines = ps.get("output") or []
        st.code("\n".join(lines[-50:]) if lines else "No output yet.", language=None)

    with st.expander("Configuration (read-only)", expanded=False):
        st.markdown(
            f"**Work dir:** `{cfg.work_dir}`  \n"
            f"**Index roots:** {[str(r) for r in cfg.index_roots]}  \n"
            f"**Embed model:** `{cfg.embed_model}` · **Chat model:** `{cfg.chat_model}`  \n"
            f"**Config extract_workers:** `{getattr(cfg, 'extract_workers', '—')}` "
            f"(overridden by Schedule mode at runtime)"
        )
        st.caption("Edit `tiga_work/config.yaml` for defaults. Use the Schedule tab to force day/night.")


# ── TAB: Archives ──────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Archives")
    tab_intro("NAS / folder roots Hunt indexes. Disable a root to skip it on the next discover.")

    dirs = api("get", "/api/directories") or []
    if not dirs:
        empty_state("No index roots configured. Add a directory path below.")
    for d in dirs:
        mounted = d.get("mounted")
        enabled = d.get("enabled", True)
        status_txt = "Online" if mounted else "Offline"
        dot_cls = "tiga-dot-ok" if mounted else "tiga-dot-err"
        st.markdown(
            f'<div class="tiga-dir-card">'
            f'<div class="tiga-dir-path"><strong>{d["path"]}</strong></div>'
            f'<div class="tiga-dir-meta">'
            f'<span class="tiga-status-dot {dot_cls}"></span>'
            f'{status_txt} · {"Enabled" if enabled else "Disabled"}</div></div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Discovered", d.get("discovered", 0))
        c2.metric("Extracted", d.get("extracted", 0))
        c3.metric("Indexed", d.get("indexed", 0))
        c4.metric("Failed", d.get("failed", 0))
        ac = st.columns(3)
        with ac[0]:
            lbl = "Disable" if enabled else "Enable"
            if st.button(lbl, key=f"tog_{d['id']}", use_container_width=True):
                api("post", f"/api/directories/toggle/{d['id']}")
                api("post", "/api/audit/log", json={"action": f"Directory {lbl.lower()}d", "detail": d["path"]})
                st.rerun()
        with ac[1]:
            if st.button("Reindex", key=f"rei_{d['id']}", use_container_width=True):
                api("post", "/api/pipeline/reindex-dir", json={"path": d["path"]})
                api("post", "/api/audit/log", json={"action": "Reindex directory", "detail": d["path"]})
                st.toast("Queued.")
        with ac[2]:
            if st.session_state.get(f"confirm_rem_{d['id']}"):
                st.warning("Remove this root from indexing?")
                ry, rn = st.columns(2)
                with ry:
                    if st.button("Confirm remove", key=f"rem_yes_{d['id']}"):
                        api("post", f"/api/directories/remove/{d['id']}")
                        api("post", "/api/audit/log", json={"action": "Removed index root", "detail": d["path"]})
                        st.session_state.pop(f"confirm_rem_{d['id']}", None)
                        st.rerun()
                with rn:
                    if st.button("Cancel", key=f"rem_no_{d['id']}"):
                        st.session_state.pop(f"confirm_rem_{d['id']}", None)
                        st.rerun()
            elif st.button("Remove", key=f"rem_{d['id']}", use_container_width=True):
                st.session_state[f"confirm_rem_{d['id']}"] = True
                st.rerun()
        st.divider()

    section_label("Add root")
    new_path = st.text_input("Path", placeholder="/path/to/archive", label_visibility="collapsed")
    if st.button("Add directory", type="primary"):
        if new_path.strip():
            r = api("post", "/api/directories/add", json={"path": new_path.strip()})
            if r:
                st.success(f"Added: {new_path}")
                st.rerun()
        else:
            st.warning("Enter a path.")


# ── TAB: Schedule (real workers) ───────────────────────────────────────────
with tabs[3]:
    st.subheader("Schedule")
    tab_intro(
        "This is what actually drives parallel extraction and embed batch size — "
        "day (portal-first) vs night (index-first). Replaces the old Auto-Brain sliders."
    )

    sched = api("get", "/api/schedule/status") or {}
    if not sched:
        empty_state("Could not load schedule status.")
    else:
        section_label("Current mode")
        metric_row([
            ("Mode", str(sched.get("mode", "—"))),
            ("Extract workers", str(sched.get("extract_workers", "—"))),
            ("Embed batch size", str(sched.get("embed_batch_size", "—"))),
        ])
        st.markdown(
            f'<div class="tiga-panel" style="margin-top:10px">'
            f'<div class="tiga-metric-label">Heavy indexing</div>'
            f'<div class="tiga-metric-value" style="font-size:1.1rem;margin-top:4px">'
            f'{"On" if sched.get("run_indexing") else "Off"}</div></div>',
            unsafe_allow_html=True,
        )
        st.caption(sched.get("description") or "")
        st.caption(f"Source: `{sched.get('source', '—')}` · mode file: `{sched.get('mode_file', '—')}`")

        section_label("Force mode")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Day mode", use_container_width=True, key="sched_day",
                         help="Low extract workers, smaller embed batches, indexing paused"):
                api("post", "/api/schedule/mode", json={"mode": "day"})
                api("post", "/api/audit/log", json={"action": "Schedule mode → day"})
                st.toast("Day mode set")
                st.rerun()
        with b2:
            if st.button("Night mode", type="primary", use_container_width=True, key="sched_night",
                         help="Max extract workers, large embed batches, indexing on"):
                api("post", "/api/schedule/mode", json={"mode": "night"})
                api("post", "/api/audit/log", json={"action": "Schedule mode → night"})
                st.toast("Night mode set")
                st.rerun()

        st.markdown(
            """
**Defaults** (from `config.yaml` → `scheduler:`):
- **Day** — light background load so LAN search stays snappy  
- **Night** — saturate CPU/GPU for Discover → Extract → Index  

Run `python tiga.py schedule --daemon` on the server host for automatic day/night switching.
"""
        )


# ── TAB: Atlas ─────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Atlas")
    tab_intro(
        "Project cards are the Atlas v1 foundation — identity, typology, stage, and "
        "stakeholders that power structured and cross-project answers. Full Atlas graph is next."
    )

    missing_only = st.toggle("Show incomplete cards only", value=False)
    data = api("get", f"/api/atlas/cards?missing_only={'true' if missing_only else 'false'}") or {}
    cards = data.get("cards") or []

    c1, c2 = st.columns(2)
    c1.markdown(
        f'<div class="tiga-metric-cell"><div class="tiga-metric-value">{data.get("total", len(cards))}</div>'
        f'<div class="tiga-metric-label">Cards</div></div>',
        unsafe_allow_html=True,
    )
    incomplete = sum(1 for c in cards if c.get("missing_fields"))
    c2.markdown(
        f'<div class="tiga-metric-cell"><div class="tiga-metric-value">{incomplete}</div>'
        f'<div class="tiga-metric-label">Incomplete</div></div>',
        unsafe_allow_html=True,
    )

    if not cards:
        empty_state(
            "No project cards yet. Create via CLI: `python tiga.py card <code>` "
            "or seed with `python tiga.py scrape-woha`."
        )
    else:
        section_label("Project cards")
        for card in cards:
            st.markdown(atlas_card_html(card), unsafe_allow_html=True)

    section_label("Coming in Atlas")
    st.markdown(
        "- Cross-project knowledge views  \n"
        "- Convention / category graph  \n"
        "- Selective OCR (Tier C)  \n"
        "- Admin editing of project cards (beyond CLI)"
    )


# ── TAB: Quality ───────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Quality")
    tab_intro("Hunt search feedback, diagnostics, and self-tests.")

    section_label("Feedback")
    summary = api("get", "/api/feedback/summary") or {}
    metric_row([
        ("Helpful", str(summary.get("total_positive", 0))),
        ("Not helpful", str(summary.get("total_negative", 0))),
        ("Helpful ratio", f"{summary.get('positive_ratio', 0):.0%}" if summary.get("positive_ratio") is not None else "—"),
    ])

    qr = api("get", "/api/feedback/queries") or []
    if qr:
        import pandas as pd
        st.dataframe(
            pd.DataFrame(qr)[["query", "results", "positive", "negative", "comments", "flagged"]],
            use_container_width=True,
        )
    else:
        empty_state("No feedback yet from the search portal.")

    zero_results = api("get", "/api/feedback/zero-results") or []
    if zero_results:
        section_label("Zero-result queries")
        for z in zero_results:
            st.write(f"- **{z['query']}** — {z['attempts']} attempts")

    csv_bytes = api("get", "/api/feedback/export")
    if csv_bytes:
        st.download_button(
            "Export feedback CSV", data=csv_bytes,
            file_name="tiga_feedback.csv", mime="text/csv",
        )

    st.divider()
    section_label("Diagnostics")
    if st.button("Run full diagnostic", type="primary", key="qual_diag"):
        api("post", "/api/audit/log", json={"action": "Triggered: Run Full Diagnostic"})
        with st.spinner("Running…"):
            diag = api("post", "/api/diagnostics/run") or {}
        checks = diag.get("checks", [])
        st.write(f"**{diag.get('passed', 0)}/{diag.get('total', 0)} checks passed**")
        for c in checks:
            dot = "tiga-check-ok" if c["ok"] else "tiga-check-fail"
            fix = f'<div style="font-size:12px;color:var(--tiga-ink-60);margin-top:2px">Fix: {c["fix"]}</div>' if not c["ok"] and c.get("fix") else ""
            st.markdown(
                f'<div class="tiga-check-row"><span class="tiga-check-dot {dot}"></span>'
                f'<div><strong>{c["name"]}</strong> — {c.get("detail", "")}{fix}</div></div>',
                unsafe_allow_html=True,
            )

    section_label("Pipeline activity")
    procs = api("get", "/api/processes") or []
    if procs:
        for p in procs:
            st.write(
                f"`PID {p.get('pid')}` · {p.get('role')} · "
                f"{p.get('runtime', 0)}s runtime"
            )
        st.caption("Shows the active Hunt pipeline thread (not OS process manager).")
    else:
        empty_state("No pipeline running.")

    section_label("Search self-test")
    st.caption("10 Tianmu-oriented probes against the live Hunt index.")
    _TIANMU_QUERIES = [
        ("Design", "facade glazing system and curtain wall details"),
        ("Programme", "building area schedule and programme breakdown"),
        ("Client", "client presentation schematic design"),
        ("Structure", "structural engineer consultant report"),
        ("Submission", "planning authority submission drawings"),
        ("Materials", "external cladding material specification"),
        ("Site", "site analysis topography survey"),
        ("Tender", "tender package drawings and specifications"),
        ("Meetings", "project meeting minutes and action items"),
        ("Coordination", "MEP services coordination drawings"),
    ]
    if st.button("Run self-test (10 queries)", type="primary", key="selftest_run"):
        results = []
        prog_bar = st.progress(0, text="Testing queries…")
        for i, (category, query) in enumerate(_TIANMU_QUERIES):
            prog_bar.progress((i + 1) / len(_TIANMU_QUERIES), text=f"Testing: {query[:50]}…")
            r = api("post", "/api/query", json={"query": query, "top_k": 1})
            if r is None:
                results.append({"Category": category, "Query": query, "Top Result": "ERROR", "Score": "—", "✓": "❌"})
            elif not r.get("results"):
                results.append({"Category": category, "Query": query, "Top Result": "(no results)", "Score": "0%", "✓": "❌"})
            else:
                top = r["results"][0]
                score = int((top.get("final_score", 0)) * 100)
                results.append({
                    "Category": category,
                    "Query": query,
                    "Top Result": top.get("title", "?"),
                    "Score": f"{score}%",
                    "✓": "✅" if score >= 35 else "⚠️",
                })
        prog_bar.empty()
        import pandas as pd
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
        passed_n = sum(1 for row in results if row["✓"] == "✅")
        if passed_n >= 8:
            st.success(f"{passed_n}/10 queries returned relevant results")
        elif passed_n >= 5:
            st.warning(f"{passed_n}/10 — some coverage gaps")
        else:
            st.error(f"{passed_n}/10 — index may need attention")


# ── TAB: System ────────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("System")
    tab_intro("Index maintenance, embedding health, config history, and audit log.")

    section_label("File lookup")
    fq = st.text_input("Search by filename or path", placeholder="e.g. facade glazing spec.pdf")
    if fq.strip():
        fr = api("get", f"/api/index/file?q={fq.strip()}") or {}
        hits = fr.get("results", [])
        if not hits:
            empty_state(f"No indexed files matching “{fq.strip()}”.")
        for f in hits:
            with st.expander(f.get("file_name", "—")):
                st.write(f"**Path:** `{f.get('file_path', '')}`")
                st.write(f"**Status:** {f.get('status')}  ·  **Lane:** {f.get('lane')}")
                st.write(f"**Project:** {f.get('project_id', '—')}  ·  **Typology:** {f.get('typology', '—')}")
                fc = st.columns(2)
                with fc[0]:
                    if st.button("Reindex", key=f"rif_{f['file_id']}"):
                        api("post", "/api/index/reindex-file", json={"path": f["file_path"]})
                        st.toast("Queued.")
                with fc[1]:
                    if st.button("Remove", key=f"rmf_{f['file_id']}"):
                        api("post", "/api/index/remove-file", json={"path": f["file_path"]})
                        st.toast("Removed.")

    section_label("Embedding health")
    eh = api("get", "/api/index/embedding-health") or {}
    if eh.get("mismatch"):
        st.error(f"Mismatch — index: {eh.get('index_dim')} dims, current: {eh.get('current_dim')} dims")
        if st.button("Re-embed all"):
            api("post", "/api/pipeline/re-embed")
            st.toast("Started.")
    elif eh.get("status") == "no_table":
        st.info("No vector table yet — run Hunt pipeline first.")
    else:
        st.success(f"{eh.get('index_dim', '?')} dims  ·  {eh.get('row_count', 0):,} vectors")

    section_label("Maintenance")
    oc = st.columns(2)
    with oc[0]:
        if st.button("Deduplicate"):
            r = api("post", "/api/index/deduplicate") or {}
            st.success(f"{r.get('duplicate_groups', 0)} groups, {r.get('total_dupes', 0)} dupes.")
    with oc[1]:
        if st.button("Integrity check"):
            r = api("post", "/api/index/integrity") or {}
            if r.get("ok"):
                st.success("OK")
            else:
                st.warning(f"{r.get('orphan_chunks', 0)} orphan chunks, {r.get('failed_files', 0)} failed files")

    section_label("Config history")
    history = api("get", "/api/config/history") or []
    for entry in history[:10]:
        hc = st.columns([4, 1])
        hc[0].caption(f"`{entry['version_id']}` — {entry['ts']}")
        with hc[1]:
            if st.button("Rollback", key=f"rb_{entry['version_id']}"):
                api("post", "/api/config/rollback", json={"version_id": entry["version_id"]})
                st.success("Rolled back.")
    if not history:
        empty_state("No config version history yet.")

    st.divider()
    section_label("Audit log")
    pg = st.number_input("Page", min_value=1, value=1, step=1)
    ad = api("get", f"/api/audit?page={int(pg)}&limit=50") or {}
    st.caption(f"{ad.get('total', 0):,} total entries")
    for e in ad.get("items", []):
        detail = f" · <em>{e.get('detail')}</em>" if e.get("detail") else ""
        st.markdown(
            f'<div class="tiga-audit-row"><code>{e.get("ts", "")}</code> · '
            f'<strong>{e.get("action", "")}</strong>{detail}</div>',
            unsafe_allow_html=True,
        )
    csv_audit = api("get", "/api/audit/export")
    if csv_audit:
        st.download_button(
            "Export audit CSV", data=csv_audit,
            file_name="tiga_audit.csv", mime="text/csv",
        )
