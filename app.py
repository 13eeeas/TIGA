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
    page_icon="⚙️",
    layout="wide",
)

# ── Design system: match main UI fonts + coral accents ──────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,300;8..60,400&display=swap');

:root {
  --tiga-coral: #C96442;
  --tiga-coral-dim: rgba(201,100,66,0.12);
  --tiga-ink: #1A1410;
  --tiga-ink-60: rgba(26,20,16,0.52);
  --tiga-surface: rgba(255,255,255,0.72);
  --tiga-border: rgba(26,20,16,0.09);
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
.block-container {padding-top: 1.5rem; max-width: 1100px;}
.stButton > button[kind="primary"] {
    background-color: var(--tiga-coral) !important;
    border-color: var(--tiga-coral) !important;
    color: white !important;
    font-family: 'Source Serif 4', Georgia, serif !important;
    border-radius: 10px !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #a8522f !important;
    border-color: #a8522f !important;
}
.stButton > button {
    font-family: 'Source Serif 4', Georgia, serif !important;
    border-radius: 10px !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    border-bottom-color: var(--tiga-coral) !important;
    color: var(--tiga-coral) !important;
    font-family: 'Source Serif 4', Georgia, serif !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Source Serif 4', Georgia, serif !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'Lora', Georgia, serif !important;
    font-size: 1.6rem !important;
}
.stProgress > div > div { background-color: var(--tiga-coral) !important; }
.stTextInput input, .stTextArea textarea, .stNumberInput input {
    font-family: 'Source Serif 4', Georgia, serif !important;
    border-radius: 10px !important;
}
section[data-testid="stSidebar"] {
    background: rgba(252,250,247,0.96) !important;
    border-right: 1px solid var(--tiga-border);
}
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
    padding: 6px 14px;
    border-radius: 100px;
    border: 1px solid var(--tiga-border);
    font-size: 13px;
    background: var(--tiga-surface);
}
.tiga-status-dot { width: 8px; height: 8px; border-radius: 50%; }
.tiga-dot-ok { background: #34a853; }
.tiga-dot-warn { background: #fbbc04; }
.tiga-dot-err { background: #ea4335; }
.tiga-empty {
    text-align: center;
    padding: 32px 20px;
    color: var(--tiga-ink-60);
    border: 1px dashed var(--tiga-border);
    border-radius: 14px;
    font-size: 14px;
}
.tiga-login-wrap {
    max-width: 380px;
    margin: 4rem auto 0;
    padding: 32px 28px;
    background: var(--tiga-surface);
    border: 1px solid var(--tiga-border);
    border-radius: 18px;
}
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
    margin: -8px 0 18px 0;
    line-height: 1.55;
}
.main div[data-testid="stMetric"] {
    background: var(--tiga-surface);
    border: 1px solid var(--tiga-border);
    border-radius: 12px;
    padding: 12px 14px;
}
.tiga-dir-card {
    background: var(--tiga-surface);
    border: 1px solid var(--tiga-border);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 10px;
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
        --tiga-surface: rgba(30,26,22,0.85);
        --tiga-border: rgba(237,232,224,0.10);
    }
    .stApp { background: #0F0E0C !important; color: #EDE8E0 !important; }
    section[data-testid="stSidebar"] {
        background: rgba(22,18,14,0.98) !important;
    }
}
</style>
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
    st.markdown("### TIGA Admin")
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

# ── Product status ─────────────────────────────────────────────────────────
_product = api("get", "/api/product/status") or {}
_status = api("get", "/api/status") or {}
_sched = api("get", "/api/schedule/status") or {}
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

# Roadmap cards
_hunt = _product.get("hunt") or {}
_atlas = _product.get("atlas") or {}
_ein = _product.get("einstein") or {}
r1, r2, r3 = st.columns(3)
with r1:
    st.markdown("**Hunt** · live")
    st.caption(_hunt.get("detail", "Search + indexing"))
    st.metric("Indexed files", f"{(_hunt.get('files_indexed') or _status.get('files_indexed') or 0):,}")
with r2:
    st.markdown("**Atlas** · building")
    st.caption(_atlas.get("detail", "Project memory"))
    st.metric("Project cards", f"{(_atlas.get('project_cards') or 0):,}")
with r3:
    st.markdown("**Einstein** · planned")
    st.caption(_ein.get("detail", "Expert reasoning — Phase 2"))
    st.metric("Status", "Phase 2" if not _ein.get("enabled") else "Enabled")

st.divider()

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
    st.markdown(
        """
| Layer | Status | Admin controls |
|-------|--------|----------------|
| **Hunt** | Live | Pipeline, Archives, Schedule, Quality |
| **Atlas** | Foundation | Project cards (this panel) |
| **Einstein** | Planned | Config flag only — not shipping yet |
"""
    )

    section_label("Active schedule")
    if _sched:
        c1, c2, c3 = st.columns(3)
        c1.metric("Mode", _sched.get("mode", "—"))
        c2.metric("Extract workers", _sched.get("extract_workers", "—"))
        c3.metric("Embed batch", _sched.get("embed_batch_size", "—"))
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
    ctrl_cols = st.columns(2)
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
        st.write(f"**Stage:** {ps.get('stage', '—')}")
        p, t = ps.get("processed", 0), ps.get("total", 0)
        if t > 0:
            st.progress(min(p / t, 1.0), text=f"{p:,} / {t:,} files")
        if ps.get("eta"):
            st.caption(f"ETA: {ps['eta']}s  ·  {ps.get('throughput', 0):.1f} files/s")
        for err in ps.get("errors", []):
            st.warning(err)
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
        status_icon = "🟢" if mounted else "🔴"
        st.markdown(
            f'<div class="tiga-dir-card">'
            f'<div class="tiga-dir-path"><strong>{d["path"]}</strong></div>'
            f'<div style="font-size:12px;color:var(--tiga-ink-60);margin-top:4px">'
            f'{status_icon} {status_txt} · {"Enabled" if enabled else "Disabled"}</div></div>',
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
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mode", sched.get("mode", "—"))
        m2.metric("Extract workers", sched.get("extract_workers", "—"))
        m3.metric("Embed batch size", sched.get("embed_batch_size", "—"))
        m4.metric("Heavy indexing", "On" if sched.get("run_indexing") else "Off")
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
    c1.metric("Cards", data.get("total", len(cards)))
    incomplete = sum(1 for c in cards if c.get("missing_fields"))
    c2.metric("Incomplete", incomplete)

    if not cards:
        empty_state(
            "No project cards yet. Create via CLI: `python tiga.py card <code>` "
            "or seed with `python tiga.py scrape-woha`."
        )
    else:
        section_label("Project cards")
        for card in cards:
            code = card.get("project_code", "?")
            name = card.get("name") or "—"
            miss = card.get("missing_fields") or []
            pct = card.get("completeness", 0)
            title = f"{code} · {name} · {pct}% complete"
            with st.expander(title, expanded=False):
                st.write(
                    f"**Typology:** {card.get('typology_primary') or '—'}  ·  "
                    f"**Location:** {card.get('location') or '—'}  ·  "
                    f"**Stage:** {card.get('stage') or '—'}  ·  "
                    f"**Client:** {card.get('client') or '—'}"
                )
                if miss:
                    st.warning("Missing / low-confidence: " + ", ".join(miss))
                else:
                    st.success("Required fields filled.")
                st.caption("Edit via CLI: `python tiga.py card " + str(code) + "`")

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
    f1, f2, f3 = st.columns(3)
    f1.metric("Helpful", summary.get("total_positive", 0))
    f2.metric("Not helpful", summary.get("total_negative", 0))
    ratio = summary.get("positive_ratio")
    f3.metric("Helpful ratio", f"{ratio:.0%}" if ratio is not None else "—")

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
            icon = "✅" if c["ok"] else "❌"
            st.write(f"{icon} **{c['name']}** — {c.get('detail', '')}")
            if not c["ok"] and c.get("fix"):
                st.caption(f"Fix: {c['fix']}")

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
        detail = f" · _{e.get('detail')}_" if e.get("detail") else ""
        st.markdown(f"`{e.get('ts', '')}` · **{e.get('action', '')}**{detail}")
    csv_audit = api("get", "/api/audit/export")
    if csv_audit:
        st.download_button(
            "Export audit CSV", data=csv_audit,
            file_name="tiga_audit.csv", mime="text/csv",
        )
