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
.tiga-metric-row [data-testid="stMetric"] {
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

# ── Dashboard status bar ───────────────────────────────────────────────────
_status = api("get", "/api/status") or {}
_health = {}
try:
    _health = requests.get(_API + "/health", timeout=3).json()
except Exception:
    pass

st.markdown(
    '<div class="tiga-status-bar">'
    + status_pill("Server online" if _status else "Server offline", bool(_status))
    + status_pill("Ollama online" if _health.get("ollama") else "Ollama offline", _health.get("ollama"))
    + status_pill(f"{(_status.get('files_indexed') or 0):,} indexed", True if _status.get('files_indexed') else None)
    + status_pill(f"{(_status.get('files_discovered') or 0):,} discovered", None)
    + '</div>',
    unsafe_allow_html=True,
)

_last = _status.get("last_indexed_at") or "Never"
if _last and _last != "Never":
    _last = str(_last)[:16].replace("T", " ")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Indexed", f"{(_status.get('files_indexed') or 0):,}")
with m2:
    st.metric("Extracted", f"{(_status.get('files_extracted') or 0):,}")
with m3:
    st.metric("Failed", f"{(_status.get('files_failed') or 0):,}")
with m4:
    st.metric("Last indexed", _last)

st.markdown('<div class="tiga-metric-row"></div>', unsafe_allow_html=True)
st.divider()

tabs = st.tabs([
    "Pipeline",
    "Directories",
    "Workers",
    "Index",
    "Diagnostics",
    "Feedback",
    "Audit",
])

# ── TAB 1: Pipeline ────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Pipeline")
    tab_intro("Run indexing stages, monitor live progress, and inspect configuration.")

    section_label("Run stages")
    run_cols = st.columns(4)
    run_actions = [
        ("Discover", "/api/pipeline/discover", "Triggered: Run Discover"),
        ("Extract",  "/api/pipeline/extract",  "Triggered: Run Extract"),
        ("OCR",      "/api/pipeline/ocr",      "Triggered: Run OCR"),
        ("Index",    "/api/pipeline/index",    "Triggered: Run Index"),
    ]
    for idx, (label, path, audit_action) in enumerate(run_actions):
        with run_cols[idx]:
            pipeline_trigger(label, path, audit_action, key=f"run_{idx}")

    full_cols = st.columns(2)
    with full_cols[0]:
        pipeline_trigger("Run full pipeline", "/api/pipeline/full", "Triggered: Run Full Pipeline", key="run_full")
    with full_cols[1]:
        if st.button("Refresh status", use_container_width=True, key="pipe_refresh"):
            st.rerun()

    section_label("Control")
    ctrl_cols = st.columns(2)
    with ctrl_cols[0]:
        pipeline_trigger("Pause / Resume", "/api/pipeline/pause", "Pipeline: Pause/Resume", key="pipe_pause")
    with ctrl_cols[1]:
        confirm_and_run(
            "Cancel pipeline",
            "confirm_cancel",
            "Stop the currently running pipeline?",
            "/api/pipeline/cancel",
            "Pipeline: Cancel",
            btn_key="pipe_cancel",
        )

    section_label("Danger zone")
    confirm_and_run(
        "Rebuild index",
        "confirm_rebuild",
        "Rebuild re-processes the entire archive. This can take a long time on large NAS volumes.",
        "/api/pipeline/rebuild",
        "Triggered: Rebuild",
        btn_key="pipe_rebuild",
    )

    st.subheader("Live progress")
    ps = api("get", "/api/pipeline/status") or {}
    if ps.get("running"):
        st.write(f"**Stage:** {ps.get('stage', '—')}")
        p, t = ps.get("processed", 0), ps.get("total", 0)
        if t > 0:
            st.progress(p / t, text=f"{p:,} / {t:,} files")
        if ps.get("eta"):
            st.caption(f"ETA: {ps['eta']}s  ·  {ps.get('throughput', 0):.1f} files/s")
        for err in ps.get("errors", []):
            st.warning(err)
    else:
        empty_state("No pipeline running. Start with Discover or Run full pipeline above.")

    with st.expander("Live output", expanded=bool(ps.get("running"))):
        lines = ps.get("output") or []
        st.code("\n".join(lines[-50:]) if lines else "No output yet.", language=None)

    with st.expander("Configuration (read-only)", expanded=False):
        st.markdown(
            f"**Work dir:** `{cfg.work_dir}`  \n"
            f"**Index roots:** {[str(r) for r in cfg.index_roots]}  \n"
            f"**Embed model:** `{cfg.embed_model}` · **Chat model:** `{cfg.chat_model}`  \n"
            f"**Embed batch size:** `{cfg.embed_batch_size}`"
        )
        st.caption("Edit `tiga_work/config.yaml` to change settings.")


# ── TAB 2: Directories ─────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Directories")
    tab_intro("Manage which archive folders TIGA indexes. Disabled roots are skipped on the next run.")

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
        c2.metric("Extracted",  d.get("extracted",  0))
        c3.metric("Indexed",    d.get("indexed",    0))
        c4.metric("Failed",     d.get("failed",     0))
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


# ── TAB 3: Workers & Auto-Brain ────────────────────────────────────────────
with tabs[2]:
    st.subheader("Workers")
    tab_intro("Tune parallel workers per stage. Enable Auto-Brain to let TIGA allocate workers automatically.")

    ab = api("get", "/api/autobrain/status") or {}
    enabled = ab.get("enabled", False)

    section_label("Auto-brain")
    tog = st.toggle("Auto-Brain enabled", value=enabled,
                    help="When on, worker sliders are locked and TIGA adjusts allocation.")
    if tog != enabled:
        api("post", "/api/autobrain/toggle", json={"enabled": tog})
        api("post", "/api/audit/log", json={"action": f"Auto-brain {'enabled' if tog else 'disabled'}"})
        st.rerun()

    section_label("Worker allocation")
    if tog:
        st.caption("Auto-Brain is managing workers. Turn it off to set manual overrides.")
    allocs = ab.get("allocations", {})
    limits = ab.get("limits", {"min": 1, "max": 16})
    vals: dict[str, int] = {}
    for stage in ["discover", "extract", "ocr", "embed", "index"]:
        vals[stage] = st.slider(f"{stage.capitalize()}", limits["min"], limits["max"],
                                allocs.get(stage, 2), key=f"sl_{stage}", disabled=tog)

    if not tog and st.button("Apply overrides", type="primary"):
        for stage, w in vals.items():
            if w != allocs.get(stage, 2):
                api("post", "/api/autobrain/override", json={"stage": stage, "workers": w})
        api("post", "/api/audit/log", json={"action": "Set worker overrides", "detail": str(vals)})
        st.toast("Applied.")

    section_label("Hard limits")
    lc = st.columns(2)
    with lc[0]:
        mn = st.number_input("Min workers", value=limits["min"], min_value=1, max_value=8)
    with lc[1]:
        mx = st.number_input("Max workers", value=limits["max"], min_value=1, max_value=64)
    if st.button("Save limits"):
        api("post", "/api/autobrain/limits", json={"min": int(mn), "max": int(mx)})
        api("post", "/api/audit/log", json={"action": f"Set worker limits min={mn} max={mx}"})
        st.toast("Saved.")

    section_label("Decision log")
    log_entries = list(reversed(ab.get("decision_log", [])[-20:]))
    if log_entries:
        for e in log_entries:
            st.caption(f"{e.get('ts','')}  {e.get('message','')}")
    else:
        empty_state("No Auto-Brain decisions logged yet.")


# ── TAB 4: Index ───────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Index")
    tab_intro("Look up indexed files, check embedding health, and run maintenance operations.")

    section_label("File lookup")
    fq = st.text_input("Search by filename or path", placeholder="e.g. facade glazing spec.pdf")
    if fq.strip():
        fr = api("get", f"/api/index/file?q={fq.strip()}") or {}
        hits = fr.get("results", [])
        if not hits:
            empty_state(f"No indexed files matching “{fq.strip()}”.")
        for f in hits:
            with st.expander(f.get("file_name", "—")):
                st.write(f"**Path:** `{f.get('file_path','')}`")
                st.write(f"**Status:** {f.get('status')}  ·  **Lane:** {f.get('lane')}")
                st.write(f"**Project:** {f.get('project_id','—')}  ·  **Typology:** {f.get('typology','—')}")
                st.write(f"**Chunks:** {f.get('chunks',0)}")
                fb = f.get("feedback", {})
                st.write(f"**Feedback:** {fb.get('positive',0)} helpful · {fb.get('negative',0)} not helpful")
                fc = st.columns(2)
                with fc[0]:
                    if st.button("Reindex", key=f"rif_{f['file_id']}"):
                        api("post", "/api/index/reindex-file", json={"path": f["file_path"]})
                        api("post", "/api/audit/log", json={"action": "Reindex file", "detail": f["file_path"]})
                        st.toast("Queued.")
                with fc[1]:
                    if st.button("Remove", key=f"rmf_{f['file_id']}"):
                        api("post", "/api/index/remove-file", json={"path": f["file_path"]})
                        api("post", "/api/audit/log", json={"action": "Remove file", "detail": f["file_path"]})
                        st.toast("Removed.")

    section_label("Embedding health")
    eh = api("get", "/api/index/embedding-health") or {}
    if eh.get("mismatch"):
        st.error(f"⚠ Mismatch — index: {eh.get('index_dim')} dims, current: {eh.get('current_dim')} dims")
        if st.button("Re-embed All"):
            api("post", "/api/pipeline/re-embed")
            api("post", "/api/audit/log", json={"action": "Triggered: Re-embed All"})
            st.toast("Started.")
    elif eh.get("status") == "no_table":
        st.info("No vector table yet.")
    else:
        st.success(f"✅ {eh.get('index_dim','?')} dims  |  {eh.get('row_count',0):,} vectors")

    section_label("Maintenance")
    oc = st.columns(2)
    with oc[0]:
        if st.button("Deduplicate"):
            r = api("post", "/api/index/deduplicate") or {}
            api("post", "/api/audit/log", json={"action": "Triggered: Deduplicate"})
            st.success(f"{r.get('duplicate_groups',0)} groups, {r.get('total_dupes',0)} dupes.")
    with oc[1]:
        if st.button("Integrity Check"):
            r = api("post", "/api/index/integrity") or {}
            api("post", "/api/audit/log", json={"action": "Triggered: Integrity Check"})
            if r.get("ok"):
                st.success("✅ OK")
            else:
                st.warning(f"{r.get('orphan_chunks',0)} orphan chunks, {r.get('failed_files',0)} failed files")

    section_label("Config history")
    history = api("get", "/api/config/history") or []
    for entry in history[:10]:
        hc = st.columns([4, 1])
        hc[0].caption(f"`{entry['version_id']}` — {entry['ts']}")
        with hc[1]:
            if st.button("Rollback", key=f"rb_{entry['version_id']}"):
                api("post", "/api/config/rollback", json={"version_id": entry["version_id"]})
                api("post", "/api/audit/log", json={"action": "Config rollback", "detail": entry["version_id"]})
                st.success("Rolled back.")
    if not history:
        empty_state("No config version history yet.")


# ── TAB 5: Diagnostics ─────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Diagnostics")
    tab_intro("Health checks, process management, and search quality self-tests.")

    section_label("System check")
    if st.button("Run full diagnostic", type="primary"):
        api("post", "/api/audit/log", json={"action": "Triggered: Run Full Diagnostic"})
        with st.spinner("Running…"):
            diag = api("post", "/api/diagnostics/run") or {}
        checks = diag.get("checks", [])
        st.write(f"**{diag.get('passed',0)}/{diag.get('total',0)} checks passed**")
        for c in checks:
            icon = "✅" if c["ok"] else "❌"
            dc = st.columns([1, 8])
            dc[0].write(icon)
            dc[1].write(f"**{c['name']}** — {c.get('detail','')}")
            if not c["ok"] and c.get("fix"):
                dc[1].caption(f"Fix: {c['fix']}")

    section_label("Processes")
    procs = api("get", "/api/processes") or []
    if procs:
        for p in procs:
            pc = st.columns([1, 2, 1, 1, 1])
            pc[0].write(str(p.get("pid", "?")))
            pc[1].write(p.get("role", "?"))
            pc[2].write("🟢 active")
            pc[3].write(f"{p.get('cpu',0):.0f}%")
            with pc[4]:
                if st.session_state.get(f"confirm_kill_{p['pid']}"):
                    if st.button("Confirm kill", key=f"kill_yes_{p['pid']}"):
                        api("post", "/api/processes/kill", json={"pid": p["pid"]})
                        api("post", "/api/audit/log", json={"action": f"Killed process PID {p['pid']}"})
                        st.session_state.pop(f"confirm_kill_{p['pid']}", None)
                        st.rerun()
                elif st.button("Kill", key=f"kill_{p['pid']}"):
                    st.session_state[f"confirm_kill_{p['pid']}"] = True
                    st.rerun()
    else:
        empty_state("No active worker processes.")

    section_label("Search quality")
    st.caption("10 Tianmu-specific queries probing the index from multiple angles.")

    _TIANMU_QUERIES = [
        ("Design",       "facade glazing system and curtain wall details"),
        ("Programme",    "building area schedule and programme breakdown"),
        ("Client",       "client presentation schematic design"),
        ("Structure",    "structural engineer consultant report"),
        ("Submission",   "planning authority submission drawings"),
        ("Materials",    "external cladding material specification"),
        ("Site",         "site analysis topography survey"),
        ("Tender",       "tender package drawings and specifications"),
        ("Meetings",     "project meeting minutes and action items"),
        ("Coordination", "MEP services coordination drawings"),
    ]

    if st.button("Run Self-Test (10 queries)", type="primary", key="selftest_run"):
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
                top   = r["results"][0]
                score = int((top.get("final_score", 0)) * 100)
                passed = score >= 35
                results.append({
                    "Category":   category,
                    "Query":      query,
                    "Top Result": top.get("title", "?"),
                    "Score":      f"{score}%",
                    "✓":          "✅" if passed else "⚠️",
                })
        prog_bar.empty()

        import pandas as pd
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

        passed_n = sum(1 for r in results if r["✓"] == "✅")
        if passed_n >= 8:
            st.success(f"✅ {passed_n}/10 queries returned relevant results")
        elif passed_n >= 5:
            st.warning(f"⚠️ {passed_n}/10 queries returned relevant results — some gaps in coverage")
        else:
            st.error(f"❌ {passed_n}/10 queries returned relevant results — index may need attention")


# ── TAB 6: Feedback ────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Feedback")
    tab_intro("Search quality signals from the portal — thumbs, comments, and zero-result queries.")

    summary = api("get", "/api/feedback/summary") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Helpful", summary.get("total_positive", 0))
    c2.metric("Not helpful", summary.get("total_negative", 0))
    r = summary.get("positive_ratio")
    c3.metric("Helpful ratio", f"{r:.0%}" if r is not None else "—")

    section_label("Per query")
    qr = api("get", "/api/feedback/queries") or []
    if qr:
        import pandas as pd
        st.dataframe(pd.DataFrame(qr)[["query","results","positive","negative","comments","flagged"]],
                     use_container_width=True)
    else:
        empty_state("No feedback recorded yet. Thumbs and comments from the search portal appear here.")

    section_label("Zero-result queries")
    zero_results = api("get", "/api/feedback/zero-results") or []
    if zero_results:
        for z in zero_results:
            st.write(f"- **{z['query']}** — {z['attempts']} attempts")
    else:
        empty_state("No zero-result queries logged.")

    csv_bytes = api("get", "/api/feedback/export")
    if csv_bytes:
        st.download_button("Export feedback CSV", data=csv_bytes,
                           file_name="tiga_feedback.csv", mime="text/csv", type="primary")


# ── TAB 7: Audit Log ───────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Audit log")
    tab_intro("Immutable record of every admin action.")

    pg_col, export_col = st.columns([1, 1])
    with pg_col:
        pg = st.number_input("Page", min_value=1, value=1, step=1)
    ad = api("get", f"/api/audit?page={int(pg)}&limit=50") or {}
    st.caption(f"{ad.get('total', 0):,} total entries")
    for e in ad.get("items", []):
        detail = f" · _{e.get('detail')}_" if e.get("detail") else ""
        st.markdown(
            f"`{e.get('ts','')}` · **{e.get('action','')}**{detail}"
        )
    with export_col:
        csv_bytes = api("get", "/api/audit/export")
        if csv_bytes:
            st.download_button("Export audit CSV", data=csv_bytes,
                               file_name="tiga_audit.csv", mime="text/csv")
