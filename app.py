"""
app.py — TIGA Hunt Admin Panel (Streamlit).

Admin-only. Accessed from the settings drawer in the main UI.
Password: admin / admin

Run via: python tiga.py ui
"""

from __future__ import annotations

import time
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

_API = f"http://localhost:{cfg.server_port}"


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
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { max-width: 400px; margin: auto; padding-top: 80px; }
    </style>
    """, unsafe_allow_html=True)

    st.title("Admin")
    st.caption("TIGA Hunt administration panel")
    st.divider()

    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")

    if st.button("Login", type="primary", use_container_width=True):
        if username == "admin" and password == "admin":
            st.session_state.admin_authed = True
            st.rerun()
        else:
            st.error("Invalid credentials.")

    st.stop()

# ---------------------------------------------------------------------------
# Admin panel (only reached when authenticated)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Admin")
    st.caption(f"API: `{_API}`")
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
    if st.button("Logout"):
        st.session_state.admin_authed = False
        st.rerun()

tabs = st.tabs([
    "Pipeline",
    "Directories",
    "Workers & Auto-Brain",
    "Index",
    "Diagnostics",
    "Feedback",
    "Audit Log",
])

# ── TAB 1: Pipeline ────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Pipeline Controls")

    btn_cols = st.columns(4)
    actions = [
        ("Run Discover",      "/api/pipeline/discover",   "Triggered: Run Discover"),
        ("Run Extract",       "/api/pipeline/extract",    "Triggered: Run Extract"),
        ("Run OCR",           "/api/pipeline/ocr",        "Triggered: Run OCR"),
        ("Run Index",         "/api/pipeline/index",      "Triggered: Run Index"),
        ("Run Full Pipeline", "/api/pipeline/full",       "Triggered: Run Full Pipeline"),
        ("Rebuild",           "/api/pipeline/rebuild",    "Triggered: Rebuild"),
        ("Pause / Resume",    "/api/pipeline/pause",      "Pipeline: Pause/Resume"),
        ("Cancel",            "/api/pipeline/cancel",     "Pipeline: Cancel"),
    ]
    for idx, (label, path, audit_action) in enumerate(actions):
        with btn_cols[idx % 4]:
            if st.button(label, use_container_width=True, key=f"pipe_{idx}"):
                r = api("post", path)
                if r:
                    api("post", "/api/audit/log", json={"action": audit_action})
                    st.toast(f"{label}: {r.get('status','ok')}")

    st.subheader("Live Progress")
    ps = api("get", "/api/pipeline/status") or {}
    if ps.get("running"):
        st.write(f"**Stage:** {ps.get('stage', '—')}")
        p, t = ps.get("processed", 0), ps.get("total", 0)
        if t > 0:
            st.progress(p / t, text=f"{p}/{t}")
        if ps.get("eta"):
            st.caption(f"ETA: {ps['eta']}s  |  {ps.get('throughput', 0):.1f} files/s")
        for err in ps.get("errors", []):
            st.warning(err)
    else:
        st.write("No pipeline running.")

    with st.expander("Live output", expanded=False):
        lines = ps.get("output") or []
        st.text("\n".join(lines[-50:]) if lines else "No output yet.")

    st.subheader("Config (read-only)")
    st.info(
        f"**Work dir:** `{cfg.work_dir}`\n\n"
        f"**Index roots:** {[str(r) for r in cfg.index_roots]}\n\n"
        f"**Embed model:** `{cfg.embed_model}`  |  **Chat model:** `{cfg.chat_model}`\n\n"
        f"**Embed batch size:** `{cfg.embed_batch_size}`"
    )
    st.caption("Edit `config.yaml` directly to change settings.")


# ── TAB 2: Directories ─────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Index Roots")

    dirs = api("get", "/api/directories") or []
    for d in dirs:
        icon = "✅" if d.get("mounted") else "❌"
        with st.expander(f"{d['path']}  —  {icon}", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Discovered", d.get("discovered", 0))
            c2.metric("Extracted",  d.get("extracted",  0))
            c3.metric("Indexed",    d.get("indexed",    0))
            c4.metric("Failed",     d.get("failed",     0))
            ac = st.columns(3)
            with ac[0]:
                lbl = "Disable" if d.get("enabled") else "Enable"
                if st.button(lbl, key=f"tog_{d['id']}"):
                    api("post", f"/api/directories/toggle/{d['id']}")
                    api("post", "/api/audit/log", json={"action": f"Directory {lbl.lower()}d", "detail": d["path"]})
                    st.rerun()
            with ac[1]:
                if st.button("Reindex", key=f"rei_{d['id']}"):
                    api("post", "/api/pipeline/reindex-dir", json={"path": d["path"]})
                    api("post", "/api/audit/log", json={"action": "Reindex directory", "detail": d["path"]})
                    st.toast("Queued.")
            with ac[2]:
                if st.button("Remove", key=f"rem_{d['id']}", type="secondary"):
                    api("post", f"/api/directories/remove/{d['id']}")
                    api("post", "/api/audit/log", json={"action": "Removed index root", "detail": d["path"]})
                    st.rerun()

    st.divider()
    new_path = st.text_input("Add root", placeholder="/path/to/archive")
    if st.button("Add"):
        if new_path.strip():
            r = api("post", "/api/directories/add", json={"path": new_path.strip()})
            if r:
                st.success(f"Added: {new_path}")
                st.rerun()
        else:
            st.warning("Enter a path.")


# ── TAB 3: Workers & Auto-Brain ────────────────────────────────────────────
with tabs[2]:
    ab = api("get", "/api/autobrain/status") or {}
    enabled = ab.get("enabled", False)

    st.subheader("Auto-Brain")
    tog = st.toggle("Auto-Brain enabled", value=enabled)
    if tog != enabled:
        api("post", "/api/autobrain/toggle", json={"enabled": tog})
        api("post", "/api/audit/log", json={"action": f"Auto-brain {'enabled' if tog else 'disabled'}"})
        st.rerun()

    st.subheader("Worker Allocation")
    allocs = ab.get("allocations", {})
    limits = ab.get("limits", {"min": 1, "max": 16})
    vals: dict[str, int] = {}
    for stage in ["discover", "extract", "ocr", "embed", "index"]:
        vals[stage] = st.slider(f"{stage.capitalize()}", limits["min"], limits["max"],
                                allocs.get(stage, 2), key=f"sl_{stage}", disabled=tog)

    if not tog and st.button("Apply overrides"):
        for stage, w in vals.items():
            if w != allocs.get(stage, 2):
                api("post", "/api/autobrain/override", json={"stage": stage, "workers": w})
        api("post", "/api/audit/log", json={"action": "Set worker overrides", "detail": str(vals)})
        st.toast("Applied.")

    st.subheader("Hard Limits")
    lc = st.columns(2)
    with lc[0]:
        mn = st.number_input("Min workers", value=limits["min"], min_value=1, max_value=8)
    with lc[1]:
        mx = st.number_input("Max workers", value=limits["max"], min_value=1, max_value=64)
    if st.button("Save limits"):
        api("post", "/api/autobrain/limits", json={"min": int(mn), "max": int(mx)})
        api("post", "/api/audit/log", json={"action": f"Set worker limits min={mn} max={mx}"})
        st.toast("Saved.")

    st.subheader("Decision Log")
    for e in reversed(ab.get("decision_log", [])[-20:]):
        st.caption(f"{e.get('ts','')}  {e.get('message','')}")


# ── TAB 4: Index ───────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("File Search")
    fq = st.text_input("Search by filename or path")
    if fq.strip():
        fr = api("get", f"/api/index/file?q={fq.strip()}") or {}
        for f in fr.get("results", []):
            with st.expander(f.get("file_name", "—")):
                st.write(f"**Path:** `{f.get('file_path','')}`")
                st.write(f"**Status:** {f.get('status')}  |  **Lane:** {f.get('lane')}")
                st.write(f"**Project:** {f.get('project_id','—')}  |  **Typology:** {f.get('typology','—')}")
                st.write(f"**Chunks:** {f.get('chunks',0)}")
                fb = f.get("feedback", {})
                st.write(f"**Feedback:** 👍 {fb.get('positive',0)}  👎 {fb.get('negative',0)}")
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

    st.subheader("Embedding Health")
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

    st.subheader("Operations")
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

    st.subheader("Config Version History")
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
        st.write("No config history yet.")


# ── TAB 5: Diagnostics ─────────────────────────────────────────────────────
with tabs[4]:
    if st.button("Run Full Diagnostic"):
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

    st.subheader("Processes")
    procs = api("get", "/api/processes") or []
    if procs:
        for p in procs:
            pc = st.columns([1, 2, 1, 1, 1])
            pc[0].write(str(p.get("pid", "?")))
            pc[1].write(p.get("role", "?"))
            pc[2].write("🟢 active")
            pc[3].write(f"{p.get('cpu',0):.0f}%")
            with pc[4]:
                if st.button("Kill", key=f"kill_{p['pid']}"):
                    api("post", "/api/processes/kill", json={"pid": p["pid"]})
                    api("post", "/api/audit/log",
                        json={"action": f"Killed process PID {p['pid']}"})
    else:
        st.write("No active workers.")


# ── TAB 6: Feedback ────────────────────────────────────────────────────────
with tabs[5]:
    summary = api("get", "/api/feedback/summary") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("👍", summary.get("total_positive", 0))
    c2.metric("👎", summary.get("total_negative", 0))
    r = summary.get("positive_ratio")
    c3.metric("Ratio", f"{r:.0%}" if r is not None else "—")

    st.subheader("Per-Query")
    qr = api("get", "/api/feedback/queries") or []
    if qr:
        import pandas as pd
        st.dataframe(pd.DataFrame(qr)[["query","results","positive","negative","comments","flagged"]],
                     use_container_width=True)
    else:
        st.write("No feedback yet.")

    st.subheader("Zero-Result Queries")
    for z in (api("get", "/api/feedback/zero-results") or []):
        st.write(f"- **{z['query']}** — {z['attempts']} attempts")

    if st.button("Export feedback CSV"):
        csv_bytes = api("get", "/api/feedback/export")
        if csv_bytes:
            st.download_button("Download", data=csv_bytes, file_name="tiga_feedback.csv", mime="text/csv")


# ── TAB 7: Audit Log ───────────────────────────────────────────────────────
with tabs[6]:
    st.caption("Immutable record of every admin action.")
    pg = st.number_input("Page", min_value=1, value=1, step=1)
    ad = api("get", f"/api/audit?page={int(pg)}&limit=50") or {}
    st.caption(f"Total: {ad.get('total',0):,}")
    for e in ad.get("items", []):
        st.write(
            f"`{e.get('ts','')}` — **{e.get('actor','')}** — {e.get('action','')}"
            + (f"  · _{e.get('detail','')}_" if e.get("detail") else "")
        )
    if st.button("Export audit CSV"):
        csv_bytes = api("get", "/api/audit/export")
        if csv_bytes:
            st.download_button("Download", data=csv_bytes, file_name="tiga_audit.csv", mime="text/csv")
