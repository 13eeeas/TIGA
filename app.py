"""
app.py — TIGA Hunt Streamlit UI (Phase 5 / Chunk B).

Three pages via sidebar radio:
  🔍 Search   — hybrid RAG search with feedback
  📊 Status   — pipeline counts + Ollama health
  ⚙️  Admin   — password-gated, 7 tabs

Connects to FastAPI server at http://localhost:{server_port}.
Does NOT call core/ modules directly.
"""

from __future__ import annotations

import time
from typing import Any

import requests
import streamlit as st

from config import cfg

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TIGA",
    page_icon="🔍",
    layout="wide",
)

_API = f"http://localhost:{cfg.server_port}"
_TYPOLOGIES = list(cfg.typology_keyword_map.keys()) if cfg.typology_keyword_map else [
    "healthcare", "education", "sports", "residential", "commercial", "cultural",
]
_FILE_TYPES = ["pdf", "docx", "pptx", "dwg", "xlsx", "jpg", "png"]


# ---------------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------------

def api(method: str, path: str, **kwargs) -> Any:
    """Call FastAPI; raises on HTTP error; returns parsed JSON or None."""
    url = _API + path
    try:
        resp = getattr(requests, method)(url, timeout=kwargs.pop("timeout", 30), **kwargs)
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        if "json" in ct:
            return resp.json()
        return resp.content   # binary (CSV downloads etc.)
    except requests.RequestException as e:
        st.error(f"API error: {e}")
        return None


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

defaults = {
    "session_id":     None,
    "search_query":   "",
    "search_results": None,
    "search_offset":  0,
    "admin_authed":   False,
    "feedback_open":  {},   # {result_id: bool} — comment box open
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("TIGA")
    page = st.radio(
        "Navigate",
        ["🔍 Search", "📊 Status", "⚙️ Admin"],
        label_visibility="collapsed",
    )
    st.divider()

    # Server health badge
    try:
        h = requests.get(_API + "/health", timeout=3).json()
        if h.get("status") == "ok":
            st.success("Server online")
        else:
            st.warning("Server degraded")
    except Exception:
        st.error("Server offline")


# ===========================================================================
# PAGE: SEARCH
# ===========================================================================

if page == "🔍 Search":

    # --- Ensure session ---
    if st.session_state.session_id is None:
        resp = api("post", "/api/session")
        if resp:
            st.session_state.session_id = resp.get("session_id")

    # --- Sidebar filters ---
    with st.sidebar:
        st.subheader("Filters")
        projects_raw = api("get", "/api/projects") or []
        project_options = [p["project_id"] for p in projects_raw]
        sel_projects  = st.multiselect("Project", project_options)
        sel_typology  = st.multiselect("Typology", _TYPOLOGIES)
        sel_filetypes = st.multiselect("File type", _FILE_TYPES)

    # --- Header ---
    st.title("TIGA")
    st.caption("Search your archive, the way you remember it.")

    # --- Query input ---
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query_input = st.text_input(
            "Search",
            value=st.session_state.search_query,
            label_visibility="collapsed",
            placeholder="What are you looking for?",
            key="query_text_input",
        )
    with col_btn:
        submitted = st.button("Search", use_container_width=True, type="primary")

    # Trigger on Enter (text_input change) or button click
    query = query_input.strip()
    if (submitted or (query and query != getattr(st.session_state, "_last_query", ""))) and query:
        st.session_state._last_query = query
        st.session_state.search_query  = query
        st.session_state.search_offset = 0

        filters: dict = {}
        if sel_projects:
            filters["project_id"] = sel_projects[0]   # server accepts one for now
        if sel_typology:
            filters["typology"] = sel_typology[0]
        if sel_filetypes:
            filters["ext"] = sel_filetypes[0]

        with st.spinner("Searching archive…"):
            result = api("post", "/api/query", json={
                "query":      query,
                "top_k":      5,
                "offset":     0,
                "session_id": st.session_state.session_id,
                "filters":    filters or None,
            })
        if result:
            st.session_state.search_results = result

    # --- Display results ---
    res = st.session_state.search_results
    if res:
        # Answer summary
        if res.get("answer_summary"):
            st.info(res["answer_summary"])

        results: list[dict] = res.get("results", [])
        offset = st.session_state.search_offset

        for i, r in enumerate(results):
            result_id = r.get("citation", r.get("rel_path", str(i)))
            ext = r.get("ext", "").lstrip(".")
            project = r.get("project_id", "—")
            typology = r.get("typology", "—")

            with st.expander(
                f"**{r.get('title', r.get('rel_path', '—'))}**  ·  {project}  ·  {typology}",
                expanded=(i == 0),
            ):
                # Snippet
                if r.get("snippet"):
                    st.markdown(r["snippet"])
                # Citation path
                st.code(r.get("citation", r.get("rel_path", "")))
                # File type badge
                if ext:
                    st.caption(f"`.{ext}`")

                # --- Feedback row ---
                fb_cols = st.columns([1, 1, 1, 8])
                with fb_cols[0]:
                    if st.button("👍", key=f"up_{result_id}_{i}"):
                        api("post", "/api/feedback", json={
                            "query":         res.get("query", ""),
                            "result_id":     result_id,
                            "rating":        1,
                            "rank_position": offset + i + 1,
                            "session_id":    st.session_state.session_id,
                        })
                        api("post", "/api/audit/log", json={"action": "Feedback 👍", "detail": result_id})
                        st.toast("Thanks for the feedback!")
                with fb_cols[1]:
                    if st.button("👎", key=f"dn_{result_id}_{i}"):
                        api("post", "/api/feedback", json={
                            "query":         res.get("query", ""),
                            "result_id":     result_id,
                            "rating":        -1,
                            "rank_position": offset + i + 1,
                            "session_id":    st.session_state.session_id,
                        })
                        api("post", "/api/audit/log", json={"action": "Feedback 👎", "detail": result_id})
                        st.toast("Thanks for the feedback!")
                with fb_cols[2]:
                    open_key = f"comment_open_{result_id}_{i}"
                    if st.button("💬", key=f"cmtbtn_{result_id}_{i}"):
                        st.session_state[open_key] = not st.session_state.get(open_key, False)

                if st.session_state.get(open_key, False):
                    comment = st.text_area("Comment", key=f"cmt_{result_id}_{i}", label_visibility="collapsed")
                    if st.button("Send", key=f"cmt_send_{result_id}_{i}"):
                        api("post", "/api/feedback", json={
                            "query":         res.get("query", ""),
                            "result_id":     result_id,
                            "rank_position": offset + i + 1,
                            "session_id":    st.session_state.session_id,
                            "comment":       comment,
                        })
                        st.session_state[open_key] = False
                        st.toast("Comment saved!")

        # --- Show more ---
        if len(results) == 5:
            if st.button("Show more results"):
                st.session_state.search_offset += 5
                next_offset = st.session_state.search_offset
                filters2: dict = {}
                if sel_projects:
                    filters2["project_id"] = sel_projects[0]
                if sel_typology:
                    filters2["typology"] = sel_typology[0]
                if sel_filetypes:
                    filters2["ext"] = sel_filetypes[0]
                with st.spinner("Loading more…"):
                    more = api("post", "/api/query", json={
                        "query":      st.session_state.search_query,
                        "top_k":      5,
                        "offset":     next_offset,
                        "session_id": st.session_state.session_id,
                        "filters":    filters2 or None,
                    })
                if more:
                    st.session_state.search_results = more
                st.rerun()

        # --- Follow-up prompts ---
        follow_ups = res.get("follow_up_prompts", [])
        if follow_ups:
            st.divider()
            st.caption("Follow-up suggestions:")
            cols = st.columns(min(len(follow_ups), 3))
            for j, prompt in enumerate(follow_ups[:3]):
                with cols[j % 3]:
                    if st.button(prompt, key=f"followup_{j}"):
                        st.session_state.search_query  = prompt
                        st.session_state.search_offset = 0
                        with st.spinner("Searching archive…"):
                            result2 = api("post", "/api/query", json={
                                "query":      prompt,
                                "top_k":      5,
                                "offset":     0,
                                "session_id": st.session_state.session_id,
                            })
                        if result2:
                            st.session_state.search_results = result2
                        st.rerun()


# ===========================================================================
# PAGE: STATUS
# ===========================================================================

elif page == "📊 Status":
    st.title("Index Status")

    status = api("get", "/api/status")
    if not status:
        st.error("Could not fetch status from server.")
        st.stop()

    # --- Pipeline counts ---
    st.subheader("Pipeline")
    stages = [
        ("Discovered", status.get("files_discovered", 0)),
        ("Extracted",  status.get("files_extracted",  0)),
        ("Embedded",   status.get("files_embedded",   0)),
        ("Indexed",    status.get("files_indexed",    0)),
        ("Failed",     status.get("files_failed",     0)),
        ("Skipped",    status.get("files_skipped",    0)),
    ]
    cols = st.columns(len(stages))
    for col, (label, count) in zip(cols, stages):
        col.metric(label, f"{count:,}")

    # --- Breakdown by file type ---
    by_ext = status.get("by_extension", {})
    if by_ext:
        st.subheader("By file type")
        ext_rows = []
        for ext, counts in sorted(by_ext.items()):
            ext_rows.append({
                "Extension": ext or "(none)",
                "Discovered": counts.get("DISCOVERED", 0),
                "Extracted":  counts.get("EXTRACTED",  0),
                "Indexed":    counts.get("INDEXED",    0),
                "Failed":     counts.get("FAILED",     0),
            })
        st.dataframe(ext_rows, use_container_width=True)

    # --- Breakdown by directory ---
    by_dir = status.get("by_directory", {})
    if by_dir:
        st.subheader("By directory")
        dir_rows = []
        for label, counts in by_dir.items():
            dir_rows.append({
                "Directory":  label,
                "Root":       counts.get("root", ""),
                "Discovered": counts.get("DISCOVERED", 0),
                "Extracted":  counts.get("EXTRACTED",  0),
                "Indexed":    counts.get("INDEXED",    0),
                "Failed":     counts.get("FAILED",     0),
            })
        st.dataframe(dir_rows, use_container_width=True)

    # --- Ollama status ---
    st.subheader("Ollama")
    oll = status.get("ollama", {})
    col1, col2 = st.columns(2)
    with col1:
        embed_ok = oll.get("embed", {}).get("ok", False)
        label = oll.get("embed", {}).get("model", cfg.embed_model)
        st.metric("Embed model", label, delta="✅ online" if embed_ok else "❌ offline")
    with col2:
        chat_ok = oll.get("chat", {}).get("ok", False)
        label2 = oll.get("chat", {}).get("model", cfg.chat_model)
        st.metric("Chat model", label2, delta="✅ online" if chat_ok else "❌ offline")

    # --- Last indexed ---
    st.subheader("Last indexed")
    st.write(status.get("last_indexed_at") or "Never")

    # --- Disk ---
    st.subheader("Storage")
    col_a, col_b = st.columns(2)
    col_a.metric("Used", f"{status.get('disk_used_gb', 0):.1f} GB")
    col_b.metric("Free", f"{status.get('disk_free_gb', 0):.1f} GB")


# ===========================================================================
# PAGE: ADMIN
# ===========================================================================

elif page == "⚙️ Admin":

    # --- Authentication gate ---
    if not st.session_state.admin_authed:
        st.title("Admin Login")
        with st.form("admin_login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Login")
        if login_btn:
            if username == "admin" and password == "admin":
                st.session_state.admin_authed = True
                st.rerun()
            else:
                st.error("Invalid credentials.")
        st.stop()

    st.title("Admin")

    tabs = st.tabs([
        "Pipeline",
        "Directories",
        "Workers & Auto-Brain",
        "Index",
        "Diagnostics",
        "Feedback",
        "Audit Log",
    ])

    # -----------------------------------------------------------------------
    # TAB 1: Pipeline
    # -----------------------------------------------------------------------
    with tabs[0]:
        st.subheader("Pipeline Controls")

        btn_cols = st.columns(4)
        actions = [
            ("Run Discover",      "POST", "/api/pipeline/discover",    "Triggered: Run Discover"),
            ("Run Extract",       "POST", "/api/pipeline/extract",     "Triggered: Run Extract"),
            ("Run OCR",           "POST", "/api/pipeline/ocr",         "Triggered: Run OCR"),
            ("Run Index",         "POST", "/api/pipeline/index",       "Triggered: Run Index"),
            ("Run Full Pipeline", "POST", "/api/pipeline/full",        "Triggered: Run Full Pipeline"),
            ("Rebuild",           "POST", "/api/pipeline/rebuild",     "Triggered: Rebuild"),
            ("Pause / Resume",    "POST", "/api/pipeline/pause",       "Pipeline: Pause/Resume"),
            ("Cancel",            "POST", "/api/pipeline/cancel",      "Pipeline: Cancel"),
        ]
        for idx, (label, method, path, audit_action) in enumerate(actions):
            with btn_cols[idx % 4]:
                if st.button(label, use_container_width=True, key=f"pipe_btn_{idx}"):
                    r = api(method.lower(), path)
                    if r:
                        st.toast(f"{label}: {r.get('status','ok')}")

        # --- Live progress ---
        st.subheader("Live Progress")
        ps_placeholder = st.empty()
        ps = api("get", "/api/pipeline/status") or {}
        with ps_placeholder.container():
            if ps.get("running"):
                st.write(f"**Stage:** {ps.get('stage', '—')}")
                processed = ps.get("processed", 0)
                total = ps.get("total", 0)
                if total > 0:
                    st.progress(processed / total, text=f"{processed}/{total}")
                if ps.get("eta"):
                    st.caption(f"ETA: {ps['eta']}s  |  {ps.get('throughput', 0):.1f} files/s")
                for err in ps.get("errors", []):
                    st.warning(err)
            else:
                st.write("No pipeline running.")

        # --- Live log ---
        with st.expander("Live output", expanded=False):
            log_lines = (ps.get("output") or [])
            st.text("\n".join(log_lines[-50:]) if log_lines else "No output yet.")

        # --- Config display (read-only) ---
        st.subheader("Config (read-only)")
        st.info(
            f"**Work directory:** `{cfg.work_dir}`\n\n"
            f"**Index roots:** {[str(r) for r in cfg.index_roots]}\n\n"
            f"**Embed model:** `{cfg.embed_model}`  |  **Chat model:** `{cfg.chat_model}`\n\n"
            f"**Chunk size:** `{getattr(cfg, 'chunk_size', '—')}`  |  "
            f"**Embed batch:** `{cfg.embed_batch_size}`"
        )
        st.caption("Edit `config.yaml` directly to change settings.")

    # -----------------------------------------------------------------------
    # TAB 2: Directories
    # -----------------------------------------------------------------------
    with tabs[1]:
        st.subheader("Index Roots")

        dirs = api("get", "/api/directories") or []
        for d in dirs:
            mounted_icon = "✅" if d.get("mounted") else "❌"
            status_label = f"{mounted_icon} {'mounted' if d.get('mounted') else 'not found'}"
            with st.expander(f"{d['path']}  —  {status_label}", expanded=False):
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Discovered", d.get("discovered", 0))
                col_b.metric("Extracted",  d.get("extracted",  0))
                col_c.metric("Indexed",    d.get("indexed",    0))
                col_d.metric("Failed",     d.get("failed",     0))

                a_cols = st.columns(3)
                with a_cols[0]:
                    toggle_label = "Disable" if d.get("enabled") else "Enable"
                    if st.button(toggle_label, key=f"toggle_{d['id']}"):
                        api("post", f"/api/directories/toggle/{d['id']}")
                        api("post", "/api/audit/log",
                            json={"action": f"Directory {toggle_label.lower()}d", "detail": d["path"]})
                        st.rerun()
                with a_cols[1]:
                    if st.button("Reindex", key=f"reindex_{d['id']}"):
                        api("post", "/api/pipeline/reindex-dir", json={"path": d["path"]})
                        api("post", "/api/audit/log",
                            json={"action": "Reindex directory", "detail": d["path"]})
                        st.toast("Reindex queued.")
                with a_cols[2]:
                    if st.button("Remove", key=f"remove_dir_{d['id']}", type="secondary"):
                        api("post", f"/api/directories/remove/{d['id']}")
                        api("post", "/api/audit/log",
                            json={"action": "Removed index root", "detail": d["path"]})
                        st.rerun()

        st.divider()
        st.subheader("Add Root")
        new_path = st.text_input("Directory path", placeholder="/Volumes/Archive/Projects")
        if st.button("Add", key="add_dir_btn"):
            if new_path.strip():
                result_add = api("post", "/api/directories/add", json={"path": new_path.strip()})
                if result_add:
                    st.success(f"Added: {new_path}")
                    st.rerun()
            else:
                st.warning("Enter a directory path.")

    # -----------------------------------------------------------------------
    # TAB 3: Workers & Auto-Brain
    # -----------------------------------------------------------------------
    with tabs[2]:
        ab = api("get", "/api/autobrain/status") or {}

        st.subheader("Auto-Brain")
        enabled = ab.get("enabled", False)
        toggle_ab = st.toggle("Auto-Brain enabled", value=enabled, key="ab_toggle")
        if toggle_ab != enabled:
            api("post", "/api/autobrain/toggle", json={"enabled": toggle_ab})
            api("post", "/api/audit/log",
                json={"action": f"Auto-brain {'enabled' if toggle_ab else 'disabled'}"})
            st.rerun()

        st.caption(
            "When enabled, auto-brain continuously adjusts worker allocation based on "
            "CPU usage, GPU VRAM, queue depth, and task urgency."
        )

        # Worker sliders
        st.subheader("Worker Allocation")
        allocs = ab.get("allocations", {})
        limits = ab.get("limits", {"min": 1, "max": 16})
        stages_list = ["discover", "extract", "ocr", "embed", "index"]
        override_needed = []
        slider_vals: dict[str, int] = {}
        for stage in stages_list:
            current = allocs.get(stage, 2)
            val = st.slider(
                f"{stage.capitalize()} workers",
                min_value=limits["min"],
                max_value=limits["max"],
                value=current,
                key=f"slider_{stage}",
                disabled=toggle_ab,
            )
            slider_vals[stage] = val
            if val != current:
                override_needed.append((stage, val))

        if not toggle_ab and st.button("Apply overrides"):
            for stage, workers in override_needed:
                api("post", "/api/autobrain/override", json={"stage": stage, "workers": workers})
            api("post", "/api/audit/log",
                json={"action": "Set worker overrides", "detail": str(slider_vals)})
            st.toast("Overrides applied.")
            st.rerun()

        # Hard limits
        st.subheader("Hard Limits")
        lim_cols = st.columns(2)
        with lim_cols[0]:
            min_w = st.number_input("Min workers (any stage)", value=limits["min"], min_value=1, max_value=8)
        with lim_cols[1]:
            max_w = st.number_input("Max total workers", value=limits["max"], min_value=1, max_value=64)
        if st.button("Save limits"):
            api("post", "/api/autobrain/limits", json={"min": int(min_w), "max": int(max_w)})
            api("post", "/api/audit/log", json={"action": f"Set worker limits: min={min_w} max={max_w}"})
            st.toast("Limits saved.")

        # Decision log
        st.subheader("Auto-Brain Decision Log")
        decision_log = ab.get("decision_log", [])
        if decision_log:
            for entry in reversed(decision_log[-20:]):
                st.caption(f"{entry.get('ts', '')}  {entry.get('message', '')}")
        else:
            st.write("No decisions yet.")

        # Escalations
        escalations = ab.get("escalations", [])
        if escalations:
            st.subheader("Escalations")
            for esc in escalations:
                st.warning(
                    f"⚠ **UNRESOLVED** — {esc.get('worker', '?')}\n"
                    f"Issue: {esc.get('issue', '?')}\n"
                    f"Auto-brain gave up at {esc.get('ts', '?')}"
                )

    # -----------------------------------------------------------------------
    # TAB 4: Index
    # -----------------------------------------------------------------------
    with tabs[3]:
        # File search
        st.subheader("File-Level Search")
        file_query = st.text_input("Search by filename or path", key="idx_file_q")
        if file_query.strip():
            file_res = api("get", f"/api/index/file?q={file_query.strip()}") or {}
            for fr in file_res.get("results", []):
                with st.expander(fr.get("file_name", "—"), expanded=True):
                    st.write(f"**Path:** `{fr.get('file_path', '')}`")
                    st.write(f"**Status:** {fr.get('status')}  |  **Lane:** {fr.get('lane')}")
                    st.write(f"**Project:** {fr.get('project_id','—')}  |  **Typology:** {fr.get('typology','—')}")
                    st.write(f"**Chunks:** {fr.get('chunks', 0)}")
                    fb_info = fr.get("feedback", {})
                    st.write(f"**Feedback:** 👍 {fb_info.get('positive',0)}  👎 {fb_info.get('negative',0)}")
                    f_cols = st.columns(2)
                    with f_cols[0]:
                        if st.button("Reindex this file", key=f"reindex_file_{fr['file_id']}"):
                            api("post", "/api/index/reindex-file", json={"path": fr["file_path"]})
                            api("post", "/api/audit/log",
                                json={"action": "Reindex file", "detail": fr["file_path"]})
                            st.toast("Queued for reindex.")
                    with f_cols[1]:
                        if st.button("Remove from index", key=f"remove_file_{fr['file_id']}"):
                            api("post", "/api/index/remove-file", json={"path": fr["file_path"]})
                            api("post", "/api/audit/log",
                                json={"action": "Remove file from index", "detail": fr["file_path"]})
                            st.toast("File removed.")

        # Embedding health
        st.subheader("Embedding Health")
        eh = api("get", "/api/index/embedding-health") or {}
        if eh.get("mismatch"):
            st.error(
                f"⚠ Mismatch detected — index: {eh.get('index_dim')} dims, "
                f"current model: {eh.get('current_dim')} dims"
            )
            if st.button("Re-embed All"):
                api("post", "/api/pipeline/re-embed")
                api("post", "/api/audit/log", json={"action": "Triggered: Re-embed All"})
                st.toast("Re-embed started.")
        elif eh.get("status") == "no_table":
            st.info("No vector table yet — run the pipeline first.")
        else:
            st.success(
                f"✅ Dimension match ({eh.get('index_dim','?')} dims)  |  "
                f"{eh.get('row_count', 0):,} vectors in index"
            )

        # Index operations
        st.subheader("Index Operations")
        op_cols = st.columns(2)
        with op_cols[0]:
            if st.button("Deduplicate (by hash)"):
                r = api("post", "/api/index/deduplicate") or {}
                api("post", "/api/audit/log", json={"action": "Triggered: Deduplicate"})
                st.success(f"Found {r.get('duplicate_groups',0)} groups, {r.get('total_dupes',0)} dupes.")
        with op_cols[1]:
            if st.button("Integrity Check"):
                r = api("post", "/api/index/integrity") or {}
                api("post", "/api/audit/log", json={"action": "Triggered: Integrity Check"})
                if r.get("ok"):
                    st.success("✅ Index integrity OK")
                else:
                    st.warning(
                        f"⚠ {r.get('orphan_chunks',0)} orphan chunks, "
                        f"{r.get('failed_files',0)} failed files"
                    )

        # Config version history
        st.subheader("Config Version History")
        history = api("get", "/api/config/history") or []
        if history:
            for entry in history[:10]:
                h_cols = st.columns([4, 1])
                with h_cols[0]:
                    st.caption(f"`{entry['version_id']}` — {entry['ts']}")
                with h_cols[1]:
                    if st.button("Rollback", key=f"rollback_{entry['version_id']}"):
                        api("post", "/api/config/rollback",
                            json={"version_id": entry["version_id"]})
                        api("post", "/api/audit/log",
                            json={"action": "Config rollback",
                                  "detail": f"→ version {entry['version_id']}"})
                        st.success("Rolled back.")
        else:
            st.write("No config history yet.")

    # -----------------------------------------------------------------------
    # TAB 5: Diagnostics
    # -----------------------------------------------------------------------
    with tabs[4]:
        st.subheader("Diagnostics")

        if st.button("Run Full Diagnostic"):
            api("post", "/api/audit/log", json={"action": "Triggered: Run Full Diagnostic"})
            with st.spinner("Running checks…"):
                diag = api("post", "/api/diagnostics/run") or {}
            checks = diag.get("checks", [])
            passed = diag.get("passed", 0)
            total  = diag.get("total", 0)
            st.write(f"**{passed}/{total} checks passed**")
            for c in checks:
                icon = "✅" if c["ok"] else ("⚠" if "low" in c.get("detail","").lower() else "❌")
                with st.container():
                    cols = st.columns([1, 6])
                    cols[0].write(icon)
                    cols[1].write(f"**{c['name']}** — {c.get('detail','')}")
                    if not c["ok"] and c.get("fix"):
                        cols[1].caption(f"Fix: {c['fix']}")

            if st.button("Export diagnostic report"):
                report_lines = [f"{c['name']}: {'PASS' if c['ok'] else 'FAIL'} — {c.get('detail','')}"
                                for c in checks]
                report_text = "\n".join(report_lines)
                st.download_button("Download report.txt", data=report_text,
                                   file_name="tiga_diagnostic.txt", mime="text/plain")

        # Live process table
        st.subheader("Live Processes")
        procs = api("get", "/api/processes") or []
        if procs:
            for p in procs:
                stalled = p.get("status") == "stalled"
                p_cols = st.columns([1, 2, 1, 1, 1, 1, 1])
                p_cols[0].write(str(p.get("pid", "?")))
                p_cols[1].write(p.get("role", "?"))
                status_str = f"🔴 {p['status']}" if stalled else f"🟢 {p['status']}"
                p_cols[2].write(status_str)
                p_cols[3].write(f"{p.get('cpu', 0):.0f}%")
                p_cols[4].write(f"{p.get('ram_gb', 0):.1f} GB")
                p_cols[5].write(f"{p.get('runtime', 0)}s")
                with p_cols[6]:
                    if st.button("Kill", key=f"kill_{p['pid']}"):
                        api("post", "/api/processes/kill", json={"pid": p["pid"]})
                        api("post", "/api/audit/log",
                            json={"action": f"Killed process: {p.get('role')} (PID {p['pid']})"})
        else:
            st.write("No active workers.")

    # -----------------------------------------------------------------------
    # TAB 6: Feedback
    # -----------------------------------------------------------------------
    with tabs[5]:
        st.subheader("Feedback Summary")

        summary = api("get", "/api/feedback/summary") or {}
        col1, col2, col3 = st.columns(3)
        col1.metric("👍 Positive", summary.get("total_positive", 0))
        col2.metric("👎 Negative", summary.get("total_negative", 0))
        ratio = summary.get("positive_ratio")
        col3.metric("Positive ratio", f"{ratio:.0%}" if ratio is not None else "—")

        trending = summary.get("trending_down", [])
        if trending:
            st.subheader("Trending down-rated queries")
            for t in trending:
                st.write(f"- **{t['query']}** — {t['count']} negative ratings")

        # Per-query table
        st.subheader("Per-Query Feedback")
        q_rows = api("get", "/api/feedback/queries") or []
        if q_rows:
            import pandas as pd
            df = pd.DataFrame(q_rows)
            def _flag(row):
                return ["background-color: #ffe0e0"] * len(row) if row.get("flagged") else [""] * len(row)
            st.dataframe(df[["query","results","positive","negative","comments","flagged"]],
                         use_container_width=True)
        else:
            st.write("No feedback recorded yet.")

        # Zero-result queries
        st.subheader("Zero-Result / Low-Confidence Queries")
        zero = api("get", "/api/feedback/zero-results") or []
        if zero:
            for z in zero:
                st.write(f"- **{z['query']}** — {z['attempts']} attempts, last: {z.get('last_seen','?')}")
        else:
            st.write("None found.")

        # Export
        st.divider()
        if st.button("Export all feedback as CSV"):
            csv_bytes = api("get", "/api/feedback/export")
            if csv_bytes:
                st.download_button("Download feedback.csv",
                                   data=csv_bytes,
                                   file_name="tiga_feedback.csv",
                                   mime="text/csv")

    # -----------------------------------------------------------------------
    # TAB 7: Audit Log
    # -----------------------------------------------------------------------
    with tabs[6]:
        st.subheader("Audit Log")
        st.caption("Immutable record of every human-initiated admin action.")

        audit_page = st.number_input("Page", min_value=1, value=1, step=1, key="audit_page")
        audit_data = api("get", f"/api/audit?page={int(audit_page)}&limit=50") or {}
        total_audit = audit_data.get("total", 0)
        items = audit_data.get("items", [])

        st.caption(f"Total entries: {total_audit:,}")
        for entry in items:
            st.write(
                f"`{entry.get('ts','')}` — **{entry.get('actor','')}** — "
                f"{entry.get('action','')}  "
                + (f"· _{entry.get('detail','')}_" if entry.get("detail") else "")
            )

        st.divider()
        if st.button("Export audit log as CSV"):
            csv_bytes = api("get", "/api/audit/export")
            if csv_bytes:
                st.download_button("Download audit.csv",
                                   data=csv_bytes,
                                   file_name="tiga_audit.csv",
                                   mime="text/csv")
