"""
app.py — TIGA Hunt Admin Panel (Streamlit).

Admin-only. Accessed from the settings drawer in the main UI.
Password: admin / admin

Run via: python tiga.py ui
Requires: python tiga.py serve (FastAPI on cfg.server_port)
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests
import streamlit as st

from config import cfg

st.set_page_config(
    page_title="TIGA Admin",
    page_icon="⚙️",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,300;8..60,400&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, .stCaption,
.stDataFrame, div[data-testid="stMetricLabel"], div[data-testid="stMetricValue"] {
    font-family: 'Source Serif 4', Georgia, serif !important;
}
h1, h2, h3, h4, .stSubheader, div[data-testid="stHeading"] {
    font-family: 'Lora', Georgia, serif !important;
    font-weight: 400 !important;
}
.stButton > button[kind="primary"] {
    background-color: #C96442 !important;
    border-color: #C96442 !important;
    color: white !important;
}
.stProgress > div > div { background-color: #C96442 !important; }
</style>
""", unsafe_allow_html=True)

_API = os.environ.get(
    "TIGA_API_URL",
    f"http://127.0.0.1:{cfg.server_port}",
).rstrip("/")


def api(method: str, path: str, **kwargs) -> Any:
    url = _API + path
    try:
        resp = getattr(requests, method)(url, timeout=kwargs.pop("timeout", 120), **kwargs)
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        if "json" in ct:
            return resp.json()
        return resp.content
    except requests.RequestException as e:
        st.error(f"API error ({path}): {e}")
        return None


if "admin_authed" not in st.session_state:
    st.session_state.admin_authed = False

if not st.session_state.admin_authed:
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


@st.fragment(run_every=5)
def _live_ops_panel() -> None:
    """Auto-refresh pipeline + validate status."""
    ps = api("get", "/api/pipeline/status", timeout=10) or {}
    vs = ps.get("validate") or api("get", "/api/validate/status", timeout=10) or {}
    status = api("get", "/api/status", timeout=10) or {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Indexed files", status.get("files_indexed", 0))
    c2.metric("Chunks", status.get("chunks_total", 0))
    c3.metric("Pipeline", ps.get("stage") or ("running" if ps.get("running") else "idle"))
    c4.metric("Validate", "running" if vs.get("running") else (
        "PASS" if vs.get("gateway_pass") else ("fail" if vs.get("exit_code") else "idle")
    ))

    if ps.get("running"):
        st.info(f"Pipeline **{ps.get('stage', '—')}** running…")
        if ps.get("total", 0) > 0:
            st.progress(ps["processed"] / ps["total"], text=f"{ps['processed']}/{ps['total']}")
    if vs.get("running"):
        st.info("Validate run in progress (fixture index + search benchmark, no API)…")

    with st.expander("Live log", expanded=bool(ps.get("running") or vs.get("running"))):
        lines = (ps.get("output") or []) + (vs.get("output") or [])
        st.text("\n".join(lines[-40:]) if lines else "No activity.")


with st.sidebar:
    st.title("⚙️ Admin")
    st.caption(f"API: `{_API}`")
    try:
        h = requests.get(_API + "/health", timeout=3).json()
        if h.get("status") == "ok":
            st.success("Server OK")
        else:
            st.warning("Server degraded")
        if h.get("ollama"):
            st.caption("Ollama: online")
        else:
            st.caption("Ollama: offline (OK if using API compose)")
        if h.get("compose_api_enabled"):
            st.caption(
                f"Einstein: {h.get('compose_provider')} "
                f"({'ready' if h.get('compose_api_ready') else 'no key'})"
            )
    except Exception:
        st.error("Server offline — run `python tiga.py serve`")
    st.divider()
    if st.button("Logout"):
        st.session_state.admin_authed = False
        st.rerun()

st.subheader("System status")
_live_ops_panel()

tabs = st.tabs([
    "Pipeline",
    "Validate & Benchmark",
    "POC Test",
    "Directories",
    "Workers & Auto-Brain",
    "Index",
    "Diagnostics",
    "Feedback",
    "Field Data",
    "Audit Log",
])

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
                    st.toast(f"{label}: {r.get('status', 'ok')}")
                    st.rerun()

    st.subheader("Config snapshot")
    st.info(
        f"**Work dir:** `{cfg.work_dir}`\n\n"
        f"**Index roots:** {[str(r) for r in cfg.index_roots]}\n\n"
        f"**Reranker:** {'on' if cfg.reranker_enabled else 'off'}  |  "
        f"**Evidence pack:** {cfg.compose_evidence_pack_size} chunks\n\n"
        f"**Compose:** {cfg.compose_provider} "
        f"({'API on' if cfg.compose_api_enabled else 'API off'})  |  "
        f"**Model:** `{cfg.compose_model}`\n\n"
        f"**Embed:** `{cfg.embed_model}`  |  **Fallback chat:** `{cfg.chat_model}`"
    )

with tabs[1]:
    st.subheader("Pre-flight validate (fixture archive)")
    st.caption(
        "Tests full index logic on `tests/fixtures/mini_archive` — **no NAS, no API**. "
        "Same as `python tiga.py validate`."
    )
    vc1, vc2, vc3 = st.columns(3)
    with vc1:
        mock_embed = st.checkbox("Mock embed (no Ollama)", value=True)
    with vc2:
        if st.button("Run validate now", type="primary"):
            r = api("post", "/api/validate", json={"mock_embed": mock_embed})
            if r:
                st.toast(r.get("status", "started"))
                st.rerun()
    with vc3:
        if st.button("Refresh status"):
            st.rerun()

    vs = api("get", "/api/validate/status") or {}
    if vs.get("running"):
        st.warning("Validate running…")
    elif vs.get("finished_at"):
        gate = vs.get("gateway_pass")
        if gate:
            st.success(f"Last validate **PASS** (exit {vs.get('exit_code')})")
        else:
            st.error(f"Last validate **FAIL** (exit {vs.get('exit_code')})")
        sr = vs.get("search") or {}
        if sr:
            if sr.get("mode") == "search_only_dual":
                st.write(
                    f"Literal **{sr.get('literal_recall_pct', '—')}%**  |  "
                    f"Paraphrase **{sr.get('paraphrase_recall_pct', '—')}%**  |  "
                    f"Citation **{sr.get('citation_valid_pct', '—')}%**  |  "
                    f"p50 **{sr.get('latency_p50_ms', '—')} ms**"
                )
            else:
                st.write(
                    f"Top-5 recall **{sr.get('top5_recall_pct', '—')}%**  |  "
                    f"Citation valid **{sr.get('citation_valid_pct', '—')}%**  |  "
                    f"Latency p50 **{sr.get('latency_p50_ms', '—')} ms**"
                )
        if vs.get("report_path"):
            st.caption(f"Report: `{vs['report_path']}`")

    st.subheader("Validate history")
    reports = api("get", "/api/validate/reports?limit=10") or {}
    items = reports.get("items") or []
    if items:
        import pandas as pd
        df = pd.DataFrame([
            {
                "When": i.get("ts"),
                "Literal %": i.get("literal_recall_pct", i.get("top5_recall_pct")),
                "Paraphrase %": i.get("paraphrase_recall_pct", "—"),
                "Pass": "✅" if i.get("gateway_pass") else "❌",
                "Files": i.get("files_indexed"),
                "Mock embed": i.get("mock_embed"),
                "Report": i.get("name"),
            }
            for i in items
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.write("No validate reports yet. Run validate above.")

    st.divider()
    st.subheader("Search recall on your index (Hunt only — no API)")
    st.caption("Uses `tiga_work/fixtures/eval_queries.yaml` against the **current** indexed corpus.")
    ek = st.number_input("Top K", min_value=1, max_value=20, value=5, key="eval_topk")
    if st.button("Run search recall eval"):
        with st.spinner("Running…"):
            result = api("post", f"/api/eval/search-recall?top_k={int(ek)}", timeout=180)
        if result and result.get("ok"):
            st.write(f"**Recall:** {result.get('top5_recall_pct')}% ({result.get('total_queries')} queries)")
            import pandas as pd
            st.dataframe(pd.DataFrame(result.get("queries", [])), use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("One-click POC test")
    st.caption(
        "Choose projects → index → auto-generate architecture-firm queries from your corpus → "
        "stress Hunt retrieval → export zip with refinement playbook. **No API cost.**"
    )
    st.info("CLI: double-click `poc-test.bat` or run `python tiga.py poc-test run`")

    ps = api("get", "/api/poc-test/status") or {}
    if ps.get("running"):
        st.warning("POC test running… check Live log in System status.")
    elif ps.get("finished_at") and ps.get("stress"):
        sr = ps["stress"]
        st.write(
            f"Last run — Literal **{sr.get('literal_recall_pct')}%** | "
            f"Paraphrase **{sr.get('paraphrase_recall_pct')}%** | "
            f"Overall **{sr.get('overall_recall_pct')}%**"
        )
        if ps.get("export_path"):
            st.caption(f"Export: `{ps['export_path']}`")

    proj = api("get", "/api/poc-test/projects") or {}
    items = proj.get("items") or []
    if items:
        labels = {
            f"{p.get('name')} ({p.get('indexed_files', p.get('indexable_files', 0))} files)": p.get("path")
            for p in items
        }
        picked = st.multiselect(
            "Projects to index & test (pick 3–5)",
            options=list(labels.keys()),
            key="poc_projects",
        )
        skip_idx = st.checkbox("Skip index — stress test only", value=False)
        if st.button("Run POC test", type="primary"):
            paths = [labels[k] for k in picked if k in labels]
            if len(paths) < 1:
                st.warning("Select at least one project.")
            else:
                r = api("post", "/api/poc-test/run", json={
                    "project_paths": paths,
                    "skip_index": skip_idx,
                    "top_k": 5,
                })
                if r:
                    st.toast(r.get("status", "started"))
                    st.rerun()
    else:
        st.write("No projects found — set `index_roots` in config.yaml.")

    exports = api("get", "/api/poc-test/exports") or {}
    for item in (exports.get("items") or [])[:5]:
        name = item.get("name", "")
        st.markdown(f"- [{name}]({_API}/api/poc-test/exports/{name})")

# ── Directories (unchanged logic) ───────────────────────────────────────────
with tabs[3]:
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
                    st.rerun()
            with ac[1]:
                if st.button("Reindex", key=f"rei_{d['id']}"):
                    api("post", "/api/pipeline/reindex-dir", json={"path": d["path"]})
                    st.toast("Queued.")
            with ac[2]:
                if st.button("Remove", key=f"rem_{d['id']}"):
                    api("post", f"/api/directories/remove/{d['id']}")
                    st.rerun()
    st.divider()
    new_path = st.text_input("Add root", placeholder="/path/to/archive")
    if st.button("Add"):
        if new_path.strip():
            api("post", "/api/directories/add", json={"path": new_path.strip()})
            st.rerun()

with tabs[4]:
    ab = api("get", "/api/autobrain/status") or {}
    enabled = ab.get("enabled", False)
    st.subheader("Auto-Brain")
    tog = st.toggle("Auto-Brain enabled", value=enabled)
    if tog != enabled:
        api("post", "/api/autobrain/toggle", json={"enabled": tog})
        st.rerun()
    allocs = ab.get("allocations", {})
    limits = ab.get("limits", {"min": 1, "max": 16})
    for stage in ["discover", "extract", "ocr", "embed", "index"]:
        st.slider(stage.capitalize(), limits["min"], limits["max"],
                  allocs.get(stage, 2), key=f"sl_{stage}", disabled=enabled)

with tabs[5]:
    st.subheader("File Search")
    fq = st.text_input("Search by filename or path")
    if fq.strip():
        fr = api("get", f"/api/index/file?q={fq.strip()}") or {}
        for f in fr.get("results", []):
            with st.expander(f.get("file_name", "—")):
                st.write(f"**Path:** `{f.get('file_path','')}`")
                st.write(f"**Status:** {f.get('status')}  |  **Chunks:** {f.get('chunks',0)}")
                if st.button("Reindex file", key=f"rif_{f['file_id']}"):
                    api("post", "/api/index/reindex-file", json={"path": f["file_path"]})
                    st.toast("Queued.")
    eh = api("get", "/api/index/embedding-health") or {}
    if eh.get("mismatch"):
        st.error("Embedding dimension mismatch")
    elif eh.get("status") == "no_table":
        st.info("No vector table yet.")
    else:
        st.success(f"Vectors OK — {eh.get('row_count',0):,} rows")
    oc = st.columns(2)
    with oc[0]:
        if st.button("Deduplicate"):
            r = api("post", "/api/index/deduplicate") or {}
            st.success(f"{r.get('duplicate_groups',0)} duplicate groups")
    with oc[1]:
        if st.button("Integrity Check"):
            r = api("post", "/api/index/integrity") or {}
            st.success("OK" if r.get("ok") else f"issues found")

with tabs[6]:
    if st.button("Run Full Diagnostic"):
        with st.spinner("Running…"):
            diag = api("post", "/api/diagnostics/run") or {}
        for c in diag.get("checks", []):
            st.write(f"{'✅' if c['ok'] else '❌'} **{c['name']}** — {c.get('detail','')}")

with tabs[7]:
    summary = api("get", "/api/feedback/summary") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("👍", summary.get("total_positive", 0))
    c2.metric("👎", summary.get("total_negative", 0))
    r = summary.get("positive_ratio")
    c3.metric("Ratio", f"{r:.0%}" if r is not None else "—")
    qr = api("get", "/api/feedback/queries") or []
    if qr:
        import pandas as pd
        st.dataframe(pd.DataFrame(qr), use_container_width=True)

with tabs[8]:
    st.subheader("Office field test collector")
    st.caption(
        "Every Hunt search is logged locally (no API keys). Export a zip when done "
        "in the office, then run `python tiga.py collect import` on dev to refine Hunt."
    )
    cs = api("get", "/api/collect/status") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Search events", cs.get("search_events", 0))
    c2.metric("Gold labels", cs.get("labels", 0))
    c3.metric("Feedback", cs.get("feedback_rows", 0))
    c4.metric("Collector", "ON" if cs.get("enabled") else "OFF")

    ex_cols = st.columns(2)
    with ex_cols[0]:
        since = st.text_input("Export since (optional)", placeholder="2026-08-01")
        if st.button("Export field bundle", type="primary"):
            url = "/api/collect/export"
            if since.strip():
                url += f"?since={since.strip()}"
            r = api("post", url)
            if r:
                st.success(f"Exported: `{r.get('name')}`")
                st.caption(f"Path: `{r.get('path')}`")
                dl = _API + (r.get("download_url") or "")
                st.markdown(f"[Download zip]({dl})")
    with ex_cols[1]:
        exports = api("get", "/api/collect/exports") or {}
        for item in (exports.get("items") or [])[:5]:
            name = item.get("name", "")
            size_kb = round((item.get("size_bytes") or 0) / 1024, 1)
            st.markdown(f"- [{name}]({_API}/api/collect/exports/{name}) ({size_kb} KB)")

    st.divider()
    st.subheader("Add gold label")
    st.caption("When staff know the correct file but Hunt missed it — feeds refinement.")
    lq = st.text_input("Query", key="label_query")
    le = st.text_area(
        "Expected path suffixes (one per line)",
        placeholder="261_tianmu/01_Brief/project_brief.txt",
        key="label_paths",
    )
    ln = st.text_input("Notes (optional)", key="label_notes")
    if st.button("Save label"):
        paths = [p.strip() for p in le.splitlines() if p.strip()]
        if lq.strip() and paths:
            r = api("post", "/api/collect/label", json={
                "query": lq.strip(),
                "expected_paths": paths,
                "notes": ln.strip() or None,
            })
            if r:
                st.success("Label saved")
                st.rerun()
        else:
            st.warning("Query and at least one expected path required.")

    labels = api("get", "/api/collect/labels") or {}
    items = labels.get("items") or []
    if items:
        st.subheader("Saved labels")
        import pandas as pd
        st.dataframe(pd.DataFrame([
            {
                "Query": i.get("query"),
                "Expected": ", ".join(i.get("expected_paths") or []),
                "Source": i.get("source"),
            }
            for i in items
        ]), use_container_width=True, hide_index=True)

with tabs[9]:
    st.caption("Immutable record of admin actions.")
    ad = api("get", "/api/audit?page=1&limit=50") or {}
    st.caption(f"Total: {ad.get('total',0):,}")
    for e in ad.get("items", []):
        st.write(
            f"`{e.get('ts','')}` — **{e.get('actor','')}** — {e.get('action','')}"
            + (f"  · _{e.get('detail','')}_" if e.get("detail") else "")
        )
