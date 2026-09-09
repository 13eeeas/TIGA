# TIGA Hunt

Office-LAN archive search for architecture firms.
Plain English → cited evidence first, then a short grounded answer.

**Charter:** [`docs/CONSTITUTION.md`](docs/CONSTITUTION.md)  
**Projects wiki (Atlas in Hunt):** [`docs/ATLAS_IN_HUNT.md`](docs/ATLAS_IN_HUNT.md) · open `/projects`  
**Office setup:** [`docs/OFFICE_QUICKSTART.md`](docs/OFFICE_QUICKSTART.md)  
**Audit log:** [`docs/AUDIT_LOG.md`](docs/AUDIT_LOG.md)

---

## One-click: download → test

### Windows (office PC)

1. Download or clone this folder onto the LAN host
2. Double-click **`START-HERE.bat`**
3. Enter your NAS project folder path(s) when prompted
4. Pick 3–5 projects in the POC test
5. Send the export zip from `tiga_work/poc_test/exports/` back to dev

### macOS / Linux

```bash
bash START-HERE.sh
```

---

## What to double-click

| File | Purpose |
|------|---------|
| **`START-HERE.bat`** | First run: install + configure + POC test |
| `launcher.bat` | Daily use after POC (portal + search) |
| `poc-test.bat` | Re-run retrieval stress test only |
| `setup.bat` | Install only (called automatically) |
| `run.bat` | Start services + background index |
| `uninstall.bat` | Remove shortcuts / optional data |

---

## Project objective

Build a **company-knowledge search layer** on firm hardware that keeps NAS files in place and returns **auditable, cited answers**.

- **Search first, AI second** — retrieve pages/chunks, then synthesize from evidence only
- **LAN-first** — Hunt works without internet; API optional and firm-approved
- **Prove before scale** — Gateway 1: 3–5 projects, 100-Q benchmark, then expand
- **Represent, don't replicate** — metadata/chunks/embeddings only

---

## CLI (optional)

```
python tiga.py configure    # set index_roots
python tiga.py poc-test run   # POC stress test
python tiga.py validate       # fixture benchmark (dev)
python tiga.py index          # incremental index
python tiga.py serve          # API server :7860
python tiga.py ui             # Admin :7861
```

Full reference: run `python tiga.py --help`

---

## Configuration

Edit `tiga_work/config.yaml` (created on first run from [`docs/config.poc.example.yaml`](docs/config.poc.example.yaml)):

- `index_roots` — NAS project folders
- `retrieval.reranker_enabled` — keep `true` for Gateway 1
- `dedupe.enabled` — skip identical file copies

---

## Stack

| Component | Role |
|-----------|------|
| SQLite + FTS5 | BM25 keyword search |
| LanceDB + Ollama embed | Vector search |
| Cross-encoder | Local rerank |
| FastAPI + static UI | LAN portal |
| Streamlit | Admin panel |

---

## Repo layout

```
START-HERE.bat     ← start here (Windows)
setup.bat          install deps + Ollama
poc-test.bat       re-run POC test
launcher.bat       daily portal
tiga.py            CLI
core/              ingest + search pipeline
docs/              constitution, milestones, quickstart
tiga_work/         local index data (gitignored)
```
