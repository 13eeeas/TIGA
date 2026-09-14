# Office quick start — Gateway 1 POC

Download the repo (zip or git clone), then **one double-click** on Windows:

```
START-HERE.bat
```

That runs:

1. **Install** — Python venv, dependencies, Ollama models (`setup.bat`, first time only)
2. **Configure** — paste your NAS project folder path(s)
3. **POC test** — pick 3–5 projects → index → stress retrieval → export zip

## Download

**Option A — Git**

```bat
git clone https://github.com/13eeeas/TIGA.git
cd TIGA
START-HERE.bat
```

**Option B — ZIP**

Download the branch zip from GitHub, extract, open the folder, double-click `START-HERE.bat`.

## After the test

| File | When |
|------|------|
| `launcher.bat` | Daily Hunt search portal |
| `update.bat` / `update.sh` | Later GitHub deploys (fast-forward only — see below) |
| `poc-test.bat` | Re-run stress test (after config tweaks) |
| `run.bat` | Start server + admin + incremental index |
| `tiga_work/poc_test/exports/` | Send zip to dev for Hunt refinement |

## Linux / macOS

```bash
bash START-HERE.sh
```

## Update an existing git clone (do not brick tiga_work)

`START-HERE` is first-run only — it does not pull GitHub. On a git clone:

```bat
update.bat
```

```bash
bash update.sh
```

The updater fetches, then **fast-forwards only**. It will **stop** (and leave the install alone) if the branch has diverged, tracked files are dirty, or untracked source would be overwritten. It never `git reset`, stash, merge, or `git clean`.

`tiga_work/config.yaml`, the Hunt DB, vectors, and Atlas overlays (`tiga_work/atlas/`) stay put.

Pin a SHA: `bash update.sh --sha <commit>` or `set TIGA_UPDATE_SHA=<commit>` then `update.bat`.

Health checks use `server.port` from `tiga_work/config.yaml`. They do **not** assume `7860`.

Rollback and failure states: [`docs/WOHA_UPDATE.md`](WOHA_UPDATE.md).

## Optional env vars

Copy `.env.example` → `.env` or set in shell:

```
TIGA_ADMIN_PASSWORD=change-me
TIGA_ADMIN_USER=admin
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Python not found | Install Python 3.10+ with “Add to PATH” |
| Ollama pull failed | Run `ollama pull nomic-embed-text` manually |
| No projects listed in POC test | Check `index_roots` paths exist — run `python tiga.py configure` |
| Test slow | Normal for first index — run overnight; use 3–5 projects only |

See also: [`docs/AUDIT_LOG.md`](AUDIT_LOG.md), [`docs/config.poc.example.yaml`](config.poc.example.yaml)
