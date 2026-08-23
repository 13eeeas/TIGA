# Gateway 1 — Backend Engineering Checklist

**Intended goal:** Shipped LAN POC product on **3–5 indexed projects** — Hunt + thin Atlas + enterprise Einstein — that **works super well** (see [`MILESTONES.md`](MILESTONES.md)).

**How to use this doc:** Work top-to-bottom within each track. Do not start Atlas polish or API spend until **Track A (corpus)** and **Track C (retrieval)** items are checked.

**Legend:** `[x]` exists in repo · `[ ]` needs work · `[~]` partial

---

## Track 0 — POC setup (before indexing)

### Corpus & config
- [ ] Lock **3–5 project codes** and NAS `index_roots` (completed jobs staff can judge)
- [x] POC config template: `docs/config.poc.example.yaml`
- [ ] Set `exclude_globs` for junk: caches, temp, backups, texture libs, autosaves
- [ ] Confirm enterprise API vendor shortlist + env var names (no keys in repo)
- [ ] Document expected index size / file count per project (baseline for “represent don’t replicate”)

### Pre-flight (before real NAS index)
- [x] `python tiga.py validate` — fixture archive, mocked embed, search benchmark
- [x] `pytest tests/test_pipeline_integration.py` — automated same checks
- [ ] `python tiga.py validate --real-embed` — optional with Ollama running
- [ ] FastAPI binds LAN (`server.host` / firewall allows office browsers)
- [ ] Night indexing window configured if embed load competes with daytime queries

---

## Track A — Hunt: ingest & data quality

**Gate:** indexed files are authoritative, not noisy.

### Discovery & fingerprint
- [x] Incremental discover (`core/discover.py`)
- [x] SHA256 fingerprint + metadata skip
- [ ] Run `discover` + `index` on project 1; record files/sec and error count
- [ ] Repeat for all 3–5 projects; fix permission / path errors

### Dedupe & version
- [x] Content hash skip on unchanged chunks (`core/extract.py`)
- [x] Path parser: revision, `is_superseded`, `is_latest` (`core/path_parser.py`)
- [ ] Verify `update_is_latest()` runs after each index batch
- [x] **Default retrieval prefers latest** — soft boost/penalty in `core/query.py`
- [ ] Near-duplicate linking or suppression (same text, multiple paths) — `[ ]` not built

### Extraction tiers
- [x] Text lane: PDF/DOCX/PPTX/TXT/… (`core/extract.py`)
- [x] Metadata-only lane for CAD/BIM/media
- [ ] OCR stays **off** unless a POC project truly needs it
- [ ] Spot-check 10 high-value files per project: text actually extracted

### Index integrity
- [x] SQLite chunks + FTS5 + vector lane (`core/db.py`, `core/vectors.py`)
- [ ] `python tiga.py status` shows expected file/chunk counts per project
- [ ] Orphan / stale vector integrity pass (`/api/index/integrity` or CLI)
- [ ] Embed queue complete (`embedded=1` for searchable chunks)

**Track A done when:** 3–5 projects `INDEXED`, latest flags sane, no obvious junk dominating chunk counts.

---

## Track B — Hunt: retrieval quality

**Gate:** right source in top 5 before spending on API polish.

### Hybrid search
- [x] BM25 + vector merge (`core/query.py`)
- [x] Query router + synonyms (`core/router.py`, `query_synonyms.yml`)
- [x] Structured / file_locator / cross_project executors
- [ ] Tune BM25 OR behavior for multi-project corpus (over-broad recall)
- [ ] Project scoping: auto-filter when project code detected in query

### Reranker (critical)
- [x] Cross-encoder module (`core/reranker.py`)
- [x] **`reranker_enabled: true`** in default init + POC template
- [x] Rerank on **full chunk text** from DB (via `chunk_text` / `reranker_chunk_chars`)
- [x] **`reranker_top_k: 50`** in defaults
- [x] `sentence-transformers` in requirements

### Evidence pool (for search results + Einstein)
- [x] Hybrid pool uses `retrieval_candidate_pool()` (≥50 when rerank on)
- [x] **`evidence_pack_size: 12`** in compose config
- [x] Compose loads full chunk text for evidence pack
- [ ] Return enough ranked results in `/api/query` for UI + eval

### Citations
- [x] Citation build + `validate_citation()` (`core/eval.py`, `core/query.py`)
- [x] Invalid citations excluded from results
- [ ] Page/ref accuracy spot-check on PDFs per project

**Track B done when:** manual smoke test ≥ 8/10 questions hit right file in top 5 on each project.

---

## Track C — Einstein: enterprise API compose

**Gate:** synthesis quality without bulk egress.

### Provider layer (new)
- [x] `core/llm_providers.py` — pluggable compose backend (OpenAI / Azure / Anthropic / Ollama)
- [x] Config block: `compose.provider`, `api_enabled`, `model`, evidence pack sizes
- [x] API key from **env only** (`TIGA_LLM_API_KEY`, `TIGA_OPENAI_API_KEY`, `TIGA_ANTHROPIC_API_KEY`)
- [x] Enterprise endpoint / base URL override (Azure fields in config)
- [ ] Firm enterprise contract + key in your office env

### Evidence-pack contract
- [x] Load chunk **full text** (cap via `chunk_char_cap`) for pack members
- [x] Hard cap: `evidence_pack_size` (default 12)
- [x] System prompt: answer ONLY from context
- [x] Citations still from search `results` only

### Fallback & cost
- [x] Ollama fallback pattern exists (`compose.py`)
- [ ] Fallback when API down / timeout / kill switch: cited excerpts + clear banner
- [ ] Log per query: provider, model, input tokens est., latency, `$` optional
- [ ] Append to `tiga_work/logs/queries.log` or `events` table

### Tests
- [ ] Unit tests with mocked HTTP — no live API in CI
- [ ] Manual 10-query API smoke on one project before full benchmark

**Track C done when:** enterprise API answers grounded in evidence pack; fallback works with API disabled.

---

## Track D — Atlas: thin project memory (Gateway 1 scope)

**Gate:** structured + cross-project queries on the indexed 3–5 only.

### Project cards
- [x] `project_cards` table + builder (`core/project_card.py`)
- [ ] Build/update cards for **every** indexed project code
- [ ] Manual review: typology, stage, GFA, key consultants, aliases
- [ ] `GET /api/project/{code}` returns sensible card in UI

### Cross-project (Atlas-lite)
- [x] `execute_cross_project_query` + router mode
- [ ] Verify typology / scale / waiver-style questions across 3–5 set
- [ ] No graph DB — cards + SQL filters only

### Structured & file locator
- [x] Router modes wired in `server.py`
- [ ] Eval questions for structured + file_locator pass on real project data
- [ ] Folder semantics aligned per project (`folder_semantics.yml`)

**Track D done when:** “Who’s the PA on 261?” and “Show me all hotel projects” work on indexed set without semantic RAG.

---

## Track E — LAN API & query pipeline

### Server
- [x] `POST /api/query` — route → search → compose (`server.py`)
- [x] Session, status, projects, index pipeline endpoints
- [ ] POC config: single work dir, documented LAN URL for testers
- [ ] Query log includes: mode, source chunk ids sent to API, provider
- [ ] `GET /api/health` or extend `status`: embed OK, API configured, index age

### CLI parity
- [x] `python tiga.py query`, `eval`, `index`, `status`
- [ ] `python tiga.py eval --routing` uses `tests/eval_questions.json` paths
- [ ] Document one-command re-index for one project

**Track E done when:** office browser + CLI both hit same backend on LAN IP.

---

## Track F — Eval & Gateway 1 proof

**Gate:** objective “works super well” — not demo vibes.

### Benchmark dataset
- [x] Path-based fixture (`tiga_work/fixtures/eval_queries.yaml`)
- [x] Routing eval set (`tests/eval_questions.json`) — 80+ questions
- [ ] **Build 100-question Gateway set** with:
  - [ ] expected answer text (or field)
  - [ ] expected source file + page/ref
  - [ ] project code + mode tag
- [ ] Stratify across 3–5 projects and modes (semantic / file_locator / structured / cross_project)

### Metrics (automate where possible)
- [x] Top-5 path recall on fixture benchmark (`tiga.py validate`)
- [x] Citation validity gate on fixture benchmark
- [ ] **100-Q Gateway set** on real 3–5 projects with expected answers
- [ ] **Answer correctness** (human or LLM-judge with rubric) → target >85%
- [ ] **Citation supports claim** → target >95%
- [ ] **Hallucination flag** on sample → target <3%
- [ ] Latency p50/p95 → target <10 s
- [ ] Export **`eval_gateway1_<date>.json`** + 1-page memo template

### Human proof
- [ ] 3 staff unprompted trial (log session ids / feedback)
- [ ] Sponsor live demo script (10 questions, mix of modes)

**Track F done when:** benchmark report hits MILESTONES Gateway 1 table + memo ready for funding ask.

---

## Track G — Security & audit (POC-minimum)

Not full SSO — enough for IT conversation.

- [ ] Data-flow one-pager: only evidence-pack excerpts leave LAN
- [ ] `compose.api_enabled` kill switch tested
- [ ] Audit log: query, user/session, chunk ids, model, timestamp (no full chunk bodies in log)
- [ ] Confirm no API keys in client/static JS
- [ ] Path open handler still restricted to `index_roots` — `[x]` server check exists
- [ ] ACL stub: optional `project_id` allowlist in config for POC if needed — `[ ]`

**Track G done when:** IT can read one page and see excerpts-only + enterprise terms + kill switch.

---

## Track H — Dependencies & ops

- [ ] `requirements.txt` includes `sentence-transformers` (or optional extra `[rerank]`)
- [ ] Document GPU/CPU expectations for 3070 host (embed by night, query by day)
- [ ] Backup: `tiga_work/db/` + `tiga_work/vectors/` procedure
- [ ] API spend tracker spreadsheet or log aggregation

---

## Suggested build order (backend)

```
Week 1   Track 0 + Track A (project 1 indexed clean)
Week 2   Track B (rerank + evidence pool + latest default)
Week 3   Track C (enterprise API compose) + Track A (remaining projects)
Week 4   Track D + Track F (cards, 100-Q run, memo)
         Track G in parallel when API goes live
```

---

## Gateway 1 exit checklist (all must pass)

- [ ] **3–5 projects** indexed on LAN host
- [ ] Hunt: rerank on, version-aware default, evidence pack 8–15
- [ ] Einstein: enterprise API with fallback + kill switch
- [ ] Atlas: cards + cross-project on indexed set
- [ ] **100-Q benchmark** ≥ Gateway 1 targets
- [ ] Total POC cost ≤ S$1,000 logged
- [ ] 3 staff would use it over NAS hunting
- [ ] Sponsor demo + POC memo delivered

---

## Known gaps in repo today (start here)

| Gap | File / area | Status |
|-----|-------------|--------|
| Enterprise API compose | `core/llm_providers.py`, `compose.py` | **Done** |
| Reranker off by default | was `config.py` | **Fixed in init template** |
| Rerank uses snippet not full text | `reranker.py`, `query.py` | **Fixed** |
| Only 3 snippets to LLM | `compose.py` | **Fixed — evidence pack 12** |
| Eval = path recall only | `core/eval.py` | Still TODO |
| Latest boost query-triggered only | `core/query.py` | **Fixed — soft default** |
| Near-dup suppression missing | — | TODO |
| Index 3–5 POC projects | your NAS + config | **Your next step** |

---

**Next engineering action:** Track 0 (lock 3–5 projects + POC config) → Track A project 1 index → Track B rerank + evidence pool.
