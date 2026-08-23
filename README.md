# TIGA Hunt

Office-LAN archive search for architecture firms.
Plain English → cited evidence first, then a short grounded answer.

**Charter:** [`docs/CONSTITUTION.md`](docs/CONSTITUTION.md) — objectives, LAN constraints, enterprise API security.  
**Milestones:** [`docs/MILESTONES.md`](docs/MILESTONES.md) — phases, funding gates, proof metrics.  
**Backend work:** [`docs/BACKEND_CHECKLIST.md`](docs/BACKEND_CHECKLIST.md) — Gateway 1 engineering checklist.  
Where docs disagree, the constitution wins.

## Quick Start

```
setup.bat        # first time only
run.bat          # start server + UI
```

Then open the UI from any browser **on the office LAN**.

## Project objective

Build a **company-knowledge search layer** that runs on firm hardware on the office LAN, keeps raw NAS files in place, and returns **auditable, cited answers**.

- **Search first, AI second** — retrieve the right pages/chunks, then synthesize only from that evidence.
- **LAN-first** — ingest, hybrid search, local rerank, and citations must work without internet; external APIs are optional and firm-approved only.
- **Prove before scale** — **Gateway 1:** a POC product on **3–5 projects** that works super well (100-Q benchmark); then expand.
- **Lean resources** — two-person build, POC aimed under ~S$1,000; spend on retrieval quality before premium models.

### Design principles
- **Represent, don't replicate**: metadata / chunks / embeddings only — not a second copy of the archive.
- **Evidence-first answers**: factual claims map to files/chunks with validated citations.
- **Version-aware retrieval**: prefer authoritative / latest; duplicates must not outvote the real source.
- **Permissions before retrieval**: never fetch restricted data and rely on the model to hide it.
- **Incremental by default**: changed-only indexing, resumable pipelines, fast warm scans.
- **Degrade gracefully**: if an API or local LLM is down, still return cited search results.

### Answer model posture
Local GPU is for the **search system** (and optional local synthesis). A stronger **approved API** may compose answers over a small evidence pack only — see constitution §7–§8 for Options A–E (LAN-only, evidence-pack API, private VPC endpoint, on-prem appliance, embeddings-only).

### Roadmap language
- **TIGA Hunt** (now): ingestion, indexing, retrieval, cited answering on the LAN.
- **TIGA Atlas** / **TIGA Einstein** (later): only after Hunt clears the retrieval + usage gate.

### Performance targets
- Ideal ~5 s; hard upper bound **10 s** for the common path.
- Degraded mode: cited results without full LLM synthesis.

### Scale guardrails
- Tiered indexing (metadata-only vs text vs selective OCR).
- Aggressive dedupe + revision / latest logic.
- Embedding budgets and night-priority queues.
- Optional external assist only behind feature flags, token caps, timeout, and local fallback.

## CLI Reference

```
python tiga.py init         # create default config.yaml
python tiga.py discover     # preview what would be indexed
python tiga.py index        # incremental index (skip unchanged)
python tiga.py rebuild      # force full re-index
python tiga.py query <q>    # search from terminal
python tiga.py status       # index stats
python tiga.py eval         # search quality test
python tiga.py serve        # start FastAPI server (port 7860)
python tiga.py ui           # start Streamlit UI (port 8501)
python tiga.py health       # check Ollama + DB
python tiga.py extract <f>  # test extraction on a file
python tiga.py embed <q>    # test Ollama embedding
```

## Configuration

Edit `tiga_work/config.yaml` to set:
- `index_roots` — directories to scan
- `ollama.chat_model` — LLM (default: mistral)
- `retrieval.top_k_default` — results per query

Override work directory:
```
set TIGA_WORK_DIR=D:\tiga_data
```

## Stack

| Component | Role |
|-----------|------|
| Ollama (+ optional approved API) | Answer synthesis; local is default / fallback |
| nomic-embed-text (local) | Embeddings |
| Vector store | Semantic retrieval lane |
| SQLite + FTS5 | BM25 keyword lane |
| Local cross-encoder | Rerank (when enabled) |
| FastAPI | LAN API server |
| Web UI | Browser UI on the office LAN |

## Project Structure

```
tiga/
├── tiga.py          CLI entrypoint
├── config.py        Config loader
├── server.py        FastAPI LAN server
├── app.py           Streamlit UI (legacy / admin)
├── docs/
│   └── CONSTITUTION.md   Binding objectives & security options
├── core/
│   ├── db.py        SQLite + FTS5
│   ├── discover.py  File discovery
│   ├── extract.py   Text extraction
│   ├── infer.py     Project / typology inference
│   ├── vectors.py   Vector store + embeddings
│   ├── index.py     Indexing pipeline
│   ├── query.py     Hybrid search
│   ├── compose.py   Answer composer
│   ├── ocr.py       Gated OCR (opt-in)
│   └── eval.py      Search quality eval
├── tests/
│   └── test_config.py
└── tiga_work/       (gitignored — local data)
    ├── config.yaml
    ├── db/
    ├── vectors/
    ├── logs/
    └── reports/
```

## Later — Atlas / Einstein

Deferred until Hunt clears the constitution POC gate. Einstein (if built) reasons over Hunt evidence and may use a firm-approved API under constitution §8 — not a requirement to train on the archive.

Enable local Einstein experiments only behind config: `einstein.enable: true`.
