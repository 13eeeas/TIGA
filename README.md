# TIGA Hunt

RAG-powered archive search for architecture firms.
Plain English queries → top-5 cited results in under 5 seconds.

## Quick Start

```
setup.bat        # first time only
run.bat          # start server + UI
```

Then open **http://localhost:8501** in any browser on the LAN.

## Product Ambition

TIGA is designed to become a **local, large-scale archive intelligence system** for architecture firms.

### North Star
- Index and understand **30TB+ of NAS project data** across 100+ projects.
- Let teams ask plain-English questions and get **relevant, cited answers** like ChatGPT Projects.
- Keep the system **lean and efficient** on local hardware (e.g. RTX 4090 + i7), with optional low-cost API augmentation.

### Design Principles
- **Local-first**: default operation on-prem/LAN with no required external API.
- **Represent, don't replicate**: index compact metadata/chunks/embeddings instead of duplicating raw archive size.
- **Evidence-first answers**: every answer should map back to files/chunks with citations.
- **Incremental by default**: changed-only indexing, resumable pipelines, and fast warm scans.

### Roadmap Language
- **TIGA Hunt**: ingestion, indexing, retrieval, and cited answering over live archives.
- **TIGA Atlas**: project memory graph ("grokopedia" for your archive) that tracks entities, decisions, revisions, and cross-project patterns.
- **TIGA Einstein**: expert reasoning layer that combines domain know-how with Atlas evidence to answer like a senior architect/director.

### Scale Guardrails
To avoid needing 30TB+ extra storage to operate on 30TB archives:
- Tiered indexing (metadata-only vs text extraction vs selective OCR).
- Aggressive dedupe (content hashes, revision/latest logic).
- Embedding budgets and priority queues per project/stage.
- Optional API rerank/assist behind feature flags, timeout, and local fallback.

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
| Ollama + mistral | Local LLM (zero external API) |
| nomic-embed-text | Embeddings |
| ChromaDB | Vector search lane |
| SQLite + FTS5 | BM25 keyword lane |
| FastAPI | LAN API server |
| Streamlit | Browser UI |

## Project Structure

```
tiga/
├── tiga.py          CLI entrypoint
├── config.py        Config loader
├── server.py        FastAPI LAN server
├── app.py           Streamlit UI
├── core/
│   ├── db.py        SQLite + FTS5
│   ├── discover.py  File discovery
│   ├── extract.py   Text extraction
│   ├── infer.py     Project / typology inference
│   ├── vectors.py   ChromaDB + embeddings
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

## Phase 2 — TIGA Einstein

Planned. Locally-trained model with two layers:
- Trained "senior architect/director" knowledge core
- Live indexed archive (built by Hunt)

Enable in config: `einstein.enable: true`
