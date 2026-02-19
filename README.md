# TIGA Hunt

RAG-powered archive search for architecture firms.
Plain English queries → top-5 cited results in under 5 seconds.

## Quick Start

```
setup.bat        # first time only
run.bat          # start server + UI
```

Then open **http://localhost:8501** in any browser on the LAN.

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
