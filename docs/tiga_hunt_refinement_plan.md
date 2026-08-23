# TIGA Hunt Refinement Plan (30TB Local Archive Vision)

> Binding rules: [`CONSTITUTION.md`](CONSTITUTION.md). This plan is implementation detail for Hunt; scale to 30TB only after the POC gate.

This plan focuses on scaling discovery/index/query for large architecture archives while staying **LAN-first**, cost-conscious, and aligned with evidence-pack API options when firm policy allows.

## 1) Discovery throughput (highest priority)

### Current strengths
- Incremental unchanged detection already exists.
- `os.scandir` walk is used for lower overhead.

### Gaps to close
- First-run scans over huge trees are still expensive.
- Directory-level pruning should happen before file checks.

### Immediate actions
1. Prune excluded directory names during walk (done in current PR).
2. Add discovery metrics: files/sec, dirs/sec, hash-time %, DB write-time %.
3. Add configurable fingerprint strategy:
   - `full` (sha256 content)
   - `metadata` (size+mtime)
   - `sampled` (head+tail blocks)

## 2) Relevance quality at project scale

### Current strengths
- Hybrid BM25 + vector retrieval with fallback and citations.

### Gaps to close
- BM25 token OR query can over-broaden on large corpora.
- Need stronger project-aware ranking and metadata filters by default.

### Immediate actions
1. Add project-level reranking features:
   - boost same `project_code`
   - boost canonical category/stage matches
2. Introduce query intent classifier (fact lookup vs file-finder vs cross-project compare).
3. Build offline eval set from real user queries + expected citations, then gate changes on NDCG/Recall.

## 3) Index economics for 30TB

### Immediate actions
1. Tiered indexing:
   - Tier A: metadata-only for binaries/media/CAD
   - Tier B: text extraction for docs/email/spreadsheets
   - Tier C: OCR only on demand or policy rules
2. Scheduling:
   - low-priority background indexing windows
   - changed-only daily ingest
3. Storage budgeting dashboard:
   - SQLite size, vector size, chunks per project, embed queue depth

## 4) LAN-first + optional API assist

### Strategy
- Keep baseline fully on the office LAN and deterministic (constitution Option A).
- Optional answer synthesis via evidence-pack API / private VPC / on-prem appliance (Options B–D) only after firm approval.
- Prefer **local** reranker; do not send candidates off-LAN for rerank unless policy explicitly allows.

### Guardrails
- Feature flag per capability; kill switch.
- Evidence pack only (≈8–15 chunks); hard token cap.
- Request/response caching; hard timeout + local fallback.
- ACL before retrieval; audit source ids used.
- No archive folder upload or unapproved SaaS (see constitution §8).

## 5) Operational hardening

1. Add benchmark command (`tiga.py benchmark-discover`) for repeatable perf tests.
2. Add integrity checks for orphan rows / stale vectors.
3. Add observability endpoint for indexing/query latency histograms.

---

## Suggested success targets
- Query user-facing latency target: **ideal 5s, max 10s** for normal archive queries.
- Warm incremental discover on 200k-file project: < 3 minutes
- Query P95 latency (hybrid): < 2.5s
- Top-5 citation relevance (human eval): +20% from current baseline
- Full local mode: works with Ollama down (BM25 fallback + cited snippets)
