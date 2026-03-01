# TIGA Atlas + TIGA Einstein Architecture (Local-First, 30TB Scale)

## Goal
Build a ChatGPT-Projects-like experience over 30TB NAS archives **without** duplicating archive size in index storage.

## Core principle: Represent, don't copy
- Keep archive files in place on NAS.
- Store only compact derived data:
  - file metadata rows
  - extracted text chunks (selected file types)
  - vector embeddings (for selected chunks)
  - optional summaries/knowledge graph edges

This mirrors how web search engines operate: they index representations, not full raw internet payloads.

## Proposed two-mind system

## 1) TIGA Einstein (Expert Mind)
Stable domain knowledge and reasoning style:
- architectural delivery process
- authority workflows (BCA/URA/CAAS/etc.)
- firm conventions and writing style

Implementation on 4090+i9:
- local instruct model (7B–14B quantized)
- prompt/system memory for policy and style
- optional adapter/LoRA later

## 2) TIGA Atlas (Studied Archive Mind)
Continuously updated retrieval memory from NAS:
- project/file/chunk index
- project cards, aliases, signals
- stage/category/discipline metadata
- query-time retrieval + rerank + citation validation

Atlas should answer "what do we know in this archive?" while Einstein decides "how to reason and explain it".

## Query pipeline (recommended)
1. Intent route: file-finder vs fact answer vs cross-project compare.
2. Candidate retrieval:
   - BM25 lane (fast lexical)
   - vector lane (semantic)
3. Metadata rerank:
   - boost project_code/stage/category recency
4. Evidence pack:
   - top chunks + exact file citations
5. Synthesis:
   - Einstein composes concise answer from evidence only
6. Safety:
   - citation check before return

## Data footprint strategy (very important)
To avoid "30TB in → 30TB+ index out":

1. Tiered indexing
- Tier A metadata-only: BIM/CAD/media binaries.
- Tier B text extraction: PDF/DOCX/PPTX/TXT/MSG/XLSX.
- Tier C selective OCR: only high-value folders or on-demand.

2. Chunk budget controls
- max chunks per file
- dedupe by content hash
- skip low-value boilerplate pages/slides

3. Embedding budget controls
- embed only chunks likely to be queried
- background embed queues by priority
- optional per-project cold storage for low-traffic vectors

4. Summaries over raw replication
- store compact project/timebox summaries
- keep links to source chunks for traceability

## Why this scales
- Most value comes from indexing textual/semantic signals, not raw binaries.
- Many archive files are duplicates/revisions/exports; dedupe and latest-selection reduce index growth.
- Retrieval quality increases with metadata + reranking, not just more embeddings.

## Practical milestones
1. Hunt v1.5: fast discovery + improved folder convention parsing + eval harness.
2. Atlas v1: project-centric retrieval memory and cross-project knowledge views.
3. Einstein v1: local reasoning model + policy-aware synthesis over Atlas evidence.
4. Atlas+Einstein v2: optional API reranker for low-cost quality bump with local fallback.
