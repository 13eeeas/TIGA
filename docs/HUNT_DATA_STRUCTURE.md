# Hunt data structure — how FIND feeds KNOW

Binding product rules: [`CONSTITUTION.md`](CONSTITUTION.md).  
Atlas ownership / tickets: [`ATLAS_HANDOFF.md`](ATLAS_HANDOFF.md).

This doc records how **Hunt** structures archive data today, what it emits for **Atlas**, and what the target architecture ontology looks like. It is the Hunt-side reflection of the intelligence-layer strategy: TIGA sees **architecture projects**, not merely files.

---

## 1. Ownership boundary

| Layer | Owns | Does not own |
|-------|------|--------------|
| **Hunt (FIND)** | Discovery, tiered ingest, BM25 + embeddings, rerank, project autoscope, authority/version signals from paths, citations, claim verification, thin-pack retry | Notion-parity editing, wiki chrome, collaboration, task suites |
| **Atlas (KNOW)** | Ontology graph storage, curated facts, pins, publish surfaces, auto-curation approval UX | Replacing Hunt retrieval |
| **Einstein (THINK)** | Reasoning / comparison over Atlas structure + Hunt evidence packs | Being the source of truth for facts |

Hunt must work **with generative LLM off**: return file + page/ref + excerpt + open-original. Synthesis is optional.

---

## 2. What Hunt stores today (file / evidence layer)

### Tables (SQLite)

| Table | Role |
|-------|------|
| `files` | One row per discovered path: fingerprint, lane, status, path-parsed metadata |
| `chunks` | Extracted text slices + content hash; feed for FTS + embeddings |
| `chunks_fts` | FTS5 BM25 lane over chunk text / title / path |
| `projects` / `project_cards` | Inferred + curated project identity (typology, stage, GFA, team, milestones, …) |
| `project_aliases` | Name matching for scoping |
| Vectors (work dir) | Embeddings for hybrid retrieval |

### Path-parsed file signals (deterministic)

Hunt already extracts architecture-native metadata from paths and names (no generative LLM):

| Field | Meaning |
|-------|---------|
| `project_code` | Firm project id from folder / card mapping |
| `folder_stage` | Competition → Concept → SD → DD → Tender → Construction (when folders say so) |
| `discipline` | Architect / C&S / M&E / … when path cues exist |
| `doc_type` / `canonical_category` | Brief, tender, CAD, BIM, renders, meetings, … |
| `revision` / `is_issued` / `is_latest` / `is_superseded` | Authority heuristics from names/dates/status |
| `duplicate_of` | Exact fingerprint collapse |
| OCR / spreadsheet / email fields | Selective extraction + review queues |

**Rule:** stage from folders, Rev D > Rev C, supersede heuristics → **no LLM**. “Why the design changed between Rev C and Rev D” → **LLM over evidence**.

### Query pipeline (evidence pack)

```
Query
  → route / project autoscope
  → BM25 lane + vector lane
  → merge + superseded penalty + project boost
  → local rerank
  → evidence pack (~8–15 chunks)
  → optional thin-pack retry
  → citations (+ optional claim verify / compose)
```

With compose/API disabled, Hunt still returns ranked cited hits. That is the Gateway 1 **no-LLM** proof.

---

## 3. What Hunt emits for Atlas (seeds, not Atlas UI)

Hunt should keep emitting **signals** Atlas consumes — not grow into a Notion clone:

| Signal | Source in Hunt | Atlas use |
|--------|----------------|-----------|
| Project identity | `project_cards`, aliases | Page shell / identity |
| Stage / category | `folder_stage`, cards | Lifecycle |
| Authority candidates | `is_superseded`, `is_latest`, revision | Authoritative vs superseded docs |
| Cited chunks | retrieval + verify | Evidence-native facts |
| OCR / structured field candidates | OCR review queue, extractors | Candidate facts (draft until human approve) |
| Low-confidence review | OCR / classification confidence | Human gate |

Overlays today: `tiga_work/atlas/*.overlay.json` (summary, pins, facts, hidden paths). Richer first-class rows are **Atlas project** work — see handoff tickets.

---

## 4. Target ontology (product graph)

This graph is the product. Hunt helps **detect**; Atlas **stores and curates**; Einstein **reasons**.

```
PROJECT
├── Typology, Location, Client, Stage, Dates, GFA, Height
├── DESIGN STRATEGIES (passive cooling, NV, landscape, façade, structure, …)
├── TEAM (Architect, C&S, M&E, ESD, Landscape, …)
├── PROJECT HISTORY (Competition → Concept → SD → DD → Tender → Construction)
├── DOCUMENTS (authoritative / superseded / revision / issue status)
├── DECISIONS
├── DETAILS
└── PRECEDENT RELATIONSHIPS
```

Pages assemble **from the model**. UI stays thin. Publishing to Notion / MCP (`tiga.get_project`) is a later surface, not the core.

---

## 5. Moat validation (before scaling ontology)

Do **not** demo only GFA lookup or a blank wiki page. Validate with 10–20 staff-shaped questions like:

> Show WOHA education projects under 24 m where naturally ventilated circulation formed part of the social strategy. Compare section strategy, cooling, structure, and presentation material — with evidence (file + page).

Fixture: [`../tests/fixtures/moat_validation.yaml`](../tests/fixtures/moat_validation.yaml).

**Gate:** if staff do not ask questions of this shape, do not scale the ontology. Also prove search with Ollama/API **disabled** still returns cited evidence.

---

## 6. Stop / defer in Hunt

- Notion-parity editing, comments, task suites, generic collaboration chrome  
- More RAG demos that do not deepen authority, stage, or strategy graph  
- Making generative LLM mandatory for FIND  

Next Hunt tickets should come from POC/staff trial evidence (wrong supersede, missing stage, bad project scope) — not another plumbing wave.
