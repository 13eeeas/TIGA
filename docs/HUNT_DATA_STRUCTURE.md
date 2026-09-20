# Hunt data structure — how FIND feeds KNOW

Binding product rules: [`CONSTITUTION.md`](CONSTITUTION.md).  
Atlas tickets: [`ATLAS_HANDOFF.md`](ATLAS_HANDOFF.md).

TIGA sees **architecture projects**, not merely files. Hunt structures the
archive for retrieval; Atlas stores curated project memory as structured rows.

---

## 1. Ownership boundary

| Layer | Owns | Does not own |
|-------|------|--------------|
| **Hunt (FIND)** | Discovery, tiered ingest, BM25 + embeddings, rerank, project autoscope, authority/version signals from paths, citations, claim verification, thin-pack retry | Notion-parity editing, wiki chrome, collaboration, task suites |
| **Atlas (KNOW)** | Schema-v2 overlay rows, assemble-from-model pages, pins/facts, auto-curation approval UX, publish surfaces | Replacing Hunt retrieval |
| **Einstein (THINK)** | Reasoning / comparison over Atlas structure + Hunt evidence packs | Being the source of truth for facts |

Hunt must work **with generative LLM off**: return file + page/ref + excerpt +
open-original. Synthesis is optional.

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

| Field | Meaning |
|-------|---------|
| `project_code` | Firm project id from folder / card mapping |
| `folder_stage` | Competition → Concept → SD → DD → Tender → Construction (when folders say so) |
| `discipline` | Architect / C&S / M&E / … when path cues exist |
| `doc_type` / `canonical_category` | Brief, tender, CAD, BIM, renders, meetings, … |
| `revision` / `is_issued` / `is_latest` / `is_superseded` | Authority heuristics from names/dates/status |
| `duplicate_of` | Exact fingerprint collapse |
| OCR / spreadsheet / email fields | Selective extraction + review queues |

**Rule:** stage from folders, Rev D > Rev C, supersede heuristics → **no LLM**.
“Why the design changed between Rev C and Rev D” → **LLM over evidence**.

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

With compose/API disabled, Hunt still returns ranked cited hits (Gateway 1
**no-LLM** proof). Run: `python tiga.py eval --moat --llm-off`.

---

## 3. What Hunt emits for Atlas (seeds, not Atlas UI)

| Signal | Source in Hunt | Atlas use |
|--------|----------------|-----------|
| Project identity | `project_cards`, aliases | Page shell / identity |
| Stage / category | `folder_stage`, cards | Lifecycle seeds |
| Authority candidates | `is_superseded`, `is_latest`, revision | Document **proposals** (`source=hunt-proposal`) |
| Cited chunks | retrieval + verify | Evidence-native facts (`cite_paths`) |
| OCR / structured field candidates | OCR review queue, extractors | Candidate facts (draft until human approve) |
| Low-confidence review | OCR / classification confidence | Human gate |

### Atlas store (schema v2)

`tiga_work/atlas/{slug}.overlay.json`:

| Row kind | Meaning |
|----------|---------|
| `project` + `lifecycle` | Curated identity / stage |
| `documents` | authoritative / superseded / candidate / proposal |
| `facts` | Claims with `cite_paths` |
| `strategies` | Design-strategy tags |
| `team` | Roles / people / orgs |
| `decisions` | Dated decisions + citations |
| `precedents` | Related project links |
| `pins` / `hidden_paths` | Legacy wiki affordances |

### Boundary rules

1. **Hunt proposes; humans confirm.** `source=hunt-proposal` / `status=proposal`
   is never silent Atlas truth until Ticket B approval.
2. **Citations are paths.** Durable Atlas claims need `cite_paths` Hunt can retrieve.
3. **Pages assemble from rows.** `GET /api/atlas/page/{code}` → `model` (+
   `model_readiness`); free-form summary is optional gloss.
4. **Published gate** still requires curated identity (typology, client, stage,
   location) + pin + cited facts — never Needs-curation at 100/100.

---

## 4. Target ontology (product graph)

Hunt helps **detect**; Atlas **stores and curates**; Einstein **reasons**.

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

---

## 5. APIs

- `GET /api/atlas/projects`
- `GET /api/atlas/page/{code}` — includes `model` and `model_readiness`
- Mutators: `pin`, `fact`, `overview`, `document`, `strategy`, `team`,
  `decision`, `precedent`, `lifecycle`, `ask`
- Auto-curation: `proposals/stage`, `proposals/approve`, `proposals/reject`

---

## 6. Moat validation (before scaling ontology)

Do **not** demo only GFA lookup or a blank wiki page.

```bash
python tiga.py eval --moat
python tiga.py eval --moat --llm-off   # FIND baseline only
```

Fixture: [`../tests/fixtures/moat_validation.yaml`](../tests/fixtures/moat_validation.yaml).  
Hunt board: [TIGA Hunt](https://app.notion.com/p/3e1807c3efa681938a8bcccc126bd756).

**Gate:** if staff do not ask architecture-comparison questions, do not scale
the ontology. Also prove search with Ollama/API **disabled** still returns cited evidence.

---

## 7. Stop / defer in Hunt

- Notion-parity editing, comments, task suites, generic collaboration chrome  
- More RAG demos that do not deepen authority, stage, or strategy graph  
- Making generative LLM mandatory for FIND  
- Treating Hunt `is_latest` / `is_superseded` as Atlas SoT without human confirm  

Next Hunt tickets come from POC/staff trial evidence — not another plumbing wave.
