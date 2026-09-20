# Atlas handoff — tickets A–E

**Board:** [TIGA Atlas](https://app.notion.com/p/3e1807c3efa68120983ac9fe5f081350) → [Atlas Tasks](https://app.notion.com/p/61ce29756cf84e63aaf75f2f133e5a1d).  
Hunt (FIND) stays in this repo. Atlas (KNOW) structured rows live in
`tiga_work/atlas/*.overlay.json` (schema v2) until a dedicated store is justified.

Binding principles: [`CONSTITUTION.md`](CONSTITUTION.md) §1, §3, §9.  
Hunt boundary: [`HUNT_DATA_STRUCTURE.md`](HUNT_DATA_STRUCTURE.md).  
Thin in-Hunt wiki: [`ATLAS_IN_HUNT.md`](ATLAS_IN_HUNT.md).

**Principle:** Atlas **database** matters more than Atlas **UI**. Pages assemble
from the model. Steal Notion UX patterns; do not chase Notion wiki/collab parity.

| Ticket | Title | Priority | Status |
|--------|-------|----------|--------|
| **A** | Atlas data model (priority over UI) | P0 | **Shipped** — `core/atlas_model.py` |
| **B** | Auto-curation loop | P0 | **Shipped** — stage / approve / reject APIs |
| **C** | Asymmetric Atlas UI | P1 | Depends on A |
| **D** | Publish surfaces (MCP / Notion sync) | P2 | After A; Notion is a renderer |
| **E** | Einstein last | P2 | Reason over Atlas + Hunt evidence |

---

## Ownership

| Owner | Scope |
|-------|--------|
| **Hunt / this repo** | FIND layer; emit seeds (path metadata, citations, OCR candidates); keep LLM-off cited search |
| **Atlas (KNOW)** | Structured overlay rows, assemble-from-model page API, auto-curation approval, publish/MCP, Einstein-over-Atlas |

---

## A — Atlas data model (shipped)

### Goal

First-class rows, not a free-form wiki dump.

### Store

- Authoritative vs superseded documents  
- Lifecycle stage  
- Design-strategy tags  
- Team  
- Cited facts (evidence-native)  
- Decisions  
- Precedent links  

Pages **assemble from the model**. UI stays thin.

### Implementation (this repo)

- `core/atlas_model.py` — schema v2 rows + `assemble_from_rows`
- `core/atlas_wiki.py` — overlay normalize/save; Published gate (curated identity)
- `GET /api/atlas/page/{code}` → `model` + `model_readiness`
- Mutators: `document`, `strategy`, `team`, `decision`, `precedent`, `lifecycle`
  (plus legacy `pin`, `fact`, `overview`, `ask`)

Hunt `is_latest` / `is_superseded` seed **proposals** only (`source=hunt-proposal`).
Machine drafts are never silent database truth (Ticket B).

**Acceptance met:** A project page can be regenerated from structured rows +
citations; blank “wiki only” is not the source of truth.

---

## B — Auto-curation loop (shipped)

### Goal

Invert Notion’s maintenance burden.

```
Archive change (Egnyte/NAS)
  → Hunt detects / extracts candidates
  → POST .../proposals/stage
  → human approve / reject
  → Atlas overlay updated (confirmed or rejected)
```

### Implementation

- `POST /api/atlas/page/{code}/proposals/stage` — persist Hunt `is_latest` /
  `is_superseded` as `source=hunt-proposal` rows
- `POST /api/atlas/page/{code}/proposals/approve` — `{path}` → `status=confirmed`,
  `source=curated` (authoritative also lifts a pin)
- `POST /api/atlas/page/{code}/proposals/reject` — `{path}` → rejected; will not
  re-stage as truth
- Model exposes `documents.pending_proposals`

**Acceptance met:** tender/façade supersede path can be staged, one doc approved
into authoritative SoT, the other rejected without silent truth.

---

## C — Asymmetric Atlas UI (steal UX, not ontology)

**Steal from Notion (UX only):** calm project pages, inline edit, properties,
mentions, comments, filtered views, command palette, human curation on machine drafts.

**Beat Notion on:** project-native pages, evidence-native claims, authority-aware
docs, intrinsic lifecycle, strategy graph, auto-curation.

**Out of scope:** free-form Notion wiki / collab / task-suite parity.

---

## D — Publish surfaces (MCP / Notion sync)

Only after Ticket A (done) and preferably B:

- `tiga.get_project` / MCP for ChatGPT/Cursor  
- Optional **Publish to Notion** / property populate  

Notion becomes a **renderer**, not the ontology.

---

## E — Einstein last

Reason over **Atlas structure + Hunt evidence packs**. Feature-flagged enterprise
API under Constitution §8.

Not a chat product until Atlas can answer: “what is this project?” and
“what superseded what?” (`model_readiness` on the page API).

---

## Hunt will keep providing

- Path/authority signals (`folder_stage`, `revision`, `is_superseded`, …)  
- Hybrid retrieval + citations + claim verify + thin-pack retry  
- OCR / structured extraction **candidates** + review queues  
- Project cards / aliases as identity seeds  
- Moat / LLM-off eval: `python tiga.py eval --moat [--llm-off]`

See [`HUNT_DATA_STRUCTURE.md`](HUNT_DATA_STRUCTURE.md).
