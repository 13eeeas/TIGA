# Atlas handoff — tickets A–E

Binding board: Notion **TIGA Atlas → Atlas Tasks**.  
Hunt (FIND) stays in this repo. Atlas (KNOW) data model lives here as structured
overlay rows under `tiga_work/atlas/*.overlay.json` until a dedicated store is
justified.

**Principle:** Atlas **database** matters more than Atlas **UI**. Pages assemble
from the model. Steal Notion UX patterns; do not chase Notion wiki/collab parity.

| Ticket | Title | Priority | Notes |
|--------|-------|----------|-------|
| **A** | Atlas data model (priority over UI) | P0 | First-class rows; page regenerated from model + citations |
| **B** | Auto-curation loop | P0 | Hunt proposes → human approve → Atlas updated |
| **C** | Asymmetric Atlas UI | P1 | Steal Notion UX only; depends on A |
| **D** | Publish surfaces (MCP / Notion sync) | P2 | After A is real; Notion is a renderer |
| **E** | Einstein last | P2 | Reason over Atlas structure + Hunt evidence packs |

Also see:

- [`ATLAS_IN_HUNT.md`](ATLAS_IN_HUNT.md) — product map inside Hunt
- [`HUNT_DATA_STRUCTURE.md`](HUNT_DATA_STRUCTURE.md) — Hunt → Atlas boundary
- [`CONSTITUTION.md`](CONSTITUTION.md) — intelligence ownership

---

## A — Atlas data model (priority over UI)

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

Self-assembling project page: identity, overview, strategies, data, team,
timeline, decisions, authoritative/superseded docs, related/precedent.

### Acceptance

A project page can be regenerated from structured rows + citations; blank
“wiki only” is not the source of truth.

### Implementation (this repo)

- `core/atlas_model.py` — schema v2 rows + `assemble_from_rows`
- `core/atlas_wiki.py` — overlay normalize/save; `GET /api/atlas/page/{code}`
  returns `model` + `model_readiness`
- APIs: `document`, `strategy`, `team`, `decision`, `precedent`, `lifecycle`

Hunt `is_latest` / `is_superseded` seed **proposals** only (`source=hunt-proposal`).
Machine drafts are never silent database truth (Ticket B).

---

## B — Auto-curation loop

### Goal

Invert Notion’s maintenance burden.

```
Archive change (Egnyte/NAS)
  → Hunt detects / extracts candidates
  → stage / revision / authority proposals
  → human approve / reject
  → Atlas (and optional Notion) updated
```

### Acceptance

At least one end-to-end path: new tender deck or superseding façade report →
candidate flag → human approve → Atlas fact/doc row updated.

---

## C — Asymmetric Atlas UI (steal UX, not ontology)

### Steal from Notion (UX only)

Calm project pages, inline edit, properties, mentions, comments, filtered views,
command palette, human curation on machine drafts.

### Beat Notion on

Project-native pages, evidence-native claims, authority-aware docs, intrinsic
lifecycle, strategy graph, auto-curation.

### Out of scope

Free-form Notion wiki / collab / task-suite parity. Depends on Ticket A.

---

## D — Publish surfaces (MCP / Notion sync)

Only after Ticket A is real.

- `tiga.get_project` / MCP for ChatGPT/Cursor
- Optional **Publish to Notion** / property populate

Notion becomes a **renderer**, not the ontology.

---

## E — Einstein last (reason over Atlas + Hunt)

Reason over **Atlas structure + Hunt evidence packs**. Feature-flagged enterprise
API under Constitution §8.

Not a chat product until Atlas can answer: “what is this project?” and
“what superseded what?” (`model_readiness` on the page API).
