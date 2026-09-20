# Atlas project handoff (from Hunt)

**Status:** Filed for the Atlas project to own. Do **not** implement these as Hunt plumbing tickets.

Notion destination: **[TIGA Atlas](https://app.notion.com/p/3e1807c3efa68120983ac9fe5f081350)** — tasks in [Atlas Tasks](https://app.notion.com/p/61ce29756cf84e63aaf75f2f133e5a1d).  
Standalone design lab (if used): `13eeeas/TIGA-Atlas` — staff still use Hunt for day-to-day.

Binding principles: [`CONSTITUTION.md`](CONSTITUTION.md) §1, §3, §9.  
Hunt boundary: [`HUNT_DATA_STRUCTURE.md`](HUNT_DATA_STRUCTURE.md).  
Thin in-Hunt wiki today: [`ATLAS_IN_HUNT.md`](ATLAS_IN_HUNT.md).

---

## Ownership

| Owner | Scope |
|-------|--------|
| **Hunt / this repo** | FIND layer; emit seeds (path metadata, citations, OCR candidates); keep LLM-off cited search |
| **Atlas project** | KNOW layer: data model, auto-curation, asymmetric UI, publish/MCP, Einstein-over-Atlas |

---

## Ticket A — Atlas data model (priority over UI)

**Goal:** First-class rows, not a free-form wiki dump.

Store:

- Authoritative vs superseded documents  
- Lifecycle stage  
- Design-strategy tags  
- Team  
- Cited facts (evidence-native)  
- Decisions  
- Precedent links  

Pages **assemble from the model**. UI stays thin.

Self-assembling project page sections: identity, overview, strategies, data, team, timeline, decisions, authoritative/superseded docs, related/precedent.

**Acceptance:** A project page can be regenerated from structured rows + citations; blank “wiki only” is not the source of truth.

---

## Ticket B — Auto-curation loop

**Goal:** Invert Notion’s maintenance burden.

```
Archive change (Egnyte/NAS)
  → Hunt detects / extracts candidates
  → stage / revision / authority proposals
  → human approve / reject
  → Atlas (and optional Notion) updated
```

**Acceptance:** At least one end-to-end path: new tender deck or superseding façade report → candidate flag → human approve → Atlas fact/doc row updated. Machine drafts are never silent database truth.

---

## Ticket C — Asymmetric Atlas UI (steal UX, not ontology)

**Steal from Notion (UX only):** calm project pages, inline edit, properties, mentions, comments, filtered views, command palette, human curation on machine drafts.

**Beat Notion on:** project-native pages, evidence-native claims, authority-aware docs, intrinsic lifecycle, strategy graph, auto-curation.

**Explicitly out of scope:** free-form Notion wiki / collab / task-suite parity.

---

## Ticket D — Publish surfaces (later)

Only after Ticket A is real:

- `tiga.get_project` / MCP for ChatGPT/Cursor  
- Optional **Publish to Notion** / property populate  

Interop after the model exists — not a sync science project before then.

---

## Ticket E — Einstein last

Reason over **Atlas structure + Hunt evidence packs**. Feature-flagged enterprise API under Constitution §8.

**Not** a chat product until Atlas can answer: “what is this project?” and “what superseded what?”

---

## Hunt will keep providing

- Path/authority signals (`folder_stage`, `revision`, `is_superseded`, …)  
- Hybrid retrieval + citations + claim verify + thin-pack retry  
- OCR / structured extraction **candidates** + review queues  
- Project cards / aliases as identity seeds  

See [`HUNT_DATA_STRUCTURE.md`](HUNT_DATA_STRUCTURE.md) §3.
