# Hunt data structure → Atlas boundary

Hunt is the **FIND** layer. Atlas is the **KNOW** layer. This doc states what
Hunt already stores (and exposes) so Atlas can consume it without inventing a
parallel ontology.

## Hunt owns (authoritative for retrieval)

| Signal | Where | Atlas use |
|--------|-------|-----------|
| File path / name / project_id | `files` | Document rows, citations |
| `is_latest` / `is_superseded` / revision | `files` (path parser) | Authority **proposals** |
| Stage / typology / client / location | `project_cards` (+ overlay) | Identity / lifecycle seeds |
| Chunks + embeddings | chunk / vector tables | Evidence packs for Ask / Einstein |
| Query candidates + citations | `/api/query` | Cite paths for facts |

## Atlas owns (authoritative for project memory)

Stored in `tiga_work/atlas/{slug}.overlay.json` (schema_version **2**):

| Row kind | Meaning |
|----------|---------|
| `project` + `lifecycle` | Curated identity / stage |
| `documents` | authoritative / superseded / candidate |
| `facts` | Claims with `cite_paths` |
| `strategies` | Design-strategy tags |
| `team` | Roles / people / orgs |
| `decisions` | Dated decisions + citations |
| `precedents` | Related project links |
| `pins` / `hidden_paths` | Legacy wiki affordances (pins lift into documents) |

## Boundary rules

1. **Hunt proposes; humans confirm.** Rows with `source=hunt-proposal` (or
   `status=proposal`) are never silent Atlas truth until Ticket B approval.
2. **Citations are paths.** Every durable Atlas claim needs `cite_paths` into
   archive files Hunt can retrieve.
3. **Pages assemble from rows.** `GET /api/atlas/page/{code}` → `model` is
   regenerated from structured rows; free-form summary is optional gloss.
4. **No generative LLM required** for Hunt or Atlas baseline (Constitution).
   Einstein (Ticket E) is the THINK layer over Atlas + Hunt packs.

## APIs

Hunt → Atlas consumers:

- `GET /api/atlas/projects`
- `GET /api/atlas/page/{code}` — includes `model` and `model_readiness`
- Mutators: `pin`, `fact`, `overview`, `document`, `strategy`, `team`,
  `decision`, `precedent`, `lifecycle`

See [`ATLAS_HANDOFF.md`](ATLAS_HANDOFF.md) for tickets A–E.
