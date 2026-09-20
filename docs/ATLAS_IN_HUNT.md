# Atlas inside Hunt — cited project wiki

**North star:** Notion-like browse + Wikipedia-like contribution + Grokopedia-like auto draft, **always cited**. Structured rows (schema v2) are the source of truth; free-form blurb is optional gloss.

## Product map (one LAN portal)

| Mode | URL | Role |
|------|-----|------|
| **Search** | `/` | Hunt — find files |
| **Projects** | `/projects` | Atlas — wiki blurb + pins/facts + structured model |
| **Ask** | `/projects#ask` | Einstein-lite — pins + cited facts only |

## Behaviour

1. **Wiki blurb first** — opening a project shows a short human overview (what / client or typology / stage / location if known), readable in under five seconds. File counts live under **Index**, never as the hero.
2. **Auto (Grokopedia)** — Hunt seed queries + project card → candidates + draft facts (uncited = rumours). Hunt `is_latest` / `is_superseded` → document **proposals** only.
3. **Wiki (Wikipedia)** — anyone can edit the overview, **Pin as truth**, **Hide** junk, **Save fact** with a citation path. No code.
4. **Gate** — **Published** (and a health score) only when curated identity fields + ≥1 pin + ≥1 cited fact (and no uncited kept facts) exist. Empty / Needs-curation cards never show Published 100/100.
5. **Ask** — refuses until a pin exists; never invents from uncited auto facts.
6. **Model** — `GET /api/atlas/page/{code}` returns `model` + `model_readiness` assembled from structured rows (`core/atlas_model.py`).

Overlays live in `tiga_work/atlas/*.overlay.json` (schema_version **2**; survive re-index). Overview fields are `summary` + `project` plus row kinds (`documents`, `strategies`, `team`, `decisions`, `precedents`, `lifecycle`).

## API

- `GET /api/atlas/projects`
- `GET /api/atlas/page/{code}` — includes `model`, `model_readiness`
- `POST /api/atlas/page/{code}/pin|hide|unhide|fact|overview|ask`
- `POST /api/atlas/page/{code}/document|strategy|team|decision|precedent|lifecycle`
- `POST /api/atlas/page/{code}/proposals/stage|approve|reject` — Ticket B auto-curation

## Tickets / boundary

- Board + A–E: [`ATLAS_HANDOFF.md`](ATLAS_HANDOFF.md) — Tickets **A** and **B shipped**; **C** UI next
- Hunt → Atlas signals: [`HUNT_DATA_STRUCTURE.md`](HUNT_DATA_STRUCTURE.md)
- Notion: [TIGA Atlas](https://app.notion.com/p/3e1807c3efa68120983ac9fe5f081350)

`13eeeas/TIGA-Atlas` remains the design lab / offline tools if used. **Staff only use Hunt.**

## Rollback

Additive on the overlay schema. v1 overlays normalize to v2 on load. Revert the PR (or `update.sh --sha <previous>`) to restore the previous Projects UI and Published gate. Do not delete existing `tiga_work/atlas/*.overlay.json` files.
