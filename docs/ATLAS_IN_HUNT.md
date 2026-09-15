# Atlas inside Hunt — cited project wiki

**North star:** Notion-like browse + Wikipedia-like contribution + Grokopedia-like auto draft, **always cited**.

## Product map (one LAN portal)

| Mode | URL | Role |
|------|-----|------|
| **Search** | `/` | Hunt — find files |
| **Projects** | `/projects` | Atlas — wiki blurb + pins/facts (same UI language as Hunt) |
| **Ask** | `/projects#ask` | Einstein-lite — pins + cited facts only |

## Behaviour

1. **Wiki blurb first** — opening a project shows a short human overview (what / client or typology / stage / location if known), readable in under five seconds. File counts live under **Index**, never as the hero.
2. **Auto (Grokopedia)** — Hunt seed queries + project card → candidates + draft facts (uncited = rumours).
3. **Wiki (Wikipedia)** — anyone can edit the overview, **Pin as truth**, **Hide** junk, **Save fact** with a citation path. No code.
4. **Gate** — **Published** (and a health score) only when curated identity fields + ≥1 pin + ≥1 cited fact (and no uncited kept facts) exist. Empty / Needs-curation cards never show Published 100/100.
5. **Ask** — refuses until a pin exists; never invents from uncited auto facts.

Overlays live in `tiga_work/atlas/*.overlay.json` (survive re-index). Overview fields are `summary` + `project` on that overlay.

## API

- `GET /api/atlas/projects`
- `GET /api/atlas/page/{code}`
- `POST /api/atlas/page/{code}/pin|hide|unhide|fact|overview|ask`

## Standalone Atlas repo

`13eeeas/TIGA-Atlas` remains the design lab / offline tools. **Staff only use Hunt.**

## Rollback

This slice is additive on the existing overlay schema (`summary` + `project` + `pins` / `facts` / `hidden_paths`). Revert the PR (or `update.sh --sha <previous>`) to restore the previous Projects UI and Published gate. Existing `tiga_work/atlas/*.overlay.json` files stay valid either way — do not delete them.
