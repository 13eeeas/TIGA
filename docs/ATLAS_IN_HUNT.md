# Atlas inside Hunt — cited project wiki

**North star:** Notion-like browse + Wikipedia-like contribution + Grokopedia-like auto draft, **always cited**.

## Product map (one LAN portal)

| Mode | URL | Role |
|------|-----|------|
| **Search** | `/` | Hunt — find files |
| **Projects** | `/projects` | Atlas — auto project pages + wiki edits |
| **Ask** | `/projects#ask` | Einstein-lite — pins + cited facts only |

## Behaviour

1. **Auto (Grokopedia)** — opening a project runs Hunt seed queries + project card → candidates + draft facts (uncited = rumours).
2. **Wiki (Wikipedia)** — anyone can **Pin as truth**, **Hide** junk, **Save fact** with a citation path. No code.
3. **Gate** — health grade; **Published** when ≥1 pin and all kept facts are cited.
4. **Ask** — refuses until a pin exists; never invents from uncited auto facts.

Overlays live in `tiga_work/atlas/*.overlay.json` (survive re-index).

## API

- `GET /api/atlas/projects`
- `GET /api/atlas/page/{code}`
- `POST /api/atlas/page/{code}/pin|hide|unhide|fact|ask`

## Standalone Atlas repo

`13eeeas/TIGA-Atlas` remains the design lab / offline tools. **Staff only use Hunt.**
