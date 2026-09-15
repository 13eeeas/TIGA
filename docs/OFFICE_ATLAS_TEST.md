# Office test — Projects cited wiki (tomorrow)

**Goal:** One auto-draft project page that a non-coder can wiki-curate in the browser.

North star: Notion browse + Wikipedia contribution + Grokopedia auto-draft, always cited.

## Before staff arrive (you)

1. On office PC, update Hunt from the git clone (do **not** `git pull` / reset):
   ```bat
   update.bat
   ```
   Linux/macOS: `bash update.sh`. Confirm the report’s old/new SHA. Then restart Hunt (`launcher.bat` — not START-HERE). If the updater refuses (dirty source, divergence), follow [`docs/WOHA_UPDATE.md`](WOHA_UPDATE.md); do not reset.
2. Confirm Search still works. Health URL is the **configured** `server.port` in `tiga_work/config.yaml`, not a hardcoded `7860`.
3. Open **Projects** in the header (or `http://<LAN>:<configured-port>/projects`).

## 30–45 min ride (one project)

Pick one indexed project (prefer **NUS BIZ3** if present).

### A. Auto draft (Grokopedia)
- [ ] Project appears in the list with a wiki **blurb** (file count lives under **Index**)
- [ ] Open it — page leads with what / client or typology / stage / location
- [ ] Badge shows **Needs curation** until identity fields + pin + cited facts pass
- [ ] Auto facts without cites look like **rumours** (not trusted)
- [ ] Empty cards never show **Published 100/100**

### B. Wiki contribute (Wikipedia — no code)
- [ ] **Save overview** (blurb + typology / client / stage / location)
- [ ] **Pin as truth** on overview / GA / authority (2–3 pins)
- [ ] **Hide** one junk candidate (e.g. Copy of…)
- [ ] **Save fact** with a real NAS citation path (2 facts)
- [ ] Or use **Add cite…** on an auto fact
- [ ] **Published** appears only after identity fields + pin + cited facts — never a fake 100/100

### C. Ask (Einstein-lite)
- [ ] Before pins: Ask should refuse / lock
- [ ] After pins: Ask returns answer grounded in pins + cited facts
- [ ] No invented numbers without a cite

### D. Survive refresh
- [ ] Reload the page — pins/facts/hides still there (`tiga_work/atlas/*.overlay.json`)

## Capture for Cursor / Codex
Drop notes in chat or `tiga_work/` export:

- Project code
- What felt Notion-like / what didn’t
- 3 bugs, 3 wins
- Screenshot if health/grade wrong

## Pass bar for tomorrow
- Non-coder can pin + cite without you at their shoulder for more than 2 minutes
- Search still works
- Overlay persists after refresh
