# TIGA Audit Log

**Date:** 2026-08-23  
**Branch:** `cursor/project-constitution-lan-api-37a0`  
**Auditor:** Cloud agent (pre–first office POC test)

---

## Executive summary

| Area | Grade | Status |
|------|-------|--------|
| Architecture | A | Constitution-aligned LAN-first design |
| Hunt (retrieval) | B+ | Strong on fixtures; unproven on real NAS |
| Ingest | B | Arch-firm lanes complete after ingest optimization |
| Einstein (compose) | B | Wired; needs firm API + real corpus |
| Atlas (cards) | C+ | Code exists; not populated per project |
| POC proof | D | No real index, no 100-Q benchmark, no staff trial |
| Security (LAN) | C | OK for dev; hardened before firm-wide |
| Ops/tooling | A- | validate, poc-test, field collector, launcher |

**Verdict:** Engineering ~60% complete for Gateway 1. Product proof ~10%.  
**Blocker:** Index 3–5 NAS projects and run `poc-test.bat` before API spend.

---

## Gateway 1 exit checklist

| Requirement | Status |
|-------------|--------|
| 3–5 projects indexed on LAN | ❌ |
| Hunt: rerank + version-aware + evidence pack | ✅ built |
| Einstein: enterprise API + fallback | ⚠️ built, untested |
| Atlas: cards + cross-project | ⚠️ partial |
| 100-Q benchmark gates | ❌ fixture-only |
| ≤ S$1,000 spend logged | ❌ |
| 3 staff would use it | ❌ |
| Sponsor demo + memo | ❌ |

---

## Security findings

| Finding | Severity | Pre-test action |
|---------|----------|-----------------|
| No FastAPI auth on `/api/*` | High (LAN) | Document; restrict subnet at firewall |
| Admin `admin/admin` hardcoded | High | **Fixed:** env `TIGA_ADMIN_USER` / `TIGA_ADMIN_PASSWORD` |
| CORS `*` | Medium | Accept for POC LAN |
| `/api/processes/kill` unauthenticated | High | Document; disable if not needed |
| API keys in env only | Good | No change |

---

## Code / test health (at audit)

- **166/166 tests passing** (1 skipped)
- Untested critical paths: reranker, project_card, router, scheduler

---

## Ingest coverage (post-optimization)

| Type | Handling |
|------|----------|
| PDF, Office, CSV, email | Full text + embed |
| CAD/BIM, images, video, zip | Metadata / path |
| Scanned PDF (no text) | Path fallback (`EXTRACT_EMPTY_FALLBACK`) |
| Duplicate files (same fingerprint) | **Fixed:** skip at discover |
| OCR | Off by default (selective pilot only) |

---

## Pre–first-test fixes applied (this commit)

1. **Audit log** — this document
2. **Fix 2 failing tests** — PDF split test without fpdf; sleep test batch_size alignment
3. **`reranker_enabled: true`** — align `config.py` default with POC template
4. **Content-hash dedupe at discover** — skip duplicate fingerprints (`dedupe.enabled`)
5. **Admin auth via env** — `TIGA_ADMIN_USER`, `TIGA_ADMIN_PASSWORD`
6. **Fingerprint index** — faster duplicate lookup on large corpora

---

## Recommended sequence (office)

```bat
setup.bat
:: Edit tiga_work/config.yaml — real index_roots (3–5 projects)
poc-test.bat
:: Send export zip back for Hunt refinement
python tiga.py collect export
```

Do **not** enable firm-wide OCR or index all 40 TB for Gateway 1.

---

## Next engineering (after first office export)

- Near-duplicate / revision linking (beyond exact hash)
- Reranker unit tests
- 100-Q benchmark merge from office exports
- Answer correctness metrics in eval (not path-only)
- API auth or subnet restriction for production LAN

---

## Change log

| Date | Entry |
|------|-------|
| 2026-08-23 | Package cleanup: START-HERE.bat one-click install + configure + POC test |
| 2026-08-23 | Initial audit + pre-test hardening (dedupe, admin env, test fixes, rerank default) |
