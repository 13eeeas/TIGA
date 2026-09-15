# TIGA Board outcome — Hunt verification

Date: 2026-09-15 (Asia/Singapore)

## UI follow-up delivery

- Hunt now requests 20 results per page, advances pagination by 20, and hides the hero heading after search while moving the search form upward.
- Atlas displays cited facts or a labelled source excerpt above archive statistics. This is an initial sourced overview, not yet a full synthesized wiki article.
- Atlas applies Hunt's Lora/Source Serif typography, coral accents, rounded cards and soft shadows. The sidebar layout remains different; complete UI parity is not claimed.
- Validation: `pytest tests/test_atlas_wiki.py -q --basetemp=.pytest_tmp_board_ui`: 5 passed. Earlier runtime check returned two overview entries for NUS BIZ3. Final styling has not received a fresh browser visual check.
- Delivery branch: `codex/hunt-atlas-board-update`. Local master diverges from origin/master; this branch is for review, not a completed master integration.

## Result

Live workaround test completed on the existing dirty build; partial acceptance. The observations below supersede the earlier blocked attempt. Latest-master acceptance is still outstanding.

## Successful workaround and live results (2026-09-15)

Started Hunt with `.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 7862` and `PYTHONUTF8=1`. Startup completed and both Search and Projects loaded in the browser.

Build tested: `43c049d661d553463d7232edd570a940336c84e8` plus existing local changes. Elevated `git fetch origin` succeeded, advancing remote tracking from `ccf67bb` to `026dae3` (includes #11/#12/#13). HEAD did not change: local and remote histories diverge by three commits each. This is not an updated-master test.

Projects UI listed five projects; four showed positive indexed-file counts: NUS BIZ3 5,419; SUSS 3,065; SportSG 2,445; NParks 1,599. HICA showed zero. The pipeline endpoint reported `indexed_projects=0`, `configured_projects=5`, contradicting the Projects UI and requiring investigation.

### Three useful queries (compose disabled)

1. `NUS BIZ3 presentation ppt`: semantic retrieval returned the native `07 Submission/Stage 2/PPT/NUS BIZ3 Tender Presentation Slides.pptx`, cited at slide 1. Browser visibly displayed the result. API latency 211 ms in the captured run.
2. `NUS BIZ3 CAD`: file locator returned 30 CAD files, including `20260619 NUS BIZ3 Parking.dwg`. Some hits are reference-project assets inside the project folder; this is not proof of authoritative drawing quality.
3. `NUS BIZ3 design brief`: returned `3_BIZ3 Design Competition Brief.pdf`, with page citations (p18 and p1). Captured repeat latency 541 ms.

### Three bad-query observations

1. `NUS BIZ3 xyzzy_nonexistent_987654.dwg`: incorrectly broadens to 30 generic CAD files instead of a precise no-match.
2. `NUS BIZ3 approved final cost 999 trillion`: returns generic quantity-surveyor agreement excerpts and claims 20 relevant evidence results; no evidence supports the requested number.
3. `NUS BIZ3 moon base approval`: returns ordinary report excerpts about performance-based approval and claims 20 relevant evidence results; no evidence supports a moon base.

These are retrieval/relevance failures. With compose disabled, no generated factual answer was tested for these queries.

### Atlas wins

- NUS BIZ3 auto-draft populated 20 candidates with source paths and real excerpts. Ask was visibly disabled before the first pin.
- Saved two distinct-category pins: tender briefing and 20260813 Stage 2 plans report. Saved two draft facts with citations to the Stage 2 project design brief: proposed new NUS Business School building, and Design Competition Stage 2.
- Browser reload preserved both pins and both cited facts; Ask stayed unlocked.

### Atlas bugs and limits

- Pinning the tender briefing after the design brief silently replaced the first pin because both use the overview category. No replacement warning was shown.
- One pin and zero facts immediately produced `Published · 100/100`, despite uncurated identity properties.
- Ask returned a local stub listing the two pins and two cited facts, with a message to configure `TIGA_LLM_API_KEY`; substantive answer composition did not run.
- No junk candidate was hidden; hide persistence and authority-pin coverage remain untested. Source excerpts support the saved facts, but source files were not independently opened.

### Remaining acceptance

Integrate or separately test latest master, resolve the query and Atlas issues, exercise hide persistence and composed Ask, and reconcile project counts. No issues were closed and no corpus cleanup was applied.

## Earlier blocked attempt (historical)

- Build SHA before update attempt: `43c049d661d553463d7232edd570a940336c84e8` (`master`).
- Local `origin/master` observation: `ccf67bb`; this is not a completed pull.
- `update.bat` was run with the existing dirty checkout preserved. It declined stashing, then failed during fetch: `cannot open '.git/FETCH_HEAD': Permission denied`.
- Working tree remains dirty; `tiga_work` was not wiped.
- Current index status observed: 96,860 total files; 332,953 total chunks; 12,693 INDEXED, 66,984 EXTRACTED, 5,984 DISCOVERED, 11,197 SKIPPED, 2 FAILED.
- Indexed roots include NUS BIZ3 and 283 HICA, but the status shows 72,968 files remaining and indexing at 0/h, so corpus completion is not claimed.

## Required checks not verified

- No old-SHA → new-SHA update report exists because fetch failed.
- Hunt restart and Search confirmation were not successful; ports 7860/7862 were not listening.
- The CLI query `NUS BIZ3 authority submission` produced no result within the 30-second verification window.
- `/projects` auto-draft, pin/cite, Ask, reload persistence, and the requested NUS BIZ3 browser flow were not exercised.
- No 3 good or 3 bad live queries are recorded.

## Atlas bugs/wins

None claimed. The live Projects flow was not exercised.

## Blockers

1. Managed permission denied `.git/FETCH_HEAD` during `update.bat` fetch.
2. Hunt was not listening on the expected ports.
3. Index is incomplete; the single throughput snapshot does not establish a persistent stall and does not itself block testing an indexed project.

## Verification clarification

No Hunt restart was actually attempted. The missing listener does not establish a startup defect. No terminal query error was captured. An elevated retry of the Git permission failure was not attempted. See `docs/TIGA_BOARD_NOTES.md` for the Kit HICA template requirement and remaining verification steps.
