# TIGA Board outcome — Hunt verification

Date: 2026-09-15 (Asia/Singapore)

## Build update

- Local checked-out build: `4827da880b41` on `codex/hunt-atlas-board-update`, with pre-existing dirty changes; it was **not** replaced or reset.
- Remote master after the requested tickets: `026dae3f3b05` (`#13`, with `#12` and `#11` directly behind it).
- Update record: `4827da880b41` → `4827da880b41` locally (no pull applied); `origin/master` was fetched as `026dae3f3b05`.
- Reason: current HEAD and `origin/master` diverge `4` local-only / `3` remote-only commits. A fast-forward is impossible and merging would risk the unrelated dirty work. This is therefore **not a latest-master acceptance test**.
- No GitHub issue was closed. `tiga_work` was not wiped.

## Restart and Projects verification

- Restarted local Hunt on `127.0.0.1:7862` from the current checkout.
- Browser Search passed for `NUS BIZ3 presentation ppt`: 23 results, headed by `07 Submission/Stage 2/PPT/NUS BIZ3 Tender Presentation Slides.pptx`.
- `/projects` listed 5 projects. Four displayed indexed-file counts: NUS BIZ3 `5,419`, SUSS `3,065`, SportSG `2,445`, NParks `1,599`; HICA displayed `0`.
- The pipeline API simultaneously returned `configured_projects=5` and `indexed_projects=0`, including zero per-project files. This is a real UI/API consistency bug, not a zero-index conclusion.

## NUS BIZ3 Atlas flow

- Auto-draft loaded 20 evidence candidates and showed 2 existing trusted pins plus 2/2 cited facts.
- Pins persisted: `3 Tender Briefing_Biz3.pdf` (overview deck) and `20260813 NUS BIZ3 Stage 2 Report - Plans PPT.pdf` (design report).
- Cited facts persisted: proposed new NUS Business School building (BIZ3), and Design competition Stage 2; both cite `2 Project Design Brief_Biz3.pdf`.
- Ask gate was open. The query “What project development and brief stage are supported by the cited facts?” returned the two pins and two cited facts only, with the explicit local-Ask-stub warning that `TIGA_LLM_API_KEY` is needed for enterprise composition.
- Reload restored the 2 pins, 2 cited facts, and open evidence gate. Overlay persistence passed.

## Three good queries — rerun on the restarted local build

1. `NUS BIZ3 presentation ppt` — returned tender-presentation PPTX evidence; top relevant deck was present. CLI: 3 results in 11.3s.
2. `NUS BIZ3 CAD` — returned project-scoped `NUS BIZ3_L3.dwg` and `NUS BIZ3_L1.dwg` paths. CLI: 3 results in 6.7s.
3. `NUS BIZ3 design brief` — returned `3_BIZ3 Design Competition Brief.pdf` with page-1 citation and the proposed-building statement. CLI: 3 results in 7.3s.

## Three bad queries — previously captured live outcomes, not rerun in this pass

1. `NUS BIZ3 xyzzy_nonexistent_987654.dwg` — broadens to generic CAD hits rather than a precise no-match.
2. `NUS BIZ3 approved final cost 999 trillion` — returns generic quantity-surveyor agreement evidence despite no support for the requested number.
3. `NUS BIZ3 moon base approval` — returns ordinary approval-related excerpts despite no support for a moon-base claim.

## Atlas wins and bugs

### Wins

- Auto-draft evidence is grouped and source-pathed; existing pins and cited facts survived a hard browser reload.
- Evidence-gated Ask exposed its source set and did not fabricate an answer in stub mode.

### Bugs / limits

- Pipeline API reports no indexed projects while Projects UI displays four positive file counts.
- Atlas labels the page `Published · 100/100` although stage, typology, client, and location remain “Needs curation.”
- The full composition backend is unavailable without `TIGA_LLM_API_KEY`; Ask is a cited-source stub.
- Search result duplication remains visible for tender-presentation variants.

## Remaining blocker

Latest-master validation requires a clean, non-divergent checkout or a user-approved integration of the dirty review branch. No destructive cleanup, reset, or forced merge was performed.
