# WOHA update and rollback

GitHub → office (WOHA) deploys go through **`update.bat`** / **`update.sh`**. Those wrappers call `tools/safe_update.py`. They never `git pull`, merge, reset, stash, or clean.

First-time office install is still **`START-HERE.bat`** / **`START-HERE.sh`**. START-HERE does not pull GitHub. After the first clone, use the updater.

## What is preserved

| Path | Role | Updater behaviour |
|------|------|-------------------|
| Tracked source (`core/`, `tiga.py`, …) | Product code | Fast-forward only. Dirty files **block** the update. |
| Untracked source | Local scratch | Survives. Blocks the update if the incoming commit has the same path. |
| `tiga_work/config.yaml` | Office config | Ignored runtime data — left intact. |
| `tiga_work/db/`, `tiga_work/vectors/` | Hunt index | Left intact. |
| `tiga_work/atlas/*.overlay.json` | Atlas wiki pins/facts | Left intact. |
| `.venv/`, `.env` | Local env | Left intact. |

## Fast-forward + SHA pin

```bat
update.bat
```

```bash
bash update.sh
```

Pin a known-good commit (must be a descendant of the current HEAD):

```bat
set TIGA_UPDATE_SHA=<full-or-abbrev-sha>
update.bat
```

```bash
TIGA_UPDATE_SHA=<sha> bash update.sh
# or
bash update.sh --sha <sha>
```

After a successful run the report prints **branch**, **old SHA**, **new SHA**, the **configured health URL**, and rollback notes.

## Health URL

The probe reads `server.port` from `tiga_work/config.yaml` (or `TIGA_WORK_DIR/config.yaml`).

It does **not** assume port `7860`. If `server.port` is missing, the updater reports that the health URL is unknown. An earlier refusal on the default port is **not** proof that Hunt failed to start.

- Default: if nothing is listening on the **configured** URL, that is a warning (restart Hunt). The git step can still succeed.
- After restart, verify with `bash update.sh --require-health` (or `TIGA_UPDATE_REQUIRE_HEALTH=1`).
- If the configured URL responds with an error status, the updater **fails** and does not claim success.

## Failure states (no “Update Complete”)

| Situation | Result | Worktree |
|-----------|--------|----------|
| Fetch failed | FAILED | Unchanged |
| Diverged / local-only commits | FAILED | Unchanged (no merge, no reset) |
| Tracked edits | FAILED | Edits survive |
| Untracked source would be overwritten | FAILED | Files survive |
| `pip install` failed | FAILED | Code may already be at the new SHA — use rollback notes |
| Health error on configured URL | FAILED | Code not reset automatically |
| Clean, behind remote | SUCCESS | Fast-forward to the target SHA |

## Rollback (code only)

The updater never auto-resets. If you must undo a fast-forward:

```bash
git merge --ff-only <old-sha-from-the-report>
# then reinstall deps if requirements.txt changed
.venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip
# restart Hunt (launcher.bat / python tiga.py serve)
# check the configured health URL from the report
```

Do **not** run `git reset --hard`, `git stash`, `git pull`, or `git clean -fd`. Those can discard office source edits and will **not** restore `tiga_work` config, DB, vectors, or Atlas overlays.

ZIP installs (no `.git`) cannot use this updater. Copy new source **beside** `tiga_work`; never replace the work directory.

## Synthetic tests

`pytest tests/test_safe_update.py` builds isolated temporary git remotes (no office corpus, no network). It covers fast-forward success, divergence, dirty/untracked source, runtime-data survival, fetch/install/health failures, and configured-port health URLs.
