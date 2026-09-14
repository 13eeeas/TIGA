"""
Isolated temporary-repository tests for the WOHA-safe updater (issue #9).

No office corpus, no network remotes: each test builds a throwaway origin +
clone under tmp_path. Success and failure paths must not reset, merge, or
discard local source or runtime data.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

from tools.safe_update import (
    FORBIDDEN_GIT_VERBS,
    Git,
    HealthProbe,
    UpdateError,
    classify_worktree,
    configured_health_url,
    format_report,
    is_runtime_path,
    read_configured_server_port,
    rollback_notes,
    run_update,
    validate_git_argv,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
UPDATE_SH = REPO_ROOT / "update.sh"
UPDATE_BAT = REPO_ROOT / "update.bat"
SAFE_UPDATE = REPO_ROOT / "tools" / "safe_update.py"


# ---------------------------------------------------------------------------
# Git fixture helpers
# ---------------------------------------------------------------------------


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("GIT_AUTHOR_NAME", "TIGA Test")
    env.setdefault("GIT_AUTHOR_EMAIL", "tiga-test@example.invalid")
    env.setdefault("GIT_COMMITTER_NAME", "TIGA Test")
    env.setdefault("GIT_COMMITTER_EMAIL", "tiga-test@example.invalid")
    proc = subprocess.run(
        [
            "git",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "init.defaultBranch=master",
            *args,
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed in {cwd}:\n{proc.stdout}\n{proc.stderr}"
        )
    return proc


def _sha(cwd: Path) -> str:
    return _git(cwd, "rev-parse", "HEAD").stdout.strip()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _init_origin(tmp_path: Path) -> Path:
    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "-b", "master")
    _write(
        seed / ".gitignore",
        "tiga_work/db/\ntiga_work/vectors/\ntiga_work/logs/\n"
        "tiga_work/atlas/\ntiga_work/config.yaml\n.venv/\n.env\n",
    )
    _write(seed / "app.py", "print('v1')\n")
    _write(seed / "requirements.txt", "# synthetic\n")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "seed v1")
    _git(tmp_path, "init", "--bare", str(origin))
    _git(seed, "remote", "add", "origin", str(origin))
    _git(seed, "push", "-u", "origin", "master")
    return origin


def _clone(origin: Path, dest: Path) -> Path:
    _git(origin.parent, "clone", str(origin), str(dest))
    _git(dest, "checkout", "master")
    return dest


def _push_version(origin: Path, tmp_path: Path, name: str, body: str) -> str:
    work = tmp_path / f"push-{name}"
    if work.exists():
        # unique-ish
        work = tmp_path / f"push-{name}-{body[:8]}"
    _git(tmp_path, "clone", str(origin), str(work))
    _git(work, "checkout", "master")
    _write(work / "app.py", body)
    extra = work / "docs" / f"{name}.md"
    _write(extra, f"# {name}\n")
    _git(work, "add", "app.py", str(extra.relative_to(work)))
    _git(work, "commit", "-m", name)
    _git(work, "push", "origin", "master")
    return _sha(work)


def _runtime_tree(repo: Path) -> dict[str, str]:
    """Office runtime files that must survive any update outcome."""
    files = {
        "tiga_work/config.yaml": textwrap.dedent(
            """\
            index_roots:
              - /office/projects
            server:
              host: 0.0.0.0
              port: 9123
            """
        ),
        "tiga_work/db/tiga.db": "SQLITE-FAKE",
        "tiga_work/vectors/store.bin": "LANCEDB-FAKE",
        "tiga_work/atlas/NUS.overlay.json": '{"pins": ["keep-me"]}',
    }
    for rel, content in files.items():
        _write(repo / rel, content)
    return files


def _assert_runtime_intact(repo: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = repo / rel
        assert path.is_file(), f"runtime path vanished: {rel}"
        assert path.read_text(encoding="utf-8") == content, f"runtime path mutated: {rel}"


def _assert_no_destructive_history(repo: Path, before: str) -> None:
    """HEAD is still before, and there is no merge commit after it."""
    assert _sha(repo) == before
    merges = _git(repo, "rev-list", "--merges", f"{before}..HEAD").stdout.strip()
    assert merges == ""


# ---------------------------------------------------------------------------
# Unit: policy helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel, expected",
    [
        ("tiga_work/config.yaml", True),
        ("tiga_work/db/tiga.db", True),
        ("tiga_work/vectors/x", True),
        ("tiga_work/atlas/NUS.overlay.json", True),
        (".venv/bin/python", True),
        (".env", True),
        ("core/query.py", False),
        ("update.sh", False),
        ("scratch.local.py", False),
    ],
)
def test_runtime_path_classification(rel: str, expected: bool) -> None:
    assert is_runtime_path(rel) is expected


def test_read_configured_port_does_not_default_to_7860(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    assert read_configured_server_port(cfg) is None
    _write(cfg, "index_roots: []\n")
    assert read_configured_server_port(cfg) is None
    _write(
        cfg,
        "server:\n  host: 0.0.0.0\n  port: 9123\nollama:\n  timeout_seconds: 30\n",
    )
    assert read_configured_server_port(cfg) == 9123


def test_configured_health_url_uses_config_port(tmp_path: Path) -> None:
    _write(
        tmp_path / "tiga_work" / "config.yaml",
        "server:\n  port: 9123\n",
    )
    assert configured_health_url(tmp_path) == "http://127.0.0.1:9123/health"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert configured_health_url(empty) is None


def test_git_wrapper_refuses_destructive_verbs(tmp_path: Path) -> None:
    git = Git(tmp_path)
    for verb in sorted(FORBIDDEN_GIT_VERBS):
        with pytest.raises(UpdateError, match="refusing git"):
            validate_git_argv([verb])
        with pytest.raises(UpdateError, match="refusing git"):
            git.run([verb, "HEAD"])
    with pytest.raises(UpdateError, match="--ff-only"):
        validate_git_argv(["merge", "origin/master"])


def test_rollback_notes_never_recommend_reset() -> None:
    notes = "\n".join(rollback_notes("abc123", "http://127.0.0.1:9123/health"))
    assert "git merge --ff-only abc123" in notes
    assert "9123" in notes
    assert "reset --hard" in notes  # mentioned as DO NOT
    assert "Do NOT" in notes
    assert "tiga_work" in notes


# ---------------------------------------------------------------------------
# Isolated repo: success
# ---------------------------------------------------------------------------


def test_clean_behind_fast_forwards_to_expected_sha(tmp_path: Path) -> None:
    origin = _init_origin(tmp_path)
    office = _clone(origin, tmp_path / "office")
    runtime = _runtime_tree(office)
    old = _sha(office)
    incoming = _push_version(origin, tmp_path, "v2", "print('v2')\n")

    report = run_update(
        office,
        skip_deps=True,
        skip_health=True,
        pin_sha=incoming,
    )
    text = format_report(report)

    assert report.ok is True
    assert report.claim_success() is True
    assert report.branch == "master"
    assert report.old_sha == old
    assert report.new_sha == incoming
    assert _sha(office) == incoming
    assert (office / "app.py").read_text(encoding="utf-8") == "print('v2')\n"
    _assert_runtime_intact(office, runtime)
    assert "SUCCESS" in text
    assert old in text and incoming in text
    assert "Rollback notes:" in text
    assert "FAILED" not in text.split("Update result:")[1].splitlines()[0]


def test_already_up_to_date_is_success(tmp_path: Path) -> None:
    origin = _init_origin(tmp_path)
    office = _clone(origin, tmp_path / "office")
    sha = _sha(office)
    report = run_update(office, skip_deps=True, skip_health=True)
    assert report.ok is True
    assert report.old_sha == report.new_sha == sha
    assert "Already at target SHA" in " ".join(report.messages)


def test_sha_pin_deploys_that_commit_not_later_tip(tmp_path: Path) -> None:
    origin = _init_origin(tmp_path)
    office = _clone(origin, tmp_path / "office")
    mid = _push_version(origin, tmp_path, "mid", "print('mid')\n")
    _push_version(origin, tmp_path, "late", "print('late')\n")

    report = run_update(office, skip_deps=True, skip_health=True, pin_sha=mid)
    assert report.ok is True
    assert report.new_sha == mid
    assert _sha(office) == mid
    assert (office / "app.py").read_text(encoding="utf-8") == "print('mid')\n"


# ---------------------------------------------------------------------------
# Isolated repo: refuse without rewriting
# ---------------------------------------------------------------------------


def test_divergent_branch_fails_without_merge_or_reset(tmp_path: Path) -> None:
    origin = _init_origin(tmp_path)
    office = _clone(origin, tmp_path / "office")
    runtime = _runtime_tree(office)
    _push_version(origin, tmp_path, "remote-only", "print('remote')\n")

    _write(office / "app.py", "print('office-local')\n")
    _git(office, "add", "app.py")
    _git(office, "commit", "-m", "office divergence")
    before = _sha(office)
    office_app = (office / "app.py").read_text(encoding="utf-8")

    report = run_update(office, skip_deps=True, skip_health=True)
    text = format_report(report)

    assert report.ok is False
    assert report.claim_success() is False
    assert "diverged" in " ".join(report.errors).lower()
    assert "SUCCESS" not in text.split("Update result:")[1]
    assert _sha(office) == before
    assert (office / "app.py").read_text(encoding="utf-8") == office_app
    _assert_no_destructive_history(office, before)
    _assert_runtime_intact(office, runtime)
    log = _git(office, "log", "--oneline", "-5").stdout
    assert "office divergence" in log


def test_tracked_edits_survive_and_block_update(tmp_path: Path) -> None:
    origin = _init_origin(tmp_path)
    office = _clone(origin, tmp_path / "office")
    _push_version(origin, tmp_path, "v2", "print('v2')\n")
    before = _sha(office)
    _write(office / "app.py", "print('office-wip')\n")

    report = run_update(office, skip_deps=True, skip_health=True)
    assert report.ok is False
    assert any("Tracked source" in e for e in report.errors)
    assert _sha(office) == before
    assert (office / "app.py").read_text(encoding="utf-8") == "print('office-wip')\n"


def test_untracked_source_survives_when_not_in_incoming(tmp_path: Path) -> None:
    origin = _init_origin(tmp_path)
    office = _clone(origin, tmp_path / "office")
    _push_version(origin, tmp_path, "v2", "print('v2')\n")
    scratch = office / "scratch.local.py"
    _write(scratch, "keep\n")

    report = run_update(office, skip_deps=True, skip_health=True)
    assert report.ok is True
    assert scratch.read_text(encoding="utf-8") == "keep\n"
    wt = classify_worktree(Git(office))
    assert "scratch.local.py" in wt.untracked_source


def test_untracked_source_blocks_when_incoming_would_overwrite(tmp_path: Path) -> None:
    origin = _init_origin(tmp_path)
    office = _clone(origin, tmp_path / "office")
    _write(office / "docs" / "v2.md", "office untracked\n")
    incoming = _push_version(origin, tmp_path, "v2", "print('v2')\n")
    before = _sha(office)

    report = run_update(office, skip_deps=True, skip_health=True)
    assert report.ok is False
    assert any("Untracked source would be overwritten" in e for e in report.errors)
    assert _sha(office) == before
    assert (office / "docs" / "v2.md").read_text(encoding="utf-8") == "office untracked\n"
    assert incoming != before


def test_runtime_data_intact_across_successful_update(tmp_path: Path) -> None:
    origin = _init_origin(tmp_path)
    office = _clone(origin, tmp_path / "office")
    runtime = _runtime_tree(office)
    _push_version(origin, tmp_path, "v2", "print('v2')\n")

    report = run_update(office, skip_deps=True, skip_health=True)
    assert report.ok is True
    _assert_runtime_intact(office, runtime)
    # ignored runtime is classified, not treated as blocking source
    wt = classify_worktree(Git(office))
    assert not any(p.startswith("tiga_work/") for p in wt.untracked_source)
    assert any("tiga_work" in p for p in wt.ignored_runtime)


# ---------------------------------------------------------------------------
# Fail states: fetch / install / health — never claim success
# ---------------------------------------------------------------------------


def test_fetch_failure_reported_worktree_unchanged(tmp_path: Path) -> None:
    origin = _init_origin(tmp_path)
    office = _clone(origin, tmp_path / "office")
    runtime = _runtime_tree(office)
    before = _sha(office)
    _git(office, "remote", "set-url", "origin", str(tmp_path / "missing.git"))

    report = run_update(office, skip_deps=True, skip_health=True)
    text = format_report(report)
    assert report.ok is False
    assert report.new_sha == before
    assert any("Fetch" in e for e in report.errors)
    assert "SUCCESS" not in text.split("Update result:")[1]
    assert _sha(office) == before
    _assert_runtime_intact(office, runtime)


def test_install_failure_does_not_claim_success(tmp_path: Path) -> None:
    origin = _init_origin(tmp_path)
    office = _clone(origin, tmp_path / "office")
    runtime = _runtime_tree(office)
    old = _sha(office)
    incoming = _push_version(origin, tmp_path, "v2", "print('v2')\n")

    def boom(_repo: Path) -> tuple[int, str]:
        return 1, "pip exploded (synthetic)"

    report = run_update(
        office,
        skip_health=True,
        install_fn=boom,
    )
    text = format_report(report)
    assert report.ok is False
    assert report.new_sha == incoming
    assert _sha(office) == incoming  # code moved; we do not silent-reset
    assert any("Dependency install failed" in e for e in report.errors)
    assert "pip exploded" in text
    assert "SUCCESS" not in text.split("Update result:")[1]
    assert old in "\n".join(report.rollback)
    _assert_runtime_intact(office, runtime)


def test_health_reads_configured_port_not_default(tmp_path: Path) -> None:
    origin = _init_origin(tmp_path)
    office = _clone(origin, tmp_path / "office")
    _runtime_tree(office)
    seen: list[str] = []

    def probe(url: str) -> HealthProbe:
        seen.append(url)
        return HealthProbe(url=url, ok=True, status=200, detail='{"status":"ok"}')

    report = run_update(
        office,
        skip_deps=True,
        health_probe_fn=probe,
    )
    assert report.ok is True
    assert report.health_url == "http://127.0.0.1:9123/health"
    assert seen == ["http://127.0.0.1:9123/health"]
    assert "7860" not in (report.health_url or "")


def test_missing_port_does_not_assume_7860_or_fail(tmp_path: Path) -> None:
    origin = _init_origin(tmp_path)
    office = _clone(origin, tmp_path / "office")
    # runtime tree without server.port
    _write(office / "tiga_work" / "config.yaml", "index_roots: []\n")

    def must_not_probe(_url: str) -> HealthProbe:
        raise AssertionError("must not probe a default port")

    report = run_update(
        office,
        skip_deps=True,
        health_probe_fn=must_not_probe,
    )
    assert report.ok is True
    assert report.health_url is None
    assert any("Not assuming" in w or "7860" in w for w in report.warnings)


def test_health_failure_does_not_claim_success(tmp_path: Path) -> None:
    origin = _init_origin(tmp_path)
    office = _clone(origin, tmp_path / "office")
    _runtime_tree(office)

    def unhealthy(url: str) -> HealthProbe:
        return HealthProbe(url=url, ok=False, status=503, detail="nope")

    report = run_update(office, skip_deps=True, health_probe_fn=unhealthy)
    text = format_report(report)
    assert report.ok is False
    assert any("Health check failed" in e for e in report.errors)
    assert "SUCCESS" not in text.split("Update result:")[1]


def test_not_listening_is_warning_unless_required(tmp_path: Path) -> None:
    origin = _init_origin(tmp_path)
    office = _clone(origin, tmp_path / "office")
    _runtime_tree(office)

    def down(url: str) -> HealthProbe:
        return HealthProbe(url=url, ok=False, not_listening=True, detail="refused")

    warn = run_update(office, skip_deps=True, health_probe_fn=down)
    assert warn.ok is True
    assert any("Nothing listening" in w for w in warn.warnings)

    fail = run_update(
        office,
        skip_deps=True,
        require_health=True,
        health_probe_fn=down,
    )
    assert fail.ok is False


def test_dry_run_does_not_move_head(tmp_path: Path) -> None:
    origin = _init_origin(tmp_path)
    office = _clone(origin, tmp_path / "office")
    before = _sha(office)
    _push_version(origin, tmp_path, "v2", "print('v2')\n")

    report = run_update(office, skip_deps=True, skip_health=True, dry_run=True)
    assert report.ok is True
    assert _sha(office) == before
    assert report.new_sha == before
    assert any("Dry run" in m for m in report.messages)


def test_unknown_pin_fails(tmp_path: Path) -> None:
    origin = _init_origin(tmp_path)
    office = _clone(origin, tmp_path / "office")
    before = _sha(office)
    report = run_update(
        office,
        skip_deps=True,
        skip_health=True,
        pin_sha="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
    )
    assert report.ok is False
    assert _sha(office) == before
    assert any("Pinned SHA" in e for e in report.errors)


# ---------------------------------------------------------------------------
# Wrappers stay non-destructive (source inspection + smoke)
# ---------------------------------------------------------------------------


def test_wrappers_delegate_and_omit_destructive_git() -> None:
    sh = UPDATE_SH.read_text(encoding="utf-8")
    bat = UPDATE_BAT.read_text(encoding="utf-8")
    py = SAFE_UPDATE.read_text(encoding="utf-8")
    for src, label in ((sh, "update.sh"), (bat, "update.bat")):
        assert "safe_update.py" in src, label
        for banned in (
            "git pull",
            "git reset",
            "git stash",
            "git clean",
            "git merge",
            "git checkout",
            "git rebase",
        ):
            assert banned not in src, f"{label} contains {banned}"
    assert "FORBIDDEN_GIT_VERBS" in py
    assert "--ff-only" in py


def test_update_sh_smoke_already_current(tmp_path: Path) -> None:
    origin = _init_origin(tmp_path)
    office = _clone(origin, tmp_path / "office")
    dest = office / "tools"
    dest.mkdir()
    dest.joinpath("safe_update.py").write_text(
        SAFE_UPDATE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    script = office / "update.sh"
    script.write_text(UPDATE_SH.read_text(encoding="utf-8"), encoding="utf-8")
    script.chmod(0o755)

    proc = subprocess.run(
        ["bash", str(script), "--skip-deps", "--skip-health"],
        cwd=office,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SUCCESS" in proc.stdout
    assert "Update Complete!" not in proc.stdout
    assert "Rollback notes:" in proc.stdout
