"""
tools/safe_update.py — Fast-forward-only GitHub → WOHA updater.

Office wrappers (update.bat / update.sh) call this module. It is stdlib-only
so it still runs when the project venv is missing or broken.

Safety rails (issue #9):
  * fetch, then fast-forward only — never merge, reset, stash, or clean
  * refuse dirty tracked files and untracked source that would be overwritten
  * leave ignored runtime data (config, DB, vectors, Atlas overlays) untouched
  * report branch, old/new SHA, install result, configured health URL, rollback
  * never claim success after fetch / install / health failure
  * health URL is read from config server.port — never assumed to be 7860
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

# git verbs this updater will never invoke (also rejected if a caller tries).
FORBIDDEN_GIT_VERBS = frozenset(
    {
        "reset",
        "stash",
        "clean",
        "rebase",
        "pull",
        "checkout",
        "switch",
        "restore",
        "revert",
        "cherry-pick",
    }
)

# Untracked / ignored paths treated as office runtime data, not source.
RUNTIME_PREFIXES = (
    "tiga_work/",
    ".venv/",
    ".pytest_cache/",
    "__pycache__/",
)
RUNTIME_NAMES = frozenset(
    {
        ".env",
        ".venv",
        "tiga_work",
        ".DS_Store",
        "Thumbs.db",
        "desktop.ini",
    }
)

DEFAULT_REMOTE = "origin"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class UpdateError(Exception):
    """User-facing failure. Does not imply the worktree was rewritten."""

    def __init__(self, message: str, *, rollback_sha: str | None = None) -> None:
        super().__init__(message)
        self.rollback_sha = rollback_sha


@dataclass
class GitResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass
class WorktreeState:
    tracked_dirty: list[str] = field(default_factory=list)
    untracked_source: list[str] = field(default_factory=list)
    ignored_runtime: list[str] = field(default_factory=list)


@dataclass
class HealthProbe:
    url: str
    ok: bool
    not_listening: bool = False
    status: int | None = None
    detail: str = ""


@dataclass
class UpdateReport:
    ok: bool
    branch: str
    old_sha: str
    new_sha: str
    expected_sha: str | None
    health_url: str | None
    messages: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rollback: list[str] = field(default_factory=list)

    def claim_success(self) -> bool:
        return self.ok and not self.errors


# ---------------------------------------------------------------------------
# Git safety wrapper
# ---------------------------------------------------------------------------


def validate_git_argv(args: Sequence[str]) -> None:
    """Refuse destructive git verbs. merge is allowed only with --ff-only."""
    if not args:
        raise UpdateError("refusing empty git command")
    verb = args[0]
    if verb in FORBIDDEN_GIT_VERBS:
        raise UpdateError(
            f"refusing git {verb} — updater never resets, stashes, merges, or discards"
        )
    if verb == "merge" and "--ff-only" not in args:
        raise UpdateError("refusing git merge without --ff-only")


class Git:
    """Run a whitelist-constrained git in one repository."""

    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd

    def run(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        timeout: int = 120,
    ) -> GitResult:
        validate_git_argv(args)
        proc = subprocess.run(
            ["git", *args],
            cwd=self.cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        result = GitResult(
            returncode=proc.returncode,
            stdout=(proc.stdout or "").strip(),
            stderr=(proc.stderr or "").strip(),
        )
        if check and result.returncode != 0:
            detail = result.stderr or result.stdout or f"exit {result.returncode}"
            raise UpdateError(f"git {' '.join(args)} failed: {detail}")
        return result

    def out(self, args: Sequence[str], **kwargs: object) -> str:
        return self.run(args, **kwargs).stdout  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Path / config helpers
# ---------------------------------------------------------------------------


def is_runtime_path(relpath: str) -> bool:
    """True for ignored office runtime data (config, DB, vectors, Atlas, venv)."""
    norm = relpath.replace("\\", "/")
    while norm.startswith("./"):
        norm = norm[2:]
    if not norm or norm in RUNTIME_NAMES:
        return True
    name = Path(norm).name
    if name in RUNTIME_NAMES:
        return True
    if name.endswith(".pyc"):
        return True
    for prefix in RUNTIME_PREFIXES:
        if norm == prefix.rstrip("/") or norm.startswith(prefix):
            return True
        if f"/{prefix}" in f"/{norm}/":
            return True
    return False


def _strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i]
    return line


def read_configured_server_port(config_path: Path) -> int | None:
    """Return server.port from config.yaml, or None.

    Does **not** fall back to 7860. A missing or unreadable port means the
    health URL is unknown — earlier default-port refusals are not startup proof.
    """
    if not config_path.is_file():
        return None
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        return None

    in_server = False
    server_indent: int | None = None
    for raw in text.splitlines():
        body = _strip_yaml_comment(raw).rstrip()
        if not body.strip():
            continue
        indent = len(body) - len(body.lstrip(" "))
        stripped = body.strip()

        if indent == 0:
            in_server = stripped.startswith("server:")
            server_indent = 0 if in_server else None
            # Inline mapping: server: { host: 0.0.0.0, port: 9123 }
            if in_server and "{" in stripped:
                return _port_from_inline(stripped)
            continue

        if in_server:
            if server_indent is not None and indent <= server_indent:
                in_server = False
                continue
            if stripped.startswith("port:"):
                value = stripped.split(":", 1)[1].strip().strip("'\"")
                try:
                    return int(value)
                except ValueError:
                    return None
    return None


def _port_from_inline(line: str) -> int | None:
    after = line.split(":", 1)[1]
    for part in after.replace("{", " ").replace("}", " ").split(","):
        piece = part.strip()
        if piece.startswith("port"):
            _, _, value = piece.partition(":")
            value = value.strip().strip("'\"")
            try:
                return int(value)
            except ValueError:
                return None
    return None


def resolve_work_dir(repo: Path, override: Path | None = None) -> Path:
    if override is not None:
        return override
    env = os.environ.get("TIGA_WORK_DIR")
    if env:
        return Path(env)
    return repo / "tiga_work"


def configured_health_url(repo: Path, work_dir: Path | None = None) -> str | None:
    """Build http://127.0.0.1:{configured_port}/health, or None if port unknown.

    Bind host in config may be 0.0.0.0; probes always use loopback.
    """
    wd = resolve_work_dir(repo, work_dir)
    port = read_configured_server_port(wd / "config.yaml")
    if port is None:
        return None
    return f"http://127.0.0.1:{port}/health"


def rollback_notes(old_sha: str, health_url: str | None) -> list[str]:
    health = health_url or (
        "not configured (no server.port in tiga_work/config.yaml; "
        "do not assume http://127.0.0.1:7860/health)"
    )
    return [
        "Rollback restores *code only*. Configuration, DB, vectors and Atlas",
        "overlays in tiga_work/ are ignored runtime data and were not moved.",
        f"  git merge --ff-only {old_sha}",
        "  then: .venv pip install -r requirements.txt   (if deps changed)",
        "  then: restart Hunt (launcher.bat / python tiga.py serve)",
        f"  health URL: {health}",
        "Do NOT run git reset --hard, git stash, git pull, or git clean -fd.",
        "Those can discard office edits and will not restore tiga_work data.",
    ]


# ---------------------------------------------------------------------------
# Worktree classification
# ---------------------------------------------------------------------------


def classify_worktree(git: Git) -> WorktreeState:
    """Separate tracked edits, untracked source, and ignored runtime data."""
    state = WorktreeState()
    porcelain = git.run(["status", "--porcelain=v1", "--untracked-files=all"])
    for line in porcelain.stdout.splitlines():
        if len(line) < 4:
            continue
        code, path = line[:2], line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        path = path.strip().replace("\\", "/")
        if code == "??":
            if is_runtime_path(path):
                state.ignored_runtime.append(path)
            else:
                state.untracked_source.append(path)
        elif code.strip():
            state.tracked_dirty.append(path)

    ignored = git.run(
        ["status", "--porcelain=v1", "--ignored=traditional", "--untracked-files=normal"],
        check=False,
    )
    if ignored.returncode == 0:
        for line in ignored.stdout.splitlines():
            if line.startswith("!! "):
                path = line[3:].replace("\\", "/")
                if is_runtime_path(path):
                    state.ignored_runtime.append(path)
    # unique, stable
    state.tracked_dirty = sorted(set(state.tracked_dirty))
    state.untracked_source = sorted(set(state.untracked_source))
    state.ignored_runtime = sorted(set(state.ignored_runtime))
    return state


def incoming_paths(git: Git, target: str) -> set[str]:
    listing = git.out(["ls-tree", "-r", "--name-only", target])
    return {p.replace("\\", "/") for p in listing.splitlines() if p}


def overwrite_conflicts(untracked_source: Iterable[str], incoming: set[str]) -> list[str]:
    return sorted(p for p in untracked_source if p in incoming)


# ---------------------------------------------------------------------------
# Deps + health
# ---------------------------------------------------------------------------


def default_venv_python(repo: Path) -> Path | None:
    if os.name == "nt":
        candidate = repo / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = repo / ".venv" / "bin" / "python"
    return candidate if candidate.is_file() else None


def default_install(repo: Path) -> tuple[int, str]:
    """Install requirements.txt with the office venv. Stdlib subprocess only."""
    req = repo / "requirements.txt"
    if not req.is_file():
        return 0, "no requirements.txt — skipped"
    py = default_venv_python(repo)
    if py is None:
        return 0, "no .venv — skipped (run setup.bat / setup.sh first)"
    proc = subprocess.run(
        [
            str(py),
            "-m",
            "pip",
            "install",
            "-r",
            str(req),
            "--disable-pip-version-check",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    log = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        return proc.returncode, log or f"pip exited {proc.returncode}"
    return 0, "pip install -r requirements.txt succeeded"


def default_health_probe(url: str, timeout: float = 3.0) -> HealthProbe:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            body = resp.read().decode("utf-8", errors="replace")[:400]
            ok = 200 <= int(status) < 300
            return HealthProbe(
                url=url,
                ok=ok,
                status=int(status),
                detail=body or f"HTTP {status}",
            )
    except urllib.error.HTTPError as exc:
        return HealthProbe(
            url=url,
            ok=False,
            status=int(exc.code),
            detail=str(exc),
        )
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        return HealthProbe(
            url=url,
            ok=False,
            not_listening=True,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# Core update
# ---------------------------------------------------------------------------


def run_update(
    repo: Path,
    *,
    remote: str = DEFAULT_REMOTE,
    pin_sha: str | None = None,
    skip_deps: bool = False,
    skip_health: bool = False,
    require_health: bool = False,
    dry_run: bool = False,
    work_dir: Path | None = None,
    install_fn: Callable[[Path], tuple[int, str]] | None = None,
    health_probe_fn: Callable[[str], HealthProbe] | None = None,
    do_fetch: bool = True,
) -> UpdateReport:
    """Fetch and fast-forward one clone. Isolated-temp-repo friendly."""
    git = Git(repo)
    messages: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    try:
        git.run(["rev-parse", "--git-dir"])
    except UpdateError as exc:
        raise UpdateError(
            "Not a git repository. ZIP installs cannot use this updater — "
            "clone with git, or copy new source beside tiga_work without replacing it."
        ) from exc

    branch = git.out(["branch", "--show-current"])
    if not branch:
        raise UpdateError(
            "HEAD is detached. Checkout the office branch (usually master) and retry."
        )

    old_sha = git.out(["rev-parse", "HEAD"])
    health_url = configured_health_url(repo, work_dir)
    notes = rollback_notes(old_sha, health_url)

    def fail(*why: str, new_sha: str | None = None) -> UpdateReport:
        errors.extend(why)
        return UpdateReport(
            ok=False,
            branch=branch,
            old_sha=old_sha,
            new_sha=new_sha or old_sha,
            expected_sha=pin_sha,
            health_url=health_url,
            messages=messages,
            errors=errors,
            warnings=warnings,
            rollback=notes,
        )

    messages.append(f"Branch: {branch}")
    messages.append(f"Current SHA: {old_sha}")
    if health_url:
        messages.append(f"Configured health URL: {health_url}")
    else:
        warnings.append(
            "No server.port in tiga_work/config.yaml — health URL unknown. "
            "Not assuming http://127.0.0.1:7860/health. An earlier default-port "
            "refusal is not a confirmed Hunt startup defect."
        )

    wt = classify_worktree(git)
    if wt.ignored_runtime:
        messages.append(
            f"Ignored runtime data left intact ({len(wt.ignored_runtime)} path(s)): "
            + ", ".join(wt.ignored_runtime[:8])
            + ("…" if len(wt.ignored_runtime) > 8 else "")
        )
    if wt.tracked_dirty:
        return fail(
            "Tracked source edits present — refusing to update so local work survives.",
            "Commit, move, or restore these files yourself. This updater will not "
            "stash, reset, or merge them:",
            *wt.tracked_dirty,
        )
    if wt.untracked_source:
        messages.append(
            "Untracked source (will survive; blocks update if incoming tree has the same path): "
            + ", ".join(wt.untracked_source[:8])
        )

    if do_fetch:
        messages.append(f"Fetching {remote}…")
        fetched = git.run(["fetch", "--prune", remote], check=False, timeout=180)
        if fetched.returncode != 0:
            return fail(
                f"Fetch from {remote} failed. Check network/VPN. Worktree unchanged.",
                fetched.stderr or fetched.stdout or f"exit {fetched.returncode}",
            )

    remote_ref = f"{remote}/{branch}"
    remote_sha_res = git.run(["rev-parse", "--verify", remote_ref], check=False)
    if remote_sha_res.returncode != 0:
        return fail(
            f"No remote-tracking ref {remote_ref} after fetch. "
            f"Set upstream: git branch --set-upstream-to={remote}/{branch}",
            remote_sha_res.stderr or remote_sha_res.stdout,
        )
    remote_sha = remote_sha_res.stdout

    target = remote_sha
    if pin_sha:
        resolved = git.run(["rev-parse", "--verify", f"{pin_sha}^{{commit}}"], check=False)
        if resolved.returncode != 0:
            return fail(
                f"Pinned SHA {pin_sha!r} is not a commit after fetch. "
                "Check TIGA_UPDATE_SHA / --sha."
            )
        target = resolved.stdout
        messages.append(f"SHA pin: {target}")

    messages.append(f"Target SHA: {target}")

    ancestor = git.run(["merge-base", "--is-ancestor", "HEAD", target], check=False)
    if ancestor.returncode != 0:
        return fail(
            f"Branch {branch} has diverged from {remote_ref} (or the pinned SHA). "
            "Refusing merge and reset so office commits and source survive.",
            f"Local HEAD:  {old_sha}",
            f"Update tip:  {target}",
            "Move local commits aside (new branch) or have a developer rebase, "
            "then rerun. Do not git reset --hard.",
        )

    conflicts = overwrite_conflicts(wt.untracked_source, incoming_paths(git, target))
    if conflicts:
        return fail(
            "Untracked source would be overwritten by the incoming tree. "
            "Move or commit these files first:",
            *conflicts,
        )

    if old_sha == target:
        messages.append(f"Already at target SHA {old_sha} — no fast-forward needed.")
        new_sha = old_sha
    elif dry_run:
        messages.append(f"Dry run: would fast-forward {old_sha} → {target}")
        return UpdateReport(
            ok=True,
            branch=branch,
            old_sha=old_sha,
            new_sha=old_sha,
            expected_sha=pin_sha,
            health_url=health_url,
            messages=messages,
            errors=errors,
            warnings=warnings,
            rollback=notes,
        )
    else:
        messages.append(f"Fast-forward {old_sha} → {target}")
        merged = git.run(["merge", "--ff-only", target], check=False)
        if merged.returncode != 0:
            return fail(
                "git merge --ff-only failed. Worktree should be unchanged.",
                merged.stderr or merged.stdout,
            )
        new_sha = git.out(["rev-parse", "HEAD"])
        if new_sha != target:
            return fail(
                f"HEAD {new_sha} is not the expected target {target} after fast-forward.",
                new_sha=new_sha,
            )
        messages.append(f"HEAD is now {new_sha}")

    if pin_sha and new_sha != target:
        return fail(
            f"SHA pin not reached: HEAD={new_sha} expected={target}",
            new_sha=new_sha,
        )

    # Re-check runtime files still present if they existed — updater never deletes them.
    if not skip_deps:
        installer = install_fn or default_install
        code, log = installer(repo)
        if code != 0:
            warnings.append(log)
            return fail(
                "Dependency install failed. Git fast-forward already happened; "
                "code is at the new SHA. Do not claim this deploy succeeded.",
                log,
                new_sha=new_sha,
            )
        messages.append(log)
    else:
        messages.append("Dependency install skipped.")

    if not skip_health:
        if health_url is None:
            warnings.append(
                "Skipping health probe — no configured port. "
                "Not treating a default-port miss as a startup defect."
            )
        else:
            probe = (health_probe_fn or default_health_probe)(health_url)
            if probe.ok:
                messages.append(
                    f"Health OK at configured URL {probe.url} (HTTP {probe.status})."
                )
            elif probe.not_listening:
                msg = (
                    f"Nothing listening at configured health URL {probe.url}. "
                    "Restart Hunt after this update. A miss on default port 7860 "
                    "is not a confirmed startup defect if this install uses another port."
                )
                if require_health:
                    return fail(msg, probe.detail, new_sha=new_sha)
                warnings.append(msg)
            else:
                return fail(
                    f"Health check failed at configured URL {probe.url} "
                    f"(HTTP {probe.status}). Deploy is not successful.",
                    probe.detail,
                    new_sha=new_sha,
                )
    else:
        messages.append("Health probe skipped.")

    return UpdateReport(
        ok=True,
        branch=branch,
        old_sha=old_sha,
        new_sha=new_sha,
        expected_sha=pin_sha,
        health_url=health_url,
        messages=messages,
        errors=errors,
        warnings=warnings,
        rollback=notes,
    )


def format_report(report: UpdateReport) -> str:
    lines = [
        "",
        "============================================================",
        "  TIGA Hunt — Safe update from GitHub",
        "============================================================",
        "",
    ]
    for msg in report.messages:
        lines.append(f"[INFO] {msg}")
    for warn in report.warnings:
        lines.append(f"[WARN] {warn}")
    for err in report.errors:
        lines.append(f"[ERROR] {err}")
    lines += [
        "",
        "============================================================",
        f"  Update result: {'SUCCESS' if report.claim_success() else 'FAILED'}",
        f"  Branch:  {report.branch or '(unknown)'}",
        f"  Old SHA: {report.old_sha or '(unknown)'}",
        f"  New SHA: {report.new_sha or '(unknown)'}",
    ]
    if report.expected_sha:
        lines.append(f"  Pin SHA: {report.expected_sha}")
    lines.append(
        f"  Health:  {report.health_url or '(not configured — not assuming :7860)'}"
    )
    lines.append("============================================================")
    lines.append("")
    lines.append("Rollback notes:")
    lines.extend(f"  {row}" for row in report.rollback)
    lines.append("")
    if report.claim_success():
        lines.append("Restart Hunt to load the new code (launcher.bat / python tiga.py serve).")
    else:
        lines.append(
            "Deployment did not succeed. Office install was not reset. See errors above."
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fast-forward-only TIGA update (never reset/merge/stash).",
    )
    p.add_argument(
        "--sha",
        default=os.environ.get("TIGA_UPDATE_SHA") or None,
        help="Pin a commit (or set TIGA_UPDATE_SHA). Must be a fast-forward from HEAD.",
    )
    p.add_argument(
        "--remote",
        default=os.environ.get("TIGA_UPDATE_REMOTE") or DEFAULT_REMOTE,
        help="Git remote name (default: origin).",
    )
    p.add_argument(
        "--skip-deps",
        action="store_true",
        default=_env_flag("TIGA_UPDATE_SKIP_DEPS"),
        help="Do not run pip install.",
    )
    p.add_argument(
        "--skip-health",
        action="store_true",
        default=_env_flag("TIGA_UPDATE_SKIP_HEALTH"),
        help="Do not probe the configured health URL.",
    )
    p.add_argument(
        "--require-health",
        action="store_true",
        default=_env_flag("TIGA_UPDATE_REQUIRE_HEALTH"),
        help="Fail if the configured health URL is not serving (use after restart).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and validate, but do not fast-forward.",
    )
    p.add_argument(
        "--repo",
        default=None,
        help="Repository root (default: parent of tools/).",
    )
    return p.parse_args(argv)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo = Path(args.repo).resolve() if args.repo else REPO_ROOT
    try:
        report = run_update(
            repo,
            remote=args.remote,
            pin_sha=args.sha,
            skip_deps=args.skip_deps,
            skip_health=args.skip_health,
            require_health=args.require_health,
            dry_run=args.dry_run,
        )
    except UpdateError as exc:
        sha = ""
        try:
            sha = Git(repo).out(["rev-parse", "HEAD"])
        except Exception:
            sha = "(unknown)"
        report = UpdateReport(
            ok=False,
            branch="",
            old_sha=sha,
            new_sha=sha,
            expected_sha=args.sha,
            health_url=configured_health_url(repo),
            errors=[str(exc)],
            rollback=rollback_notes(getattr(exc, "rollback_sha", None) or sha, None),
        )
    text = format_report(report)
    sys.stdout.write(text)
    if not report.claim_success():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
