"""
tools/office_setup.py — Office one-click setup helpers.

Used by START-HERE.bat / START-HERE.sh before the first POC test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
POC_TEMPLATE = REPO_ROOT / "docs" / "config.poc.example.yaml"


def _load_config_path() -> Path:
    import os
    from config import load_config

    work = Path(os.environ.get("TIGA_WORK_DIR", REPO_ROOT / "tiga_work"))
    cfg_file = work / "config.yaml"
    if not cfg_file.exists():
        ensure_config(cfg_file)
    return cfg_file


def ensure_config(cfg_file: Path | None = None) -> Path:
    """Create tiga_work/config.yaml from POC template if missing."""
    import os

    work = Path(os.environ.get("TIGA_WORK_DIR", REPO_ROOT / "tiga_work"))
    path = cfg_file or (work / "config.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return path

    if POC_TEMPLATE.exists():
        path.write_text(POC_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        path.write_text(
            "work_dir: ./tiga_work\nindex_roots:\n  - '/path/to/your/projects'\n",
            encoding="utf-8",
        )

    print(f"Created config: {path}")
    return path


def roots_need_setup(cfg_file: Path) -> bool:
    """True when no index_root exists on disk."""
    data = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
    roots = data.get("index_roots") or []
    if not roots:
        return True
    return not any(Path(r).exists() for r in roots)


def configure_interactive(cfg_file: Path | None = None, *, force: bool = False) -> int:
    """
    Prompt for NAS/project folder paths and write index_roots to config.yaml.
    Returns 0 on success, 1 on cancel/error.
    """
    path = cfg_file or _load_config_path()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    if not force and not roots_need_setup(path):
        print("\nindex_roots look configured:")
        for r in data.get("index_roots") or []:
            mark = "OK" if Path(r).exists() else "MISSING"
            print(f"  [{mark}] {r}")
        ans = input("\nKeep these paths? [Y/n] ").strip().lower()
        if ans in ("", "y", "yes"):
            return 0

    print("\n" + "=" * 60)
    print("  TIGA — Where are your project folders?")
    print("=" * 60)
    print("\nEnter 1–5 folder paths (comma-separated).")
    print("Examples:")
    print('  Z:\\Projects')
    print('  Z:\\261_Tianmu, Z:\\262_Hotel, Z:\\263_School')
    print("\nPaths:")

    raw = input().strip()
    if not raw:
        print("Cancelled — edit tiga_work/config.yaml manually.")
        return 1

    roots: list[str] = []
    for part in raw.replace(";", ",").split(","):
        p = part.strip().strip('"').strip("'")
        if not p:
            continue
        resolved = Path(p).expanduser()
        if not resolved.exists():
            print(f"  [WARN] Path not found (will save anyway): {p}")
        roots.append(str(resolved))

    if not roots:
        print("No paths entered.")
        return 1

    data["index_roots"] = roots
    path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"\nSaved {len(roots)} index root(s) to {path}")
    return 0


def ensure_work_dirs() -> None:
    from config import load_config

    cfg = load_config()
    cfg.ensure_dirs()


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    cmd = args[0] if args else "configure"

    if cmd == "configure":
        ensure_config()
        code = configure_interactive(force="--force" in args)
        if code == 0:
            ensure_work_dirs()
        return code

    if cmd == "check":
        path = _load_config_path()
        if roots_need_setup(path):
            print("NEEDS_CONFIGURE")
            return 2
        print("READY")
        return 0

    if cmd == "ensure-config":
        ensure_config()
        ensure_work_dirs()
        return 0

    print(f"Unknown command: {cmd}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
