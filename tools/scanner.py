"""
tools/scanner.py — Windirstat-style project folder scanner for TIGA.

Quickly scans a directory and reports:
  • File type breakdown (extension → count + size)
  • Top-20 largest files
  • Treemap-style ASCII visualisation
  • Recommendation: which folders to use for phased index testing

Usage
-----
  python tools/scanner.py /path/to/project/folder
  python tools/scanner.py /path/to/archive --phases         # multi-project scan
  tiga scan /path/to/project/folder
  tiga scan /path/to/archive --phases

Options
-------
  --phases      Scan all sub-folders as separate projects, recommend
                which to use for 1/2/3/5/10-project test phases.
  --depth N     How many directory levels to treat as "project root"
                when --phases is used (default 1).
  --json        Output machine-readable JSON instead of text.
  --top N       Show top-N file types (default 20).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Extension categories — drives colour / icon in the ASCII treemap
# ---------------------------------------------------------------------------

_EXT_GROUPS: dict[str, list[str]] = {
    "pdf":      [".pdf"],
    "word":     [".doc", ".docx"],
    "excel":    [".xls", ".xlsx", ".xlsm"],
    "ppt":      [".ppt", ".pptx"],
    "cad":      [".dwg", ".dxf", ".dwf", ".dgn"],
    "bim":      [".rvt", ".rfa", ".rte", ".ifc", ".skp", ".3dm", ".nwd"],
    "image":    [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff",
                 ".webp", ".svg", ".ai", ".eps", ".psd", ".psb"],
    "video":    [".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".m4v"],
    "text":     [".txt", ".md", ".rst", ".log", ".csv", ".tsv", ".json",
                 ".xml", ".yaml", ".yml", ".ini", ".cfg"],
    "archive":  [".zip", ".rar", ".7z", ".tar", ".gz", ".bz2"],
    "email":    [".msg", ".eml", ".pst"],
    "other":    [],
}

_EXT_TO_GROUP: dict[str, str] = {}
for _grp, _exts in _EXT_GROUPS.items():
    for _e in _exts:
        _EXT_TO_GROUP[_e] = _grp


def _group(ext: str) -> str:
    return _EXT_TO_GROUP.get(ext.lower(), "other")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExtStat:
    ext:   str
    count: int = 0
    bytes: int = 0

    @property
    def mb(self) -> float:
        return self.bytes / (1024 * 1024)


@dataclass
class FolderScan:
    path:          str
    total_files:   int = 0
    total_bytes:   int = 0
    ext_stats:     dict[str, ExtStat] = field(default_factory=dict)
    top_files:     list[dict[str, Any]] = field(default_factory=list)
    errors:        int = 0

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024 ** 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path":        self.path,
            "total_files": self.total_files,
            "total_bytes": self.total_bytes,
            "total_mb":    round(self.total_mb, 1),
            "total_gb":    round(self.total_gb, 3),
            "ext_stats":   [asdict(v) for v in sorted(
                self.ext_stats.values(), key=lambda x: -x.bytes)],
            "top_files":   self.top_files,
            "errors":      self.errors,
        }


# ---------------------------------------------------------------------------
# Core scanner
# ---------------------------------------------------------------------------

def scan_folder(path: str | Path, top_files: int = 20) -> FolderScan:
    """
    Recursively scan *path* and return a FolderScan.

    Does NOT follow symlinks outside the target tree.
    Reads only stat() — never opens file content.
    """
    root = Path(path)
    result = FolderScan(path=str(root))

    # Collect all files
    _big: list[tuple[int, str]] = []  # (bytes, path)

    for dirpath, _dirs, filenames in os.walk(root, followlinks=False):
        for fname in filenames:
            fp = Path(dirpath) / fname
            try:
                size = fp.stat().st_size
            except OSError:
                result.errors += 1
                continue

            ext = fp.suffix.lower() or "(no ext)"
            if ext not in result.ext_stats:
                result.ext_stats[ext] = ExtStat(ext=ext)
            result.ext_stats[ext].count += 1
            result.ext_stats[ext].bytes += size

            result.total_files += 1
            result.total_bytes += size

            _big.append((size, str(fp)))

    # Top-N largest files
    _big.sort(key=lambda x: -x[0])
    result.top_files = [
        {"path": p, "mb": round(s / (1024 * 1024), 2)}
        for s, p in _big[:top_files]
    ]

    return result


# ---------------------------------------------------------------------------
# ASCII treemap renderer
# ---------------------------------------------------------------------------

_BAR_WIDTH = 40
_ICONS = {
    "pdf": "📄", "word": "📝", "excel": "📊", "ppt": "📑",
    "cad": "📐", "bim": "🏗",  "image": "🖼", "video": "🎬",
    "text": "📃", "archive": "📦", "email": "📧", "other": "📂",
}


def _bar(fraction: float, width: int = _BAR_WIDTH) -> str:
    filled = round(fraction * width)
    return "█" * filled + "░" * (width - filled)


def render_text(scan: FolderScan, top: int = 20) -> str:
    lines: list[str] = []

    size_str = (
        f"{scan.total_gb:.2f} GB" if scan.total_gb >= 1
        else f"{scan.total_mb:.1f} MB"
    )
    lines.append(f"\n{'='*62}")
    lines.append(f"  {scan.path}")
    lines.append(f"  {scan.total_files:,} files  ·  {size_str}")
    lines.append(f"{'='*62}")

    if not scan.ext_stats:
        lines.append("  (empty folder)")
        return "\n".join(lines)

    lines.append(f"\n  {'Extension':<14} {'Files':>7}  {'Size':>9}  {'% size':>6}  Bar")
    lines.append(f"  {'-'*14} {'-'*7}  {'-'*9}  {'-'*6}  {'-'*_BAR_WIDTH}")

    sorted_exts = sorted(scan.ext_stats.values(), key=lambda x: -x.bytes)
    tb = max(scan.total_bytes, 1)

    for es in sorted_exts[:top]:
        frac = es.bytes / tb
        icon = _ICONS.get(_group(es.ext), "📂")
        mb_str = f"{es.mb:,.1f} MB" if es.mb < 1000 else f"{es.bytes/(1024**3):.2f} GB"
        lines.append(
            f"  {icon} {es.ext:<12} {es.count:>7,}  {mb_str:>9}  {frac:>5.1%}  {_bar(frac, 30)}"
        )

    if len(sorted_exts) > top:
        rest_count = sum(e.count for e in sorted_exts[top:])
        rest_bytes = sum(e.bytes for e in sorted_exts[top:])
        rest_mb = rest_bytes / (1024 * 1024)
        lines.append(
            f"  {'… other':<14} {rest_count:>7,}  {rest_mb:>8.1f}M"
        )

    if scan.top_files:
        lines.append(f"\n  Top {len(scan.top_files)} largest files:")
        for f in scan.top_files[:10]:
            name = Path(f["path"]).name
            lines.append(f"    {f['mb']:>8.1f} MB  {name}")

    if scan.errors:
        lines.append(f"\n  ⚠ {scan.errors} access errors (permission denied / broken symlink)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase-recommendation scanner
# ---------------------------------------------------------------------------

@dataclass
class ProjectProfile:
    path:        str
    name:        str
    total_files: int
    total_bytes: int
    indexable:   int   # text-extractable files
    size_label:  str   # "small" | "medium" | "large" | "xlarge"

    @property
    def mb(self) -> float:
        return self.total_bytes / (1024 * 1024)


_INDEXABLE_EXTS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".txt", ".md", ".rst", ".csv",
}


def _classify_size(total_files: int) -> str:
    if total_files <= 50:
        return "small"
    if total_files <= 300:
        return "medium"
    if total_files <= 2000:
        return "large"
    return "xlarge"


def scan_for_phases(
    root: str | Path,
    depth: int = 1,
) -> dict[str, Any]:
    """
    Scan sub-folders of *root* at *depth* levels as individual projects.

    Returns a dict with:
      "projects": [ProjectProfile ...] sorted by indexable file count ASC
      "recommendations": {
          "phase_1":  [project_name],
          "phase_2":  [project_name, project_name],
          ...
          "phase_10": [...],
      }
    """
    root_path = Path(root)

    # Discover candidate project folders at the requested depth
    candidates: list[Path] = []
    if depth == 1:
        candidates = [p for p in root_path.iterdir() if p.is_dir()]
    else:
        candidates = [
            p for p in root_path.rglob("*")
            if p.is_dir()
            and len(p.relative_to(root_path).parts) == depth
        ]

    profiles: list[ProjectProfile] = []

    for cand in sorted(candidates):
        total_files = 0
        total_bytes = 0
        indexable   = 0
        for dp, _ds, fns in os.walk(cand, followlinks=False):
            for fn in fns:
                fp = Path(dp) / fn
                try:
                    s = fp.stat().st_size
                except OSError:
                    continue
                total_files += 1
                total_bytes += s
                if fp.suffix.lower() in _INDEXABLE_EXTS:
                    indexable += 1

        profiles.append(ProjectProfile(
            path        = str(cand),
            name        = cand.name,
            total_files = total_files,
            total_bytes = total_bytes,
            indexable   = indexable,
            size_label  = _classify_size(total_files),
        ))

    # Sort by indexable count ascending (smallest first)
    profiles.sort(key=lambda p: p.indexable)

    # Build recommendations for each phase count
    _phase_counts = [1, 2, 3, 5, 10]
    recommendations: dict[str, Any] = {}

    for n in _phase_counts:
        key = f"phase_{n}"
        if n >= len(profiles):
            # Use all projects
            recommendations[key] = [p.name for p in profiles]
        else:
            # Pick n projects that span small→large (evenly spaced)
            if n == 1:
                # Smallest project
                chosen = [profiles[0]]
            elif n == 2:
                # Smallest + largest
                chosen = [profiles[0], profiles[-1]]
            else:
                # Evenly-spaced indices across sorted list
                import math
                step = (len(profiles) - 1) / (n - 1)
                indices = {round(i * step) for i in range(n)}
                chosen = [profiles[i] for i in sorted(indices)]

            recommendations[key] = [
                {
                    "name":       p.name,
                    "files":      p.total_files,
                    "indexable":  p.indexable,
                    "size_label": p.size_label,
                    "mb":         round(p.mb, 1),
                }
                for p in chosen
            ]

    return {
        "root":            str(root_path),
        "project_count":   len(profiles),
        "projects":        [
            {
                "name":       p.name,
                "path":       p.path,
                "files":      p.total_files,
                "indexable":  p.indexable,
                "size_label": p.size_label,
                "mb":         round(p.mb, 1),
            }
            for p in profiles
        ],
        "recommendations": recommendations,
    }


def render_phases_text(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"\n{'='*62}")
    lines.append(f"  Phase-test recommendations  —  {result['root']}")
    lines.append(f"  {result['project_count']} projects found")
    lines.append(f"{'='*62}")

    lines.append(f"\n  {'Project':<30} {'Files':>6}  {'Indexable':>9}  {'Size':>10}  Label")
    lines.append(f"  {'-'*30} {'-'*6}  {'-'*9}  {'-'*10}  -----")

    for p in result["projects"]:
        mb = p["mb"]
        sz = f"{mb:.0f} MB" if mb < 1000 else f"{mb/1024:.2f} GB"
        lines.append(
            f"  {p['name']:<30} {p['files']:>6,}  {p['indexable']:>9,}  {sz:>10}  {p['size_label']}"
        )

    lines.append("\n  Recommended projects for each test phase:")
    for phase_key, picks in result["recommendations"].items():
        n = phase_key.split("_")[1]
        if isinstance(picks[0], dict):
            names = ", ".join(f"{p['name']} ({p['size_label']})" for p in picks)
        else:
            names = ", ".join(picks)
        lines.append(f"    Phase {n:>2} project(s): {names}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Windirstat-style TIGA folder scanner"
    )
    ap.add_argument("path", help="Folder to scan")
    ap.add_argument(
        "--phases", action="store_true",
        help="Scan sub-folders as projects and recommend test phases",
    )
    ap.add_argument(
        "--depth", type=int, default=1,
        help="Sub-folder depth when using --phases (default 1)",
    )
    ap.add_argument(
        "--json", dest="as_json", action="store_true",
        help="Output JSON instead of text",
    )
    ap.add_argument(
        "--top", type=int, default=20,
        help="Show top-N file types (default 20)",
    )
    args = ap.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"[ERROR] Path does not exist: {target}", file=sys.stderr)
        sys.exit(1)

    if args.phases:
        result = scan_for_phases(target, depth=args.depth)
        if args.as_json:
            print(json.dumps(result, indent=2))
        else:
            print(render_phases_text(result))
    else:
        scan = scan_folder(target, top_files=args.top)
        if args.as_json:
            print(json.dumps(scan.to_dict(), indent=2))
        else:
            print(render_text(scan, top=args.top))


if __name__ == "__main__":
    _main()
