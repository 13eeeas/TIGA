"""
core/scheduler.py — Time-of-day resource scheduler for TIGA.

Balances the i9+RTX 4090 between serving LAN users (day) and running
heavy indexing (night) without manual intervention.

Modes
-----
  day   (default 06:00–22:00): portal-serving, low background load
    embed_batch_size  = 16    (VRAM headroom for the chat model)
    extract_workers   = 2     (background CPU only)
    run_indexing      = False (pause heavy phases)

  night (22:00–06:00): index-optimised, no users expected
    embed_batch_size  = 256   (saturate the 4090)
    extract_workers   = 8     (max CPU for PDF/DOCX parsing)
    run_indexing      = True

A mode file (tiga_work/scheduler/mode.txt) is written here and read by
the pipeline at each batch boundary — nothing needs to restart.

Priority ladder
---------------
  1. TIGA_MODE env var   ("day" | "night") — highest
  2. mode.txt content    (written by apply_mode / daemon)
  3. Wall-clock time     (fallback)

Usage
-----
  from core.scheduler import get_current_mode, get_schedule_cfg, apply_mode

  mode = get_current_mode()          # "day" | "night"
  scfg = get_schedule_cfg()          # ScheduleConfig for current mode
  scfg.embed_batch_size              # 16 or 256
  scfg.extract_workers               # 2 or 8
  scfg.run_indexing                  # False or True

  # Override for the day:
  apply_mode("night")                # force night-mode now

  # Daemon (blocking, checks every 60 s):
  from core.scheduler import run_daemon
  run_daemon()
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MODE_CHOICES = ("day", "night")
_MODE_FILE = "mode.txt"


# ---------------------------------------------------------------------------
# ScheduleConfig — per-mode resource settings
# ---------------------------------------------------------------------------

@dataclass
class ScheduleConfig:
    mode: str
    embed_batch_size: int
    extract_workers: int
    run_indexing: bool
    ollama_num_parallel: int
    description: str

    @classmethod
    def from_cfg(cls, mode: str) -> "ScheduleConfig":
        """
        Build ScheduleConfig from config.yaml scheduler section.
        Falls back to hardcoded defaults when config is unavailable.
        """
        try:
            from config import cfg as _cfg
            if mode == "night":
                return cls(
                    mode="night",
                    embed_batch_size=_cfg.scheduler_night_embed_batch,
                    extract_workers=_cfg.scheduler_night_workers,
                    run_indexing=_cfg.scheduler_night_run_indexing,
                    ollama_num_parallel=1,
                    description=(
                        f"Night mode: embed_batch={_cfg.scheduler_night_embed_batch} "
                        f"workers={_cfg.scheduler_night_workers} indexing=on"
                    ),
                )
            else:
                return cls(
                    mode="day",
                    embed_batch_size=_cfg.scheduler_day_embed_batch,
                    extract_workers=_cfg.scheduler_day_workers,
                    run_indexing=_cfg.scheduler_day_run_indexing,
                    ollama_num_parallel=1,
                    description=(
                        f"Day mode: embed_batch={_cfg.scheduler_day_embed_batch} "
                        f"workers={_cfg.scheduler_day_workers} indexing=off"
                    ),
                )
        except Exception:
            # Config unavailable — safe defaults
            if mode == "night":
                return cls(
                    mode="night",
                    embed_batch_size=256,
                    extract_workers=8,
                    run_indexing=True,
                    ollama_num_parallel=1,
                    description="Night mode (defaults): max embed throughput, full indexing",
                )
            return cls(
                mode="day",
                embed_batch_size=16,
                extract_workers=2,
                run_indexing=False,
                ollama_num_parallel=1,
                description="Day mode (defaults): query-optimised, indexing paused",
            )


# ---------------------------------------------------------------------------
# Mode directory
# ---------------------------------------------------------------------------

def _mode_dir() -> Path:
    try:
        from config import cfg as _cfg
        d = _cfg.work_dir / "scheduler"
    except Exception:
        d = Path(__file__).resolve().parent.parent / "tiga_work" / "scheduler"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

def get_current_mode(hour: int | None = None) -> str:
    """
    Return "day" or "night".

    Priority:
      1. TIGA_MODE env var  ("day" | "night")
      2. mode.txt in scheduler dir
      3. Wall-clock hour: 06:00–22:00 → day, else → night
    """
    env = os.environ.get("TIGA_MODE", "").lower().strip()
    if env in _MODE_CHOICES:
        return env

    mode_file = _mode_dir() / _MODE_FILE
    if mode_file.exists():
        try:
            content = mode_file.read_text(encoding="utf-8").strip().lower()
            if content in _MODE_CHOICES:
                return content
        except OSError:
            pass

    h = hour if hour is not None else datetime.now().hour
    try:
        from config import cfg as _cfg
        day_start = _cfg.scheduler_day_start_hour
        night_start = _cfg.scheduler_night_start_hour
    except Exception:
        day_start, night_start = 6, 22

    return "day" if day_start <= h < night_start else "night"


def apply_mode(mode: str) -> ScheduleConfig:
    """
    Write mode to mode.txt and return its ScheduleConfig.
    The pipeline reads this file at each batch boundary.
    """
    if mode not in _MODE_CHOICES:
        raise ValueError(f"mode must be one of {_MODE_CHOICES!r}, got {mode!r}")

    mode_file = _mode_dir() / _MODE_FILE
    mode_file.write_text(mode, encoding="utf-8")

    scfg = ScheduleConfig.from_cfg(mode)
    logger.info(
        "scheduler: applied mode=%s — %s", mode, scfg.description
    )
    return scfg


def get_schedule_cfg() -> ScheduleConfig:
    """Return the ScheduleConfig for the current mode."""
    return ScheduleConfig.from_cfg(get_current_mode())


# ---------------------------------------------------------------------------
# Optional: renice Ollama (Linux only, non-fatal)
# ---------------------------------------------------------------------------

def _renice_ollama(mode: str) -> None:
    """
    Adjust Ollama process scheduling priority via renice/ionice (Linux).

      night: nice=10, ionice=idle  — TIGA indexing gets CPU/IO priority
      day:   nice=0               — Ollama responds quickly to user queries
    """
    import shutil
    import subprocess

    if not shutil.which("pgrep") or not shutil.which("renice"):
        return

    try:
        result = subprocess.run(
            ["pgrep", "-x", "ollama"], capture_output=True, text=True
        )
        pids = result.stdout.strip().split()
        if not pids:
            return

        nice_val = "10" if mode == "night" else "0"
        for pid in pids:
            subprocess.run(
                ["renice", "-n", nice_val, "-p", pid],
                capture_output=True,
            )

        if shutil.which("ionice") and mode == "night":
            for pid in pids:
                # class 3 = idle: only gets IO when nothing else needs it
                subprocess.run(
                    ["ionice", "-c", "3", "-p", pid],
                    capture_output=True,
                )
        logger.debug("renice: pids=%s nice=%s mode=%s", pids, nice_val, mode)
    except Exception as e:
        logger.debug("renice_ollama (non-fatal): %s", e)


# ---------------------------------------------------------------------------
# Status summary (for `tiga schedule status`)
# ---------------------------------------------------------------------------

def status_summary() -> dict[str, Any]:
    """Return a dict describing current scheduler state."""
    mode = get_current_mode()
    scfg = ScheduleConfig.from_cfg(mode)
    mode_file = _mode_dir() / _MODE_FILE
    source = "env" if os.environ.get("TIGA_MODE") else (
        "mode.txt" if mode_file.exists() else "time-of-day"
    )
    return {
        "mode": mode,
        "source": source,
        "embed_batch_size": scfg.embed_batch_size,
        "extract_workers": scfg.extract_workers,
        "run_indexing": scfg.run_indexing,
        "description": scfg.description,
        "mode_file": str(mode_file),
    }


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------

def run_daemon(check_interval_s: int = 60) -> None:
    """
    Blocking daemon: checks time every check_interval_s seconds, switches
    mode when the day/night boundary is crossed, and optionally renices Ollama.

    Run as:
        python -m core.scheduler
    or via:
        tiga schedule --daemon
    """
    current: str | None = None
    logger.info("scheduler daemon started (check_interval=%ds)", check_interval_s)

    try:
        while True:
            mode = get_current_mode()
            if mode != current:
                scfg = apply_mode(mode)
                _renice_ollama(mode)
                logger.info(
                    "scheduler: switched to %s — %s", mode, scfg.description
                )
                current = mode
            time.sleep(check_interval_s)
    except KeyboardInterrupt:
        logger.info("scheduler daemon stopped")


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description="TIGA resource scheduler")
    ap.add_argument("--daemon", action="store_true", help="Run as continuous daemon")
    ap.add_argument("--mode", choices=list(_MODE_CHOICES), help="Force a specific mode")
    ap.add_argument("--interval", type=int, default=60, help="Daemon check interval (seconds)")
    args = ap.parse_args()

    if args.mode:
        scfg = apply_mode(args.mode)
        _renice_ollama(args.mode)
        print(f"Mode set to: {scfg.description}")
    elif args.daemon:
        run_daemon(check_interval_s=args.interval)
    else:
        s = status_summary()
        print(f"Current mode : {s['mode']}  (source: {s['source']})")
        print(f"Description  : {s['description']}")
        print(f"embed_batch  : {s['embed_batch_size']}")
        print(f"workers      : {s['extract_workers']}")
        print(f"run_indexing : {s['run_indexing']}")
