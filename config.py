"""
config.py — TIGA configuration loader.

Load order:
  1. Determine work_dir: TIGA_WORK_DIR env var → default ./tiga_work
  2. Read {work_dir}/config.yaml
  3. Validate required fields → raise ConfigError if missing
  4. Normalize all paths via pathlib.Path.resolve()
  5. Apply \\\\?\\ prefix on Windows when path len > 248

Public API:
  load_config(config_file, work_dir) -> Config
  ollama_available(base_url) -> bool
  cfg: Config  (module-level singleton, loaded on import)
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """Raised when config.yaml is missing required fields or is malformed."""


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_WIN_PATH_LIMIT = 248


def _win_long_path(p: Path) -> Path:
    """Apply \\\\?\\ prefix on Windows if absolute path exceeds limit."""
    if platform.system() == "Windows":
        s = str(p)
        if len(s) > _WIN_PATH_LIMIT and not s.startswith("\\\\?\\"):
            return Path("\\\\?\\" + s)
    return p


def _resolve(raw: str | Path) -> Path:
    """Resolve to absolute path and apply Windows long-path prefix if needed."""
    return _win_long_path(Path(raw).resolve())


# ---------------------------------------------------------------------------
# Work directory discovery
# ---------------------------------------------------------------------------

_WORK_DIR_ENV = "TIGA_WORK_DIR"
_CONFIG_FILENAME = "config.yaml"
_REQUIRED_FIELDS = ["index_roots"]


def _get_work_dir() -> Path:
    env = os.environ.get(_WORK_DIR_ENV)
    if env:
        return _resolve(env)
    # Default: ./tiga_work relative to repo root (where this file lives)
    return Path(__file__).resolve().parent / "tiga_work"


# ---------------------------------------------------------------------------
# Config class
# ---------------------------------------------------------------------------

class Config:
    def __init__(
        self,
        data: dict[str, Any],
        work_dir: Path,
        config_file: Path,
    ) -> None:
        self._data = data
        self.work_dir = work_dir
        self.config_file = config_file

        # --- index_roots (required) ---
        self.index_roots: list[Path] = [
            _resolve(r) for r in data["index_roots"]
        ]

        # --- glob filters ---
        self.include_globs: list[str] = data.get("include_globs", ["**/*"])
        self.exclude_globs: list[str] = data.get("exclude_globs", [])

        # --- file size limit ---
        max_mb: int = data.get("max_file_mb", 2048)
        self.max_file_bytes: int = max_mb * 1024 * 1024

        # --- lane rules ---
        lane = data.get("lane_rules", {})
        self.text_extractable_exts: set[str] = {
            e.lower()
            for e in lane.get(
                "text_extractable_exts",
                [".pdf", ".docx", ".pptx", ".txt", ".md"],
            )
        }
        self.metadata_only_exts: set[str] = {
            e.lower()
            for e in lane.get(
                "metadata_only_exts",
                [".dwg", ".rvt", ".ifc", ".skp", ".jpg", ".jpeg", ".png",
                 ".mp4", ".mov", ".avi"],
            )
        }

        # --- Ollama ---
        oll = data.get("ollama", {})
        self.ollama_base_url: str = oll.get("base_url", "http://localhost:11434")
        self.embed_model: str = oll.get("embed_model", "nomic-embed-text")
        self.chat_model: str = oll.get("chat_model", "mistral")
        self.ollama_timeout: int = oll.get("timeout_seconds", 30)
        self.gpu_layers: int = oll.get("gpu_layers", -1)
        self.num_ctx: int = oll.get("num_ctx", 4096)

        # --- batch size for embeddings ---
        self.batch_size: int = data.get("batch_size", 32)

        # --- Server ---
        srv = data.get("server", {})
        self.server_host: str = srv.get("host", "0.0.0.0")
        self.server_port: int = srv.get("port", 7860)
        self.server_workers: int = srv.get("workers", 2)

        # --- UI ---
        ui = data.get("ui", {})
        self.ui_port: int = ui.get("port", 8501)
        self.ui_title: str = ui.get("title", "TIGA Hunt")
        self.max_session_history: int = ui.get("max_session_history", 20)

        # --- Project inference ---
        proj = data.get("project_inference", {})
        self.project_inference_enable: bool = proj.get("enable", True)
        self.project_confidence_threshold: float = proj.get(
            "confidence_threshold_unknown", 0.5
        )
        self.project_patterns: list[dict[str, Any]] = proj.get("patterns", [])
        self.project_keyword_boosts: list[dict[str, Any]] = proj.get(
            "keyword_boosts", []
        )

        # --- Typology inference (building type) ---
        typo = data.get("typology_inference", {})
        self.typology_confidence_threshold: float = typo.get(
            "confidence_threshold_unknown", 0.5
        )
        self.typology_keyword_map: dict[str, list[str]] = typo.get(
            "keyword_map", {}
        )

        # --- Retrieval ---
        ret = data.get("retrieval", {})
        self.top_k: int = ret.get("top_k_default", 5)
        self.fts_weight: float = ret.get("hybrid_weight_bm25", 0.4)
        self.vector_weight: float = ret.get("hybrid_weight_vector", 0.6)

        # --- OCR (opt-in only) ---
        ocr = data.get("ocr", {})
        self.ocr_enabled: bool = ocr.get("enabled", False)
        self.tesseract_cmd: str = ocr.get("tesseract_cmd", "tesseract")

        # --- Einstein (Phase 2) ---
        ein = data.get("einstein", {})
        self.einstein_enable: bool = ein.get("enable", False)
        self.einstein_base_model: str = ein.get("base_model", "mistral")
        _adp = ein.get("adapter_path")
        self.einstein_adapter_path: Path | None = _resolve(_adp) if _adp else None

    # --- Path helpers ---

    def get_db_path(self) -> Path:
        return self.work_dir / "db" / "tiga.db"

    def get_vector_dir(self) -> Path:
        return self.work_dir / "vectors"

    def get_log_dir(self) -> Path:
        return self.work_dir / "logs"

    def get_report_dir(self) -> Path:
        return self.work_dir / "reports"

    def ensure_dirs(self) -> None:
        """Create all work subdirectories if they don't exist."""
        for d in [
            self.get_db_path().parent,
            self.get_vector_dir(),
            self.get_log_dir(),
            self.get_report_dir(),
        ]:
            d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_yaml(config_file: Path) -> dict[str, Any]:
    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_file}\n"
            "Run `python tiga.py init` to create it."
        )
    with config_file.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _validate(data: dict[str, Any]) -> None:
    for field in _REQUIRED_FIELDS:
        if not data.get(field):
            raise ConfigError(
                f"config.yaml missing required field: '{field}'. "
                "Please set it before running."
            )


def load_config(
    config_file: Path | None = None,
    work_dir: Path | None = None,
) -> Config:
    """
    Load and return a Config instance.

    Args:
        config_file: Explicit path to config.yaml (overrides discovery).
        work_dir:    Override work directory (overrides env var + default).
    """
    _work_dir = work_dir or _get_work_dir()
    _config_file = config_file or (_work_dir / _CONFIG_FILENAME)
    data = _load_yaml(_config_file)
    _validate(data)
    return Config(data, _work_dir, _config_file)


# ---------------------------------------------------------------------------
# Ollama availability check
# ---------------------------------------------------------------------------

def ollama_available(base_url: str | None = None) -> bool:
    """
    Ping Ollama's /api/tags endpoint.
    Returns False without raising if Ollama is unreachable.
    """
    import urllib.request

    url = (base_url or "http://localhost:11434").rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

# Loaded on first import of this module.
# If config.yaml doesn't exist, raises FileNotFoundError with clear message.
# The `init` subcommand in tiga.py intentionally avoids importing cfg,
# so `python tiga.py init` works before config.yaml exists.
cfg: Config = load_config()
