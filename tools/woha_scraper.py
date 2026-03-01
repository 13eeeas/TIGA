"""
tools/woha_scraper.py — Crawl woha.net to seed TIGA project_cards.

WOHA organises its portfolio by typology at:
  https://woha.net/typology/{slug}/
and individual project pages at:
  https://woha.net/project/{slug}/

This scraper:
  1. Iterates through all typology listing pages to collect project URLs
  2. Calls project_card.scrape_woha_project() for each project page
  3. Generates a deterministic project_code from the URL slug if none exists
  4. Upserts results into project_cards with source="woha_web"

The scraped data populates:
  name, typology_primary, location, gfa_sqm, concept_summary,
  milestone_dates.completion, awards, woha_url

Usage
-----
  python tools/woha_scraper.py [--dry-run] [--delay 1.5] [--limit 20]
  tiga scrape-woha [--dry-run] [--delay 1.5] [--limit 20]

Options
-------
  --dry-run     Print what would be upserted without writing to DB
  --delay N     Seconds to wait between HTTP requests (default 1.5)
  --limit N     Stop after N projects (0 = no limit; default 0)
  --typology T  Only scrape one typology page (e.g. "hospitality")
  --db PATH     Override path to tiga.db

Requirements
------------
  pip install requests beautifulsoup4
"""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typology page index
# All known WOHA typology slugs → TIGA canonical typology label
# ---------------------------------------------------------------------------

TYPOLOGY_PAGES: list[tuple[str, str]] = [
    ("hospitality",         "hospitality"),
    ("resort",              "hospitality"),
    ("residential_high_rise", "residential"),
    ("residential_mid_rise",  "residential"),
    ("residential_low_rise",  "residential"),
    ("mixed_use",           "mixed-use"),
    ("commercial",          "commercial"),
    ("civic_institutional", "civic"),
    ("educational",         "education"),
    ("masterplan_conceptual", "masterplan"),
    ("adaptive-re-use",     "adaptive-re-use"),
    ("interiors",           "interiors"),
    ("cultural",            "civic"),
    ("conservation",        "adaptive-re-use"),
    ("competition",         "masterplan"),
    ("infrastructure",      "civic"),
    ("landscape",           "masterplan"),
    ("offices",             "commercial"),
    ("exhibition",          "civic"),
]

_WOHA_BASE = "https://woha.net"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _fetch(url: str, timeout: int = 15, retries: int = 3) -> str | None:
    """Fetch URL content with retry on failure. Returns HTML string or None."""
    try:
        import requests
    except ImportError:
        raise ImportError("pip install requests beautifulsoup4")

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=timeout)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 404:
                logger.debug("404 for %s", url)
                return None
            logger.warning("HTTP %d for %s (attempt %d)", resp.status_code, url, attempt)
        except Exception as e:
            logger.warning("Fetch error %s (attempt %d): %s", url, attempt, e)
        if attempt < retries:
            time.sleep(2 ** attempt)  # exponential back-off: 2s, 4s
    return None


# ---------------------------------------------------------------------------
# Project URL discovery
# ---------------------------------------------------------------------------

def _project_urls_from_typology_page(
    typology_slug: str,
    delay: float = 1.5,
) -> list[tuple[str, str]]:
    """
    Scrape a typology listing page and return [(project_url, typology_label), ...].

    The WOHA site renders project cards as <a href="/project/slug/"> elements
    within the typology page.  We look for any link whose href matches the
    /project/{slug}/ pattern.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("pip install requests beautifulsoup4")

    url = f"{_WOHA_BASE}/typology/{typology_slug}/"
    html = _fetch(url)
    if not html:
        logger.warning("Could not fetch typology page: %s", url)
        return []

    soup = BeautifulSoup(html, "html.parser")
    found: list[tuple[str, str]] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        # Match /project/{slug}/ or https://woha.net/project/{slug}/
        m = re.match(r"(?:https://woha\.net)?(/project/[^/]+/)", href)
        if m:
            rel = m.group(1)
            if rel not in seen:
                seen.add(rel)
                found.append((_WOHA_BASE + rel, typology_slug))

    logger.info("typology/%s: found %d project links", typology_slug, len(found))
    time.sleep(delay)
    return found


def discover_all_project_urls(
    typology_filter: str | None = None,
    delay: float = 1.5,
) -> list[tuple[str, str]]:
    """
    Iterate all typology pages and collect unique project URLs.

    Returns [(project_url, typology_slug), ...], deduplicated by URL.
    """
    all_urls: dict[str, str] = {}  # url → typology_slug (first seen wins)

    pages = (
        [(s, t) for s, t in TYPOLOGY_PAGES if s == typology_filter]
        if typology_filter
        else TYPOLOGY_PAGES
    )

    for slug, _label in pages:
        pairs = _project_urls_from_typology_page(slug, delay=delay)
        for url, tsslug in pairs:
            if url not in all_urls:
                all_urls[url] = tsslug

    return list(all_urls.items())


# ---------------------------------------------------------------------------
# Project code derivation
# ---------------------------------------------------------------------------

def _code_from_url(url: str) -> str:
    """
    Derive a deterministic project code from the WOHA project URL slug.

    /project/parkroyal-collection-pickering/ → WOHA-PRCP
    /project/sky-habitat/                    → WOHA-SH
    """
    m = re.search(r"/project/([^/]+)/", url)
    if not m:
        return "WOHA-UNKNOWN"
    slug = m.group(1).upper()
    # Take first letter of each dash-separated word, cap at 6 chars
    parts = slug.split("-")
    initials = "".join(p[0] for p in parts if p)[:6]
    return f"WOHA-{initials}"


# ---------------------------------------------------------------------------
# Main scrape runner
# ---------------------------------------------------------------------------

def run_scraper(
    dry_run: bool = False,
    delay: float = 1.5,
    limit: int = 0,
    typology_filter: str | None = None,
    db_path: str | None = None,
) -> dict[str, int]:
    """
    Full scraper run: discover all project URLs → scrape each → upsert to DB.

    Args:
        dry_run:          Print results without writing to DB.
        delay:            Seconds between HTTP requests.
        limit:            Stop after this many projects (0 = all).
        typology_filter:  Only scrape one typology page slug.
        db_path:          Override path to tiga.db.

    Returns:
        {"discovered": N, "scraped": N, "upserted": N, "failed": N}
    """
    try:
        from bs4 import BeautifulSoup  # noqa: F401  (ensure installed early)
        import requests  # noqa: F401
    except ImportError:
        raise ImportError(
            "WOHA scraper requires: pip install requests beautifulsoup4"
        )

    from core.project_card import scrape_woha_project, upsert_project_card

    stats = {"discovered": 0, "scraped": 0, "upserted": 0, "failed": 0}

    # 1. Discover project URLs from typology listing pages
    logger.info("Discovering project URLs from WOHA typology pages…")
    pairs = discover_all_project_urls(typology_filter=typology_filter, delay=delay)
    stats["discovered"] = len(pairs)
    logger.info("Discovered %d unique project URLs", len(pairs))

    if limit:
        pairs = pairs[:limit]

    # 2. Open DB connection (unless dry-run)
    conn: sqlite3.Connection | None = None
    if not dry_run:
        try:
            if db_path:
                _db = Path(db_path)
            else:
                from config import cfg as _cfg
                _db = _cfg.get_db_path()
            conn = sqlite3.connect(str(_db))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
        except Exception as e:
            logger.error("Could not open DB: %s", e)
            raise

    # 3. Scrape each project page and upsert
    for idx, (url, typology_slug) in enumerate(pairs, start=1):
        logger.info("[%d/%d] Scraping %s", idx, len(pairs), url)
        try:
            data = scrape_woha_project(url)
            if not data:
                logger.warning("No data returned for %s", url)
                stats["failed"] += 1
                continue

            # Ensure typology_primary is set from the listing page
            # (individual pages may not have structured typology markup)
            typology_label = dict(TYPOLOGY_PAGES).get(typology_slug, typology_slug)
            if not data.get("typology_primary"):
                data["typology_primary"] = typology_label

            # Derive project_code if not already extracted
            if not data.get("project_code"):
                data["project_code"] = _code_from_url(url)

            stats["scraped"] += 1

            if dry_run:
                print(f"\n--- {data.get('project_code')} ---")
                for k, v in data.items():
                    if not k.startswith("_") and k != "data_sources":
                        print(f"  {k}: {v!r}")
            else:
                upsert_project_card(conn, data)  # type: ignore[arg-type]
                stats["upserted"] += 1
                logger.info(
                    "Upserted: %s — %s (%s)",
                    data.get("project_code"),
                    data.get("name", "?"),
                    data.get("location", "?"),
                )

        except Exception as e:
            logger.error("Failed to scrape %s: %s", url, e)
            stats["failed"] += 1

        time.sleep(delay)

    if conn:
        conn.commit()
        conn.close()

    logger.info(
        "Scrape complete: discovered=%d scraped=%d upserted=%d failed=%d",
        stats["discovered"], stats["scraped"], stats["upserted"], stats["failed"],
    )
    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    ap = argparse.ArgumentParser(
        description="Scrape woha.net projects into TIGA project_cards"
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print results without writing to DB",
    )
    ap.add_argument(
        "--delay", type=float, default=1.5,
        help="Seconds between HTTP requests (default 1.5)",
    )
    ap.add_argument(
        "--limit", type=int, default=0,
        help="Stop after N projects (0 = no limit)",
    )
    ap.add_argument(
        "--typology", default=None,
        help="Only scrape one typology page slug (e.g. 'hospitality')",
    )
    ap.add_argument(
        "--db", default=None,
        help="Override path to tiga.db",
    )
    args = ap.parse_args()

    stats = run_scraper(
        dry_run=args.dry_run,
        delay=args.delay,
        limit=args.limit,
        typology_filter=args.typology,
        db_path=args.db,
    )
    print(f"\nResults: {stats}")


if __name__ == "__main__":
    _main()
