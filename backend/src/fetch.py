"""Phase 1 — Fetch + cache hourly ERA5 weather for Varna.

Downloads ~85 years of hourly data from the Open-Meteo archive API in ~5-year chunks,
concatenates and sorts them, validates a continuous gap-free hourly index, and caches the
result to ``data/varna_hourly.parquet``.

Usage (from the ``backend/`` directory)::

    python -m src.fetch            # use cache if present
    python -m src.fetch --refresh  # force re-download

Data: Open-Meteo Historical Weather API (ECMWF ERA5), CC BY 4.0.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, timedelta

import pandas as pd
import requests

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fetch")

REQUEST_TIMEOUT = 120  # seconds per chunk request
INTER_CHUNK_SLEEP = 2.0  # polite pause between chunks to respect rate limits
MAX_RETRIES = 6  # retries on 429 / transient errors
BACKOFF_BASE = 5.0  # seconds; exponential backoff doubles each retry


def _end_date() -> date:
    """ERA5 lags by ~5-7 days; pull up to today minus the configured lag."""
    return date.today() - timedelta(days=config.ERA5_LAG_DAYS)


def _chunk_ranges(start: date, end: date, years: int):
    """Yield (start_date, end_date) pairs covering [start, end] in ~`years`-year windows."""
    cur = start
    while cur <= end:
        # Inclusive chunk end: last day before the same calendar day `years` later.
        try:
            nxt = cur.replace(year=cur.year + years)
        except ValueError:  # Feb 29 edge case
            nxt = cur.replace(year=cur.year + years, day=28)
        chunk_end = min(nxt - timedelta(days=1), end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def _get_with_retry(params: dict, label: str) -> dict:
    """GET the archive endpoint with exponential backoff on 429 / transient errors.

    Open-Meteo enforces per-minute/hour rate limits; on HTTP 429 we honor the
    ``Retry-After`` header when present, otherwise back off exponentially.
    """
    delay = BACKOFF_BASE
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(config.ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                log.warning(
                    "Rate limited (429) on %s; retry %d/%d after %.0fs",
                    label, attempt, MAX_RETRIES, wait,
                )
                time.sleep(wait)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.ConnectionError, requests.Timeout) as exc:
            log.warning(
                "Transient error on %s (%s); retry %d/%d after %.0fs",
                label, exc, attempt, MAX_RETRIES, delay,
            )
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"Giving up on {label} after {MAX_RETRIES} retries (rate limit?)")


def _fetch_chunk(start: date, end: date) -> pd.DataFrame:
    """Fetch a single date range from the archive API and return a tidy DataFrame."""
    params = {
        "latitude": config.LAT,
        "longitude": config.LON,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(config.HOURLY_VARS),
        "timezone": config.TZ,
    }
    log.info("Fetching %s -> %s ...", start, end)
    payload = _get_with_retry(params, label=f"{start}..{end}")

    if "hourly" not in payload:
        raise RuntimeError(f"Unexpected API response for {start}..{end}: {payload}")

    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    log.info("  -> %d rows", len(df))
    return df


def fetch_all(refresh: bool = False) -> pd.DataFrame:
    """Fetch the full history (chunked), validate, cache, and return the DataFrame."""
    config.ensure_dirs()

    if config.CACHE_PARQUET.exists() and not refresh:
        log.info("Cache hit: %s (use --refresh to re-download)", config.CACHE_PARQUET)
        df = pd.read_parquet(config.CACHE_PARQUET)
        _report(df)
        return df

    start = date.fromisoformat(config.START_DATE)
    end = _end_date()
    log.info("Downloading %s .. %s in ~%d-year chunks", start, end, config.FETCH_CHUNK_YEARS)

    frames = []
    for c_start, c_end in _chunk_ranges(start, end, config.FETCH_CHUNK_YEARS):
        frames.append(_fetch_chunk(c_start, c_end))
        time.sleep(INTER_CHUNK_SLEEP)  # be polite to the API between chunks

    df = pd.concat(frames)
    # Chunk boundaries are inclusive on both ends; drop any overlap and sort.
    df = df[~df.index.duplicated(keep="first")].sort_index()

    df = _validate_and_fill(df)

    df.to_parquet(config.CACHE_PARQUET)
    log.info("Cached -> %s", config.CACHE_PARQUET)
    _report(df)
    return df


def _validate_and_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Assert a continuous hourly index (reindexing gaps) and forward-fill stray nulls."""
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    missing = full_index.difference(df.index)
    if len(missing) > 0:
        log.warning(
            "Reindexing to a continuous hourly index: %d missing timestamps inserted",
            len(missing),
        )
        df = df.reindex(full_index)
    df.index.name = "time"

    null_counts = df.isna().sum()
    total_nulls = int(null_counts.sum())
    if total_nulls > 0:
        log.warning(
            "Forward-filling %d stray null cells across columns:\n%s",
            total_nulls,
            null_counts[null_counts > 0].to_string(),
        )
        df = df.ffill().bfill()

    # Hard guarantees the rest of the pipeline relies on.
    assert df.index.is_monotonic_increasing, "index must be sorted"
    assert not df.index.has_duplicates, "index must be unique"
    inferred = pd.infer_freq(df.index)
    assert inferred in ("h", "H"), f"expected hourly frequency, inferred {inferred!r}"
    assert not df.isna().any().any(), "nulls remain after fill"
    return df


def _report(df: pd.DataFrame) -> None:
    log.info("Rows: %d", len(df))
    log.info("Date range: %s .. %s", df.index.min(), df.index.max())
    log.info("Columns: %s", list(df.columns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch + cache hourly ERA5 weather for Varna.")
    parser.add_argument(
        "--refresh", action="store_true", help="Force re-download even if the cache exists."
    )
    args = parser.parse_args()
    fetch_all(refresh=args.refresh)


if __name__ == "__main__":
    main()
