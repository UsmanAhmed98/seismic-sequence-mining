"""Paginated USGS FDSNWS Event API client.

Workflow
--------
1. Hit the FDSNWS /count endpoint to learn how many events live in the
   requested time / space / magnitude window.
2. If the window holds <= 20 000 events (USGS per-query cap), call /query
   and cache the raw GeoJSON to data/raw/. Otherwise split the window in
   half and recurse — each half is then evaluated the same way.
3. Flatten every cached GeoJSON page into a single tidy DataFrame and
   write it to data/processed/catalog.parquet.

Run with --help for CLI options.

See PROJECT_PROPOSAL.md sections 5 and 7 (Phase A).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


USGS_BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1"

# FDSNWS hard limit per /query call. Asking for more returns HTTP 400,
# which is why fetch_catalog_features() bisects the window when count > this.
USGS_MAX_EVENTS_PER_QUERY = 20_000

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Pakistan + Iran + Afghanistan — the South Asia plate-boundary region.
DEFAULT_BBOX = (23.0, 38.0, 60.0, 78.0)
DEFAULT_MIN_MAGNITUDE = 3.5
DEFAULT_START_DATE = date(1973, 1, 1)

USER_AGENT = (
    "seismic-sequence-mining/0.1 (academic; "
    "+https://github.com/UsmanAhmed98/seismic-sequence-mining)"
)

# Per-event GeoJSON properties kept in the flat table (PROJECT_PROPOSAL.md §5).
_GEOJSON_PROPERTIES = (
    "magType", "place", "type", "status",
    "net", "rms", "gap", "dmin", "nst",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BBox:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float


@dataclass(frozen=True)
class FetchConfig:
    """Bundle of parameters that stay constant across every API call in a run."""
    bbox: BBox
    min_magnitude: float
    raw_dir: Path
    force: bool = False


def make_http_session() -> requests.Session:
    """Session with exponential backoff on transient 5xx errors and a polite UA."""
    retry_policy = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry_policy))
    session.headers["User-Agent"] = USER_AGENT
    return session


def _build_query_params(start: date, end: date, config: FetchConfig) -> dict:
    """FDSN query parameters. starttime is inclusive, endtime exclusive."""
    return {
        "format": "geojson",
        "starttime": start.isoformat(),
        "endtime": end.isoformat(),
        "minlatitude": config.bbox.min_lat,
        "maxlatitude": config.bbox.max_lat,
        "minlongitude": config.bbox.min_lon,
        "maxlongitude": config.bbox.max_lon,
        "minmagnitude": config.min_magnitude,
        "orderby": "time-asc",
    }


def _get_json(session: requests.Session, path: str, params: dict, timeout: int) -> dict:
    response = session.get(f"{USGS_BASE_URL}/{path}", params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def count_events(
    session: requests.Session, start: date, end: date, config: FetchConfig,
) -> int:
    """Number of events matching the window — used to decide whether to bisect."""
    payload = _get_json(session, "count", _build_query_params(start, end, config), timeout=60)
    return int(payload["count"])


def _cache_path(start: date, end: date, raw_dir: Path) -> Path:
    return raw_dir / f"usgs_{start.isoformat()}_{end.isoformat()}.geojson"


def _load_cached_geojson(cache: Path) -> list[dict]:
    with cache.open() as f:
        return json.load(f).get("features", [])


def _save_to_cache(cache: Path, payload: dict) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as f:
        json.dump(payload, f)


def _download_window(
    session: requests.Session, start: date, end: date, config: FetchConfig,
) -> dict:
    """One /query call for a leaf window; returns the full GeoJSON payload."""
    return _get_json(session, "query", _build_query_params(start, end, config), timeout=120)


def fetch_window_features(
    session: requests.Session, start: date, end: date, config: FetchConfig,
) -> list[dict]:
    """Return the GeoJSON `features` for one window, hitting the disk cache when present."""
    cache = _cache_path(start, end, config.raw_dir)
    if cache.exists() and not config.force:
        return _load_cached_geojson(cache)
    payload = _download_window(session, start, end, config)
    _save_to_cache(cache, payload)
    return payload.get("features", [])


def fetch_catalog_features(
    session: requests.Session, start: date, end: date, config: FetchConfig,
) -> Iterator[dict]:
    """Yield events for [start, end), bisecting whenever a window exceeds the per-query cap.

    Worked example. Suppose 1973-01-01 → 2026-05-07 holds 50 000 events:
      1. count_events returns 50 000 → bisect at 1999-09-04.
      2. Left half [1973, 1999) returns 22 000 → bisect at 1986-05-03.
      3. [1973, 1986) returns 6 000 → leaf, fetch_window_features is called.
      4. … recursion continues for the remaining halves until every leaf fits one query.
    """
    n_events = count_events(session, start, end, config)
    logger.info("Window %s → %s: %d events", start, end, n_events)

    if n_events == 0:
        return
    if n_events <= USGS_MAX_EVENTS_PER_QUERY:
        yield from fetch_window_features(session, start, end, config)
        return

    span_days = (end - start).days
    if span_days <= 1:
        # Cannot bisect a 1-day window further; only happens for huge swarms.
        raise RuntimeError(
            f"Single day {start} has {n_events} events > USGS limit "
            f"{USGS_MAX_EVENTS_PER_QUERY}; cannot bisect further"
        )
    midpoint = start + timedelta(days=span_days // 2)
    yield from fetch_catalog_features(session, start, midpoint, config)
    yield from fetch_catalog_features(session, midpoint, end, config)


def _feature_to_row(feature: dict) -> dict:
    """Flatten one GeoJSON feature into a single tabular row."""
    properties = feature.get("properties", {})
    coordinates = feature.get("geometry", {}).get("coordinates") or [None, None, None]
    longitude, latitude, depth = coordinates
    row = {
        "id": feature.get("id"),
        "time": properties.get("time"),
        "longitude": longitude,
        "latitude": latitude,
        "depth": depth,
        "mag": properties.get("mag"),
    }
    for name in _GEOJSON_PROPERTIES:
        row[name] = properties.get(name)
    return row


def features_to_dataframe(features: Iterable[dict]) -> pd.DataFrame:
    """Flatten an iterable of GeoJSON features into a tidy, deduplicated DataFrame."""
    df = pd.DataFrame(_feature_to_row(feature) for feature in features)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    # Bisection can revisit the same event at half-window boundaries.
    return df.drop_duplicates(subset="id").sort_values("time").reset_index(drop=True)


def build_catalog(
    session: requests.Session, start: date, end: date, config: FetchConfig,
) -> pd.DataFrame:
    """High-level entry point: fetch every event in the window and return the table."""
    features = list(fetch_catalog_features(session, start, end, config))
    logger.info("Fetched %d features (pre-dedup)", len(features))
    return features_to_dataframe(features)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START_DATE,
                   help=f"start date YYYY-MM-DD (default: {DEFAULT_START_DATE})")
    p.add_argument("--end", type=date.fromisoformat, default=date.today(),
                   help="end date YYYY-MM-DD (default: today)")
    p.add_argument("--min-mag", type=float, default=DEFAULT_MIN_MAGNITUDE,
                   help=f"minimum magnitude (default: {DEFAULT_MIN_MAGNITUDE})")
    p.add_argument("--bbox", nargs=4, type=float,
                   metavar=("MIN_LAT", "MAX_LAT", "MIN_LON", "MAX_LON"),
                   default=list(DEFAULT_BBOX),
                   help=f"bounding box (default: {DEFAULT_BBOX})")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--force", action="store_true",
                   help="re-download windows even if cached")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _log_catalog_summary(df: pd.DataFrame) -> None:
    logger.info(
        "Catalog: %d unique events, %s → %s, mag %s → %s",
        len(df),
        df["time"].min(), df["time"].max(),
        df["mag"].min(), df["mag"].max(),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_logging(args.verbose)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    config = FetchConfig(
        bbox=BBox(*args.bbox),
        min_magnitude=args.min_mag,
        raw_dir=args.raw_dir,
        force=args.force,
    )
    df = build_catalog(make_http_session(), args.start, args.end, config)

    if df.empty:
        logger.warning("No events in the requested window — nothing written")
        return 0

    _log_catalog_summary(df)
    output_path = args.processed_dir / "catalog.parquet"
    df.to_parquet(output_path, index=False)
    logger.info("Wrote %s (%d rows)", output_path, len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
