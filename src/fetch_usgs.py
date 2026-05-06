"""Paginated USGS FDSNWS Event API client.

Fetches the South Asia earthquake catalog from
https://earthquake.usgs.gov/fdsnws/event/1/query, caches raw GeoJSON pages
in data/raw/, and produces a tidy Parquet table at
data/processed/catalog.parquet.

Run:  python -m src.fetch_usgs --help

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
from typing import Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


USGS_BASE = "https://earthquake.usgs.gov/fdsnws/event/1"
USGS_PAGE_LIMIT = 20_000

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_BBOX = (23.0, 38.0, 60.0, 78.0)
DEFAULT_MIN_MAG = 3.5
DEFAULT_START = date(1973, 1, 1)

USER_AGENT = (
    "seismic-sequence-mining/0.1 (academic; "
    "+https://github.com/UsmanAhmed98/seismic-sequence-mining)"
)

# GeoJSON property fields kept in the flat table (per PROJECT_PROPOSAL.md §5).
_PROP_FIELDS = ("magType", "place", "type", "status", "net", "rms", "gap", "dmin", "nst")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BBox:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float


def make_session() -> requests.Session:
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = USER_AGENT
    return session


def _params(start: date, end: date, bbox: BBox, min_mag: float) -> dict:
    return {
        "format": "geojson",
        "starttime": start.isoformat(),
        "endtime": end.isoformat(),
        "minlatitude": bbox.min_lat,
        "maxlatitude": bbox.max_lat,
        "minlongitude": bbox.min_lon,
        "maxlongitude": bbox.max_lon,
        "minmagnitude": min_mag,
        "orderby": "time-asc",
    }


def count_events(
    session: requests.Session,
    start: date,
    end: date,
    bbox: BBox,
    min_mag: float,
) -> int:
    r = session.get(f"{USGS_BASE}/count", params=_params(start, end, bbox, min_mag), timeout=60)
    r.raise_for_status()
    return int(r.json()["count"])


def _cache_path(raw_dir: Path, start: date, end: date) -> Path:
    return raw_dir / f"usgs_{start.isoformat()}_{end.isoformat()}.geojson"


def fetch_window(
    session: requests.Session,
    start: date,
    end: date,
    bbox: BBox,
    min_mag: float,
    raw_dir: Path,
    force: bool,
) -> list[dict]:
    cache = _cache_path(raw_dir, start, end)
    if cache.exists() and not force:
        with cache.open() as fh:
            return json.load(fh).get("features", [])

    r = session.get(f"{USGS_BASE}/query", params=_params(start, end, bbox, min_mag), timeout=120)
    r.raise_for_status()
    payload = r.json()
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as fh:
        json.dump(payload, fh)
    return payload.get("features", [])


def fetch_catalog(
    session: requests.Session,
    start: date,
    end: date,
    bbox: BBox,
    min_mag: float,
    raw_dir: Path,
    force: bool = False,
) -> Iterable[dict]:
    """Yield features for [start, end), bisecting when a window exceeds USGS_PAGE_LIMIT."""
    n = count_events(session, start, end, bbox, min_mag)
    logger.info("Window %s → %s: %d events", start, end, n)
    if n == 0:
        return
    if n <= USGS_PAGE_LIMIT:
        yield from fetch_window(session, start, end, bbox, min_mag, raw_dir, force)
        return
    span = (end - start).days
    if span <= 1:
        raise RuntimeError(
            f"Single-day window {start} has {n} events > USGS limit {USGS_PAGE_LIMIT}"
        )
    mid = start + timedelta(days=span // 2)
    yield from fetch_catalog(session, start, mid, bbox, min_mag, raw_dir, force)
    yield from fetch_catalog(session, mid, end, bbox, min_mag, raw_dir, force)


def features_to_dataframe(features: Iterable[dict]) -> pd.DataFrame:
    rows = []
    for feat in features:
        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates") or [None, None, None]
        row = {
            "id": feat.get("id"),
            "time": props.get("time"),
            "longitude": coords[0],
            "latitude": coords[1],
            "depth": coords[2],
            "mag": props.get("mag"),
        }
        for f in _PROP_FIELDS:
            row[f] = props.get(f)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="id").sort_values("time").reset_index(drop=True)
    return df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START,
                   help=f"start date YYYY-MM-DD (default: {DEFAULT_START})")
    p.add_argument("--end", type=date.fromisoformat, default=date.today(),
                   help="end date YYYY-MM-DD (default: today)")
    p.add_argument("--min-mag", type=float, default=DEFAULT_MIN_MAG,
                   help=f"minimum magnitude (default: {DEFAULT_MIN_MAG})")
    p.add_argument("--bbox", nargs=4, type=float,
                   metavar=("MIN_LAT", "MAX_LAT", "MIN_LON", "MAX_LON"),
                   default=list(DEFAULT_BBOX),
                   help=f"bounding box (default: {DEFAULT_BBOX})")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--force", action="store_true",
                   help="re-download cached windows")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    bbox = BBox(*args.bbox)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    session = make_session()
    features = list(fetch_catalog(
        session, args.start, args.end, bbox, args.min_mag, args.raw_dir, args.force,
    ))
    logger.info("Fetched %d features (pre-dedup)", len(features))

    df = features_to_dataframe(features)
    if df.empty:
        logger.warning("No events in the requested window — nothing written")
        return 0

    logger.info(
        "Catalog: %d unique events, %s → %s, mag %s → %s",
        len(df), df["time"].min(), df["time"].max(), df["mag"].min(), df["mag"].max(),
    )
    out = args.processed_dir / "catalog.parquet"
    df.to_parquet(out, index=False)
    logger.info("Wrote %s (%d rows)", out, len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
