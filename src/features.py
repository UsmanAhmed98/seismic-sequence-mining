"""Feature engineering for the earthquake catalog.

Loads data/processed/catalog.parquet and writes data/processed/features.parquet
with per-event features used by the clustering and classification stages:

  Catalog passthrough:
    mag, depth

  Pairwise (event vs. previous event):
    dt_prev          time since previous catalog event, in days
    d_prev_km        great-circle distance to previous event, km

  Zaliapin & Ben-Zion (2008) nearest-neighbour distance:
    zaliapin_T       rescaled time              (t_j - t_i) * 10^(-q*b*M_i)
    zaliapin_R       rescaled distance          r_ij^d * 10^(-(1-q)*b*M_i)
    eta_zaliapin     T * R   — the bimodal headline feature
    log10_eta        log10(eta_zaliapin)
    nn_distance      Zaliapin without magnitude scaling (b=0); raw space-time NN
    log10_nn_distance

  Local activity in a (50 km, 30 day) window before each event:
    local_density_7d
    local_density_30d
    mag_diff_local_max     mag - max(mag in window); negative => smaller than locals
    bath_residual          (max(mag) in window) - mag - 1.2
    omori_residual         observed 7-day rate - Omori-law prediction

Defaults follow Zaliapin & Ben-Zion (2008): b=1.0, d=1.6, q=0.5; window radius
50 km, time windows 7 / 30 days; Omori K=10/day, c=0.05 d, p=1.0.

Run:  python -m src.features --help

See PROJECT_PROPOSAL.md sections 7 (Phase C) and 9.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "processed" / "catalog.parquet"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "features.parquet"

EARTH_RADIUS_KM = 6371.0
SECONDS_PER_DAY = 86_400.0
DAYS_PER_YEAR = 365.25

logger = logging.getLogger(__name__)


def _times_in_days(time_series: pd.Series) -> np.ndarray:
    """Days since epoch, resolution- and timezone-independent.

    Going via total_seconds() on a UTC delta avoids both the
    int64-resolution gotcha (ms vs. ns) and the numpy "no representation
    of timezones" warning when casting tz-aware datetimes.
    """
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    return (time_series - epoch).dt.total_seconds().to_numpy() / SECONDS_PER_DAY


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _validate_sorted(df: pd.DataFrame) -> None:
    if not df["time"].is_monotonic_increasing:
        raise ValueError("Catalog must be sorted ascending by time")


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    times_days = _times_in_days(df["time"])
    lats = df["latitude"].to_numpy()
    lons = df["longitude"].to_numpy()

    dt_prev = np.full(len(df), np.nan)
    d_prev = np.full(len(df), np.nan)
    dt_prev[1:] = times_days[1:] - times_days[:-1]
    d_prev[1:] = haversine_km(lats[:-1], lons[:-1], lats[1:], lons[1:])

    df["dt_prev"] = dt_prev
    df["d_prev_km"] = d_prev
    return df


def _zaliapin_for_event(
    j: int,
    times_years: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    mags: np.ndarray,
    b: float,
    d: float,
    q: float,
) -> tuple[float, float, float, int]:
    """Return (T, R, eta, parent_idx) for event j; NaN/-1 if no valid parent."""
    if j == 0:
        return (np.nan, np.nan, np.nan, -1)
    dt_years = times_years[j] - times_years[:j]
    valid = (dt_years > 0) & np.isfinite(mags[:j])
    if not valid.any():
        return (np.nan, np.nan, np.nan, -1)
    idx = np.flatnonzero(valid)
    dt = dt_years[idx]
    r_km = haversine_km(lats[j], lons[j], lats[idx], lons[idx])
    r_km = np.maximum(r_km, 1e-3)  # avoid log(0); 1 m floor for co-located events
    m_parent = mags[idx]
    T = dt * 10.0 ** (-q * b * m_parent)
    R = (r_km ** d) * 10.0 ** (-(1.0 - q) * b * m_parent)
    eta = T * R
    k_local = int(np.argmin(eta))
    return (float(T[k_local]), float(R[k_local]), float(eta[k_local]), int(idx[k_local]))


def add_zaliapin(
    df: pd.DataFrame,
    b: float = 1.0,
    d: float = 1.6,
    q: float = 0.5,
) -> pd.DataFrame:
    n = len(df)
    times_years = _times_in_days(df["time"]) / DAYS_PER_YEAR
    lats = df["latitude"].to_numpy()
    lons = df["longitude"].to_numpy()
    mags = df["mag"].to_numpy()

    T = np.full(n, np.nan)
    R = np.full(n, np.nan)
    eta = np.full(n, np.nan)
    parent = np.full(n, -1, dtype=np.int64)

    nn_eta = np.full(n, np.nan)  # b=0 variant: no magnitude rescaling
    for j in range(n):
        T[j], R[j], eta[j], parent[j] = _zaliapin_for_event(j, times_years, lats, lons, mags, b, d, q)
        _, _, nn_eta[j], _ = _zaliapin_for_event(j, times_years, lats, lons, mags, 0.0, d, q)

    df["zaliapin_T"] = T
    df["zaliapin_R"] = R
    df["eta_zaliapin"] = eta
    df["log10_eta"] = np.log10(np.where(eta > 0, eta, np.nan))
    df["zaliapin_parent_idx"] = parent
    df["nn_distance"] = nn_eta
    df["log10_nn_distance"] = np.log10(np.where(nn_eta > 0, nn_eta, np.nan))
    return df


def add_local_activity(
    df: pd.DataFrame,
    radius_km: float = 50.0,
    window_days_long: float = 30.0,
    window_days_short: float = 7.0,
) -> pd.DataFrame:
    n = len(df)
    times_days = _times_in_days(df["time"])
    lats = df["latitude"].to_numpy()
    lons = df["longitude"].to_numpy()
    mags = df["mag"].to_numpy()

    count_long = np.zeros(n, dtype=np.int32)
    count_short = np.zeros(n, dtype=np.int32)
    max_mag_long = np.full(n, np.nan)
    last_big_dt = np.full(n, np.nan)  # days since largest event in long window

    left = 0
    for j in range(n):
        while left < j and (times_days[j] - times_days[left]) > window_days_long:
            left += 1
        if left == j:
            continue
        idx = np.arange(left, j)
        r_km = haversine_km(lats[j], lons[j], lats[idx], lons[idx])
        in_circle = r_km <= radius_km
        if not in_circle.any():
            continue
        sel = idx[in_circle]
        count_long[j] = sel.size
        # Short window subset
        dt_days = times_days[j] - times_days[sel]
        count_short[j] = int((dt_days <= window_days_short).sum())
        local_mags = mags[sel]
        finite = np.isfinite(local_mags)
        if finite.any():
            k = int(np.argmax(np.where(finite, local_mags, -np.inf)))
            max_mag_long[j] = local_mags[k]
            last_big_dt[j] = dt_days[k]

    df["local_density_30d"] = count_long
    df["local_density_7d"] = count_short
    df["mag_diff_local_max"] = df["mag"] - max_mag_long
    df["bath_residual"] = max_mag_long - df["mag"].to_numpy() - 1.2
    df["_local_max_dt_days"] = last_big_dt
    return df


def add_omori_residual(
    df: pd.DataFrame,
    radius_km: float = 50.0,
    window_days_short: float = 7.0,
    K: float = 10.0,
    c: float = 0.05,
    p: float = 1.0,
) -> pd.DataFrame:
    """Observed 7-day rate minus Omori prediction at lag from local max.

    For each event we already know (from add_local_activity):
      local_density_7d   - count in last 7 days within radius
      _local_max_dt_days - days since the largest event in the 30-day window

    The Omori expected rate is K / (dt + c)^p events/day at lag dt; observed
    rate is local_density_7d / window_days_short.
    """
    if "_local_max_dt_days" not in df.columns:
        raise ValueError("add_local_activity must be called first")
    dt = df["_local_max_dt_days"].to_numpy()
    observed_rate = df["local_density_7d"].to_numpy() / window_days_short
    expected_rate = np.where(np.isfinite(dt), K / (dt + c) ** p, np.nan)
    df["omori_residual"] = observed_rate - expected_rate
    return df.drop(columns=["_local_max_dt_days"])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    _validate_sorted(df)
    df = df.copy()
    add_basic_features(df)
    add_zaliapin(df)
    add_local_activity(df)
    add_omori_residual(df)
    return df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.catalog.exists():
        logger.error("Catalog not found: %s — run `python -m src.fetch_usgs` first", args.catalog)
        return 1

    logger.info("Loading %s", args.catalog)
    df = pd.read_parquet(args.catalog)
    logger.info("Loaded %d events", len(df))

    logger.info("Computing features (this is O(N^2) for Zaliapin η)…")
    feats = build_features(df)

    finite_eta = feats["eta_zaliapin"].dropna()
    logger.info(
        "η: n=%d, log10 median=%.2f, [p10, p90]=[%.2f, %.2f]",
        len(finite_eta),
        np.log10(finite_eta).median(),
        np.log10(finite_eta).quantile(0.1),
        np.log10(finite_eta).quantile(0.9),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(args.out, index=False)
    logger.info("Wrote %s (%d rows × %d cols)", args.out, len(feats), feats.shape[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
