"""Paginated USGS FDSNWS Event API client.

Fetches the South Asia earthquake catalog (1973-present, M >= 3.5) from
https://earthquake.usgs.gov/fdsnws/event/1/query, writes raw GeoJSON pages
to data/raw/, and produces a tidy Parquet file at data/processed/catalog.parquet.

Run:  python -m src.fetch_usgs --help

See PROJECT_PROPOSAL.md sections 5 and 7 (Phase A).
"""


def main() -> None:
    raise NotImplementedError("Week 2 deliverable")


if __name__ == "__main__":
    main()
