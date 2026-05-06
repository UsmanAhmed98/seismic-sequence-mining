# seismic-sequence-mining

Declustering earthquake catalogs in South Asia with classical machine learning — DBSCAN, K-means, LDA, SVM, Decision Tree, Naive Bayes, KNN — on USGS FDSNWS data (1973–present), benchmarked against the Reasenberg (1985) algorithm. MS Data Science semester project.

> Full design and methodology: see [PROJECT_PROPOSAL.md](PROJECT_PROPOSAL.md).

## Status

🚧 **Scaffold stage** — proposal complete, source modules and notebooks are stubs awaiting implementation per the 12-week schedule in §12 of the proposal.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/UsmanAhmed98/seismic-sequence-mining.git
cd seismic-sequence-mining
uv sync                 # install dependencies into .venv/
make all                # run the full pipeline (once implemented)
make lab                # launch Jupyter Lab for the notebooks
```

## Repository layout

```
.
├── src/                Python modules — pipeline implementation
│   ├── fetch_usgs.py    USGS FDSNWS API client
│   ├── features.py      feature engineering (Zaliapin η, Omori, Båth …)
│   ├── reasenberg.py    1985 declustering baseline
│   ├── clustering.py    DBSCAN / ST-DBSCAN
│   ├── classify.py      SVM / DT / NB / KNN
│   └── evaluate.py      metrics, plots, baseline comparison
├── notebooks/          exploration and results notebooks (01–05)
├── data/
│   ├── raw/             raw USGS GeoJSON dumps (gitignored)
│   └── processed/       catalog.parquet, features.parquet (gitignored)
├── report/             final IEEE-style report (PDF, gitignored)
├── figures/            plots referenced in the report (gitignored)
├── PROJECT_PROPOSAL.md  full project specification
├── Makefile             pipeline orchestration
└── pyproject.toml       uv project file
```

## Data source

[USGS FDSNWS Event Web Service](https://earthquake.usgs.gov/fdsnws/event/1/) — public domain. Cross-validation against the [International Seismological Centre Bulletin](http://www.isc.ac.uk/iscbulletin/search/catalogue/) where useful.

Filter applied: lat 23–38, lon 60–78 (Pakistan + Iran + Afghanistan), 1973-01-01 → present, magnitude ≥ 3.5. Expected catalog size 30k–60k events.

## License

MIT — see [LICENSE](LICENSE).
