.PHONY: install data features cluster classify evaluate all clean lab help

PYTHON := uv run python

help:
	@echo "Targets:"
	@echo "  install   Install / sync dependencies (uv sync)"
	@echo "  data      Fetch USGS catalog -> data/processed/catalog.parquet"
	@echo "  features  Build feature table -> data/processed/features.parquet"
	@echo "  cluster   Run DBSCAN / ST-DBSCAN clustering"
	@echo "  classify  Train SVM / DT / NB / KNN classifiers"
	@echo "  evaluate  Generate metrics, plots, baseline comparison"
	@echo "  all       Run the full pipeline (data -> evaluate)"
	@echo "  lab       Launch Jupyter Lab"
	@echo "  clean     Remove processed data and figures"

install:
	uv sync

data:
	$(PYTHON) -m src.fetch_usgs

features: data
	$(PYTHON) -m src.features

cluster: features
	$(PYTHON) -m src.clustering

classify: cluster
	$(PYTHON) -m src.classify

evaluate: classify
	$(PYTHON) -m src.evaluate

all: evaluate

lab:
	uv run jupyter lab

clean:
	rm -rf data/processed/*.parquet figures/*.png figures/*.pdf
