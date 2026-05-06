# Earthquake Sequence Mining for South Asia — Semester Project Proposal

> **Course:** Machine Learning · **Program:** MS Data Science (2nd Semester)
> **Format:** Solo · **Duration:** ~12 weeks
> **Status:** Proposal / planning document

---

## TL;DR

Earthquake catalogs are a mix of **independent (background) earthquakes** and **clustered sequences** (foreshock → mainshock → aftershock). Telling them apart is called **declustering** and it is a real, ongoing problem in seismology. This project downloads the **USGS** earthquake catalog for South Asia (Pakistan / Iran / Afghanistan, 1973–present), uses **DBSCAN** to find the clusters automatically, **K-means / LDA** to characterise the background, and **SVM / Decision Tree / Naive Bayes / KNN** to classify each event. We compare the result against the classical **Reasenberg (1985)** baseline that seismologists have used for 40 years.

The deliverable is a reproducible pipeline + a 12–20 page report that uses every algorithm in the syllabus on a real, primary-source dataset that no one downloads from Kaggle.

---

## Table of Contents

1. [Background — In Plain Words](#1-background--in-plain-words)
2. [Why This Project Matters](#2-why-this-project-matters)
3. [Project Goals](#3-project-goals)
4. [Out of Scope](#4-out-of-scope)
5. [Data Source](#5-data-source)
6. [Domain Concepts You Need to Know](#6-domain-concepts-you-need-to-know)
7. [Methodology](#7-methodology)
8. [Algorithm-to-Subtask Mapping](#8-algorithm-to-subtask-mapping)
9. [Feature Engineering](#9-feature-engineering)
10. [Evaluation Plan](#10-evaluation-plan)
11. [Expected Findings](#11-expected-findings)
12. [12-Week Schedule](#12-12-week-schedule)
13. [Project Structure](#13-project-structure)
14. [Tech Stack](#14-tech-stack)
15. [Risks & Mitigations](#15-risks--mitigations)
16. [Ethics & Data Usage](#16-ethics--data-usage)
17. [Future Extensions](#17-future-extensions)
18. [Glossary](#18-glossary)
19. [References & Sources](#19-references--sources)

---

## 1. Background — In Plain Words

When a large earthquake happens, it is rarely a single event. The same region typically produces:

- **Foreshocks** — small earthquakes *before* the main one. We can only call them "fore" *after* the big one hits.
- **Mainshock** — the largest event in the sequence.
- **Aftershocks** — many smaller earthquakes that follow the mainshock, sometimes for years.
- **Background events** — earthquakes that are not part of any sequence, just routine tectonic noise.

A country's earthquake catalog therefore looks like this: imagine 50,000 dots on a map. Most dots are loners (background). The rest come in tight families of 200–500 dots clustered close in space and close in time — those are aftershock sequences.

The problem of separating these two things is called **declustering**. It is the central preprocessing step before computing seismic hazard for a region (i.e., "how often does this place produce a big earthquake?"). Until very recently, declustering used hand-tuned space-time windows (Gardner-Knopoff 1974, Reasenberg 1985). Modern research uses **DBSCAN** and supervised ML — which is exactly what this project will replicate and extend.

---

## 2. Why This Project Matters

- **Public safety**: Pakistan sits on the Eurasian–Indian plate boundary. The 2005 Kashmir earthquake killed ~87,000 people. Knowing the *background* rate of large quakes (after declustering) is what tells planners how often to expect another one.
- **Methodologically interesting**: The features themselves come from earthquake physics (Omori, Båth, Zaliapin), not generic ML feature engineering. This makes the project distinctive.
- **Real baseline to beat**: Reasenberg's 1985 algorithm is the standard. Comparing ML against it gives the report a sharp, defensible punchline.
- **Authentic data**: USGS is a primary government source. Not Kaggle.

---

## 3. Project Goals

| # | Goal | How it's measured |
|---|------|-------------------|
| G1 | Build a reproducible USGS catalog ingestion pipeline | One command rebuilds the dataset |
| G2 | Decluster the catalog using DBSCAN / ST-DBSCAN | Cluster count, silhouette, recovery of known sequences |
| G3 | Classify event types (fore / main / after / background) with ≥ 3 classifiers | F1-score per class, confusion matrix, k-fold CV |
| G4 | Estimate magnitudes of aftershocks via KNN | MAE / RMSE on held-out sequences |
| G5 | Profile background seismicity into zones via K-means + LDA | Visual map + zone descriptions |
| G6 | Compare ML pipeline vs. Reasenberg baseline | Poisson goodness-of-fit on declustered residual catalog |
| G7 | Produce a 12–20 page IEEE-style report + reproducible repo | Submission package |

---

## 4. Out of Scope

To keep the project finishable in 12 weeks:

- ❌ **Real-time earthquake prediction** — impossible and dishonest to claim.
- ❌ **Deep learning** — syllabus is classical ML; no CNNs/LSTMs.
- ❌ **Tsunami modeling** — needs ocean physics, separate field.
- ❌ **Building damage estimation** — needs structural / GIS data we will not collect.
- ❌ **Other regions outside the 23–38°N, 60–78°E box** — global scope blows up the timeline.

---

## 5. Data Source

### Primary

- **USGS FDSNWS Event Web Service** (free, no API key)
  - Documentation: https://earthquake.usgs.gov/fdsnws/event/1/
  - Endpoint: `https://earthquake.usgs.gov/fdsnws/event/1/query`
  - Format: GeoJSON / CSV / QuakeML
  - ANSS ComCat (the underlying database): https://earthquake.usgs.gov/data/comcat/

**Filter for this project:**
- Bounding box: lat 23–38, lon 60–78 (Pakistan + Iran + Afghanistan)
- Time: 1973-01-01 → today (post-WWSSN era for reliable instrumental records)
- Magnitude: ≥ 3.5 (above magnitude of completeness for the region)
- Expected size: 30,000–60,000 events

**Fields used:** `time, latitude, longitude, depth, mag, magType, place, type, status, net, rms, gap, dmin, nst`.

### Cross-validation source

- **ISC Bulletin** (International Seismological Centre): http://www.isc.ac.uk/iscbulletin/search/catalogue/

### Optional regional source

- **Pakistan Meteorological Department — Seismic Monitoring Centre**: https://www.pmd.gov.pk/en/ (check current data accessibility)

---

## 6. Domain Concepts You Need to Know

Read these before writing any code. They drive the feature engineering.

| Concept | Plain meaning | Formal form |
|---|---|---|
| **Omori's law** (1894) | Aftershocks decay roughly as 1 / time | `n(t) = K / (t + c)^p` |
| **Båth's law** (1965) | Biggest aftershock ≈ mainshock magnitude − 1.2 | `M_main − M_largest_after ≈ 1.2` |
| **Gutenberg-Richter law** | Earthquake frequency drops by ~10× per +1 magnitude | `log10 N = a − b·M` |
| **Zaliapin nearest-neighbor distance (η)** | Single number that says "is this event part of a cluster?" | `η = T · R^d`, where T = rescaled time gap, R = rescaled distance, d ≈ 1.6 (fractal dim of seismicity) |
| **Reasenberg algorithm** (1985) | Classical link-based declustering using physically motivated space-time windows | Reference baseline |
| **ETAS model** (Ogata 1988) | Statistical model that predicts each event's "triggering" rate | Used in advanced ML approaches |

Reading order:
1. [Wikipedia — Aftershock](https://en.wikipedia.org/wiki/Aftershock) (orientation)
2. Zaliapin & Ben-Zion (2013) — Earthquake clusters in southern California (the η-distance paper)
3. [Helmstetter & Sornette 2003 — Foreshocks explained by cascades of triggered seismicity](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2003JB002409)
4. [Springer 2025 — Review of clustering algorithms for spatiotemporal seismicity](https://link.springer.com/article/10.1007/s10462-025-11229-3)

---

## 7. Methodology

### Phase A — Data acquisition

Hit the USGS FDSNWS endpoint with paginated requests (USGS limits each query to 20,000 records). Save raw GeoJSON to `data/raw/`. Convert to a single tidy Parquet file in `data/processed/catalog.parquet`.

### Phase B — Exploratory Data Analysis

- Magnitude histogram → estimate magnitude of completeness M_c (use maximum-curvature method).
- Map of events colored by depth (Cartopy).
- Time series of monthly event counts.
- Inter-event time histogram (should show heavy tail — that's the clustering signature).

### Phase C — Feature engineering

Build features for each event (see [Section 9](#9-feature-engineering)).

### Phase D — Unsupervised clustering

- **DBSCAN** in (lat, lon, time) space with rescaling so that 1 km ≈ 1 day after normalisation.
- Variant: **ST-DBSCAN** with separate ε for space and time.
- Sweep `eps` and `min_samples`; pick by stability + silhouette.
- Each non-noise cluster = one earthquake sequence. Within each cluster, label the largest-magnitude event = mainshock, earlier events in the cluster = foreshocks, later events = aftershocks. Noise points = background.

### Phase E — Background regime profiling

- Take only the noise (background) events.
- Run **K-means** on (lat, lon, depth, magnitude) with `k` chosen by elbow + silhouette.
- Each cluster = one tectonic regime. Label them (e.g. "Hindu Kush deep", "Makran shallow", "Kashmir thrust").
- **LDA** on the labeled regimes to project to 2D for the report's headline figure.

### Phase F — Supervised classification

Inputs: features from Phase C. Labels: 4 classes from Phase D (fore / main / after / background).
- Train **SVM** (linear + RBF), **Decision Tree** (depth tuned), **Gaussian Naive Bayes**.
- Stratified 5-fold CV. Report per-class precision/recall/F1, macro-F1, confusion matrices.
- Discuss class imbalance (background dominates) and how each algorithm handles it.

### Phase G — Magnitude estimation

For each mainshock, build a precursor feature vector from the preceding 30-day window. Use **KNN** (Euclidean and Mahalanobis) to predict the magnitude *bin* of the largest aftershock to follow. Compare to the Båth's-law prediction (M_main − 1.2).

### Phase H — Baseline comparison

Implement (or wrap) the **Reasenberg (1985)** algorithm. Run it on the same catalog. Compare:
1. Number of clusters identified.
2. Catalog size after declustering.
3. Inter-event time distribution of declustered catalog vs. exponential (Kolmogorov–Smirnov test). A truly declustered catalog should pass the K–S test for a Poisson process.

---

## 8. Algorithm-to-Subtask Mapping

| Subtask | Algorithm | Why this algorithm |
|---|---|---|
| Cluster discovery (declustering) | **DBSCAN** + **ST-DBSCAN** | Density-based, no `k` needed, handles arbitrary cluster shape |
| Background-zone profiling | **K-means** | Compact, well-separated tectonic regimes |
| 2D projection / visualisation | **LDA** | Maximum class-separating projection of labeled events |
| Event-type classification | **SVM**, **Decision Tree**, **Naive Bayes** | Three different decision-boundary families to compare |
| Magnitude prediction | **KNN** | "Similar past sequences" intuition fits instance-based learning |

Every algorithm in the syllabus has a defensible role — none is "shoe-horned in".

---

## 9. Feature Engineering

For every event `i`, compute:

| Feature | Formula / source | Reason |
|---|---|---|
| `mag` | from catalog | base feature |
| `depth` | from catalog | shallow vs. deep events behave differently |
| `dt_prev` | time since previous event in catalog | clustering signature |
| `d_prev_km` | great-circle distance to previous event | clustering signature |
| `eta_zaliapin` | `T · R^1.6` where T, R are rescaled | the single most diagnostic feature |
| `nn_distance` | distance to nearest event by space-time metric | supports density-based clustering |
| `local_density_30d` | count of events within 50 km in last 30 days | precursor signal |
| `mag_diff_local_max` | mag − max(mag) in last 30 days, 50 km | foreshock indicator |
| `omori_residual` | observed rate − Omori prediction in last 7 days | aftershock-decay deviation |
| `bath_residual` | M_main − M_event − 1.2 | Båth-law deviation |

The Zaliapin η-distribution is bimodal (background lobe + clustered lobe), and the saddle between the lobes gives the natural threshold. Plotting it is **the** diagnostic figure of the project.

---

## 10. Evaluation Plan

### Sanity checks (must pass before reporting any result)

1. **Gutenberg-Richter test** — log-frequency vs. magnitude should be linear above M_c with slope ≈ −1.
2. **Zaliapin η-distribution** — must be bimodal. If not, something is wrong with feature engineering.
3. **Known-sequence recovery** — the 2005 Kashmir M7.6, 2013 Awaran M7.7, and 2015 Hindu Kush M7.5 mainshocks must each fall inside a recovered cluster, and the in-cluster event count must decay per Omori's law.

### Quantitative metrics

| Subtask | Metric |
|---|---|
| Clustering | Silhouette, cluster count, % events in clusters, sequence-recovery on labeled known events |
| Classification | Per-class precision / recall / F1, macro-F1, ROC-AUC, confusion matrix, 95% CI via stratified 5-fold CV |
| KNN magnitude | MAE, RMSE, % within ±0.5 of true |
| Baseline comparison | K–S statistic of declustered inter-event times vs. exponential distribution |

### Reproducibility check

`git clone … && pip install -r requirements.txt && make all` should regenerate every figure in the report from scratch.

---

## 11. Expected Findings

Concrete, falsifiable claims the report should be able to make:

1. ✅ Pakistan's catalog follows **Gutenberg-Richter** with b-value ≈ 0.9–1.1.
2. ✅ Aftershock decay in recovered sequences fits **Omori's law** with p ≈ 0.7–1.3.
3. ✅ Largest aftershocks satisfy **Båth's law** within ±0.4 magnitude on average.
4. ✅ DBSCAN produces a **bimodal η-distribution** — empirically confirming the cluster vs. background dichotomy.
5. ✅ One of {SVM, Decision Tree, Naive Bayes} achieves macro-F1 ≥ 0.75 on the 4-class problem.
6. ✅ K-means recovers ≥ 4 distinguishable tectonic regimes (Hindu Kush, Makran, Kashmir thrust, central Iran, Sulaiman/Kirthar belt).
7. ⚖️ **Open question**: does the ML pipeline produce a more strongly Poisson declustered catalog than Reasenberg? This is the headline question.

---

## 12. 12-Week Schedule

| Wk | Focus | Deliverable |
|---|---|---|
| 1 | Proposal, lit scan | 2-page proposal + reading notes |
| 2 | Data ingestion, EDA | `catalog.parquet`, EDA notebook |
| 3 | Feature engineering | `features.parquet` |
| 4 | Reasenberg baseline | Baseline labels |
| 5 | DBSCAN parameter sweep | Cluster assignments + plots |
| 6 | ST-DBSCAN extension, sequence-recovery validation | Sanity-check report |
| 7 | K-means + LDA on background | Regime map |
| 8 | SVM + DT + NB training | Model comparison table |
| 9 | Cross-validation, error analysis | Confusion matrices + CIs |
| 10 | KNN magnitude estimator | KNN evaluation |
| 11 | Pipeline vs. Reasenberg comparison | Comparison figures |
| 12 | Report writing, slide deck, repo cleanup | Final submission |

Buffer: weeks 9 and 11 are the easiest to compress if earlier weeks slip.

---

## 13. Project Structure

```
SemesterProject/
├── README.md                       # Setup + headline results
├── PROJECT_PROPOSAL.md             # This document
├── requirements.txt
├── Makefile                        # `make all` regenerates everything
├── data/
│   ├── raw/                        # USGS GeoJSON dumps (gitignored)
│   └── processed/                  # catalog.parquet, features.parquet
├── src/
│   ├── fetch_usgs.py               # paginated USGS API client
│   ├── features.py                 # Zaliapin η, Omori/Båth residuals, etc.
│   ├── reasenberg.py               # baseline declustering
│   ├── clustering.py               # DBSCAN / ST-DBSCAN wrappers
│   ├── classify.py                 # SVM / DT / NB / KNN training
│   └── evaluate.py                 # metrics, plots, baseline comparison
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_clustering.ipynb
│   ├── 04_classification.ipynb
│   └── 05_results.ipynb
├── report/
│   └── report.pdf                  # IEEE/ACM style, 12–20 pages
└── figures/                        # all plots used in the report
```

---

## 14. Tech Stack

- **Python 3.11+**
- **Data**: `pandas`, `numpy`, `pyarrow` (Parquet), `requests`
- **ML**: `scikit-learn` (all required algorithms live here)
- **Geo / maps**: `cartopy`, `geopy` (great-circle distance)
- **Stats**: `scipy.stats` (K–S test, distributions)
- **Plotting**: `matplotlib`, `seaborn`
- **Notebooks**: `jupyter`
- **Reproducibility**: `make`, `pip-compile` / `requirements.txt`, fixed random seeds

---

## 15. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| USGS API rate limits or downtime | Low | Medium | Cache raw responses; ISC Bulletin as fallback |
| Catalog too sparse for ML (< 10k events) | Low | High | Lower magnitude floor to 3.0; widen bounding box to all South Asia |
| DBSCAN parameters very sensitive | Medium | Medium | Use Zaliapin η threshold from bimodal saddle as principled default |
| Class imbalance kills classifier | High | Medium | Stratified sampling, class weights, macro-F1 instead of accuracy |
| Reasenberg implementation buggy | Medium | High | Cross-check against published declustered counts for known catalogs (e.g., Southern California test set used in literature) |
| Schedule slip in weeks 5–6 | Medium | High | Weeks 9 and 11 are pre-identified compression points |

---

## 16. Ethics & Data Usage

- **Source**: USGS earthquake data is public-domain (US-government work). No licensing constraints.
- **Attribution**: cite USGS ANSS ComCat in every figure caption that uses the catalog.
- **No PII**: earthquake data has no personal information.
- **No "prediction" claims**: short-term earthquake prediction is not scientifically valid; the report must distinguish *clustering / declustering / hazard estimation* (legitimate) from *prediction* (not legitimate). This wording matters for a graded report.

---

## 17. Future Extensions

For the report's "Future Work" section:

- Replace DBSCAN with **HDBSCAN** (variable density).
- Add the **ETAS** statistical model and compare its triggering probabilities to the ML classifier output.
- Extend to **deep learning** (graph neural networks on the event graph) — natural follow-up for a thesis.
- **Cross-region transfer**: train on California, test on Pakistan, study transferability.
- Build a small **web dashboard** (Streamlit) showing the live declustered catalog.

---

## 18. Glossary

| Term | Definition |
|---|---|
| **Catalog** | A list of recorded earthquakes with time, location, magnitude, depth |
| **Declustering** | Separating clustered events (foreshocks/aftershocks) from independent (background) events |
| **Mainshock** | The largest event in a sequence |
| **Foreshock** | An event that precedes a mainshock and is part of the same sequence |
| **Aftershock** | An event that follows a mainshock and is part of the same sequence |
| **Background event** | An earthquake not associated with any cluster |
| **Magnitude of completeness (M_c)** | Smallest magnitude above which the catalog is complete (every event of that size or larger was recorded) |
| **b-value** | Slope of the Gutenberg-Richter law; usually ≈ 1 |
| **Sequence** | A foreshock + mainshock + aftershock cluster |
| **FDSNWS** | Federation of Digital Seismograph Networks Web Service — a standard API spec |
| **Reasenberg algorithm** | Classical 1985 link-based declustering algorithm |
| **ETAS** | Epidemic-Type Aftershock Sequence — a self-exciting point-process model |

---

## 19. References & Sources

### Data sources
- [USGS FDSNWS Event API documentation](https://earthquake.usgs.gov/fdsnws/event/1/)
- [USGS ANSS ComCat documentation](https://earthquake.usgs.gov/data/comcat/)
- [USGS Earthquake Catalog Search UI](https://earthquake.usgs.gov/earthquakes/search/)
- [USGS Real-time GeoJSON feeds](https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php)
- [International Seismological Centre — Bulletin](http://www.isc.ac.uk/iscbulletin/search/catalogue/)
- [Pakistan Meteorological Department](https://www.pmd.gov.pk/en/)

### Domain knowledge
- [Wikipedia — Aftershock](https://en.wikipedia.org/wiki/Aftershock)
- [Wikipedia — Omori's law](https://en.wikipedia.org/wiki/Aftershock#Omori's_law)
- [Wikipedia — Gutenberg–Richter law](https://en.wikipedia.org/wiki/Gutenberg%E2%80%93Richter_law)

### Methodological literature
- [Helmstetter & Sornette (2003) — Foreshocks explained by cascades of triggered seismicity](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2003JB002409)
- [Helmstetter (2003) — Mainshocks are aftershocks of conditional foreshocks](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2002JB001991)
- [Springer 2025 — A review on clustering algorithms for spatiotemporal seismicity analysis](https://link.springer.com/article/10.1007/s10462-025-11229-3)
- [arXiv 2025 — Earthquake Declustering Using Supervised Machine Learning](https://arxiv.org/html/2504.08052)
- [Springer 2023 — ST-DBSCAN-EV for seismic clustering in Chile](https://link.springer.com/article/10.1007/s10651-023-00594-3)
- [Springer 2022 — Identification and spatio-temporal analysis of earthquake clusters using SOM–DBSCAN](https://link.springer.com/article/10.1007/s00521-022-08085-5)
- [MDPI 2022 — Identification and Temporal Characteristics of Earthquake Clusters in Greece](https://www.mdpi.com/2076-3417/12/4/1908)
- [Springer 2018 — A Variable ε-DBSCAN Algorithm for Declustering Earthquake Catalogs](https://link.springer.com/chapter/10.1007/978-981-13-1592-3_50)

### Tools / Libraries
- [scikit-learn — DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [scikit-learn — KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [scikit-learn — LinearDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
- [scikit-learn — SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [scikit-learn — DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [scikit-learn — GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
- [scikit-learn — KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [Cartopy — geographic plotting](https://scitools.org.uk/cartopy/docs/latest/)

---

*Document maintained at `PROJECT_PROPOSAL.md` in the project root. Update as the project evolves.*
