# Separating Level and Spatial Skill in Satellite Nocturnal Urban Heat Estimation

> Ayush Gouda, Aditya Prakash, Hema M S

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-orange)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.5.3-green)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Intra-urban nocturnal land surface temperature (LST) is estimated at 1 km node
spacing across three inland Indian cities in two seasonal windows, from
ERA5-Land atmospheric variables and open building morphology. Models are
evaluated under spatially blocked cross-validation.

The study is about **evaluation**, not architecture. Prediction error splits
exactly into a citywide *level* component and a spatial *anomaly* component:

```
MSE = MSE_level + MSE_anomaly
```

The cross-term vanishes because spatial anomalies sum to zero within a night,
so the split is algebraic rather than approximate. Aggregate RMSE is dominated
by the level term — a quantity with no intra-urban content — which means the
*magnitude* of an RMSE difference says little about the difference in spatial
skill that produced it.

---

## Findings

**Level dominates RMSE.** Across six city–season configurations, tree models
and a graph network differ in RMSE by 0.38–1.00 °C while their spatial-anomaly
errors differ by 0.031–0.168 °C.

**RMSE ranks correctly but distorts magnitude.** The trees are better on
spatial skill in all six configurations (bootstrap intervals over nights
exclude zero). But the pre-monsoon Bengaluru RMSE gap is 1.6× the post-monsoon
Hyderabad gap, while its spatial-skill gap is 8× larger.

**Morphology is invisible to RMSE.** Removing the five morphological
predictors costs 0.161 mean spatial correlation in all six configurations,
while RMSE changes by at most 0.148 °C and improves in 4 of 6.

**Every architecture under-disperses.** Predicted spatial standard deviation
falls below observed in every configuration and model family, with spread
ratios of 0.61–0.87.

### Error decomposition (°C)

| Config | Tree RMSE | Tree level | Tree anom. | Graph RMSE | Graph level | Graph anom. | Structureless ref. |
|---|---|---|---|---|---|---|---|
| BLR pre  | 0.611 | 0.178 | 0.584 | 1.339 | 1.108 | 0.752 | 0.845 |
| BLR post | 0.721 | 0.279 | 0.665 | 1.480 | 1.271 | 0.758 | 0.853 |
| HYD pre  | 0.759 | 0.393 | 0.650 | 1.139 | 0.883 | 0.719 | 0.951 |
| HYD post | 0.715 | 0.385 | 0.602 | 1.175 | 0.997 | 0.622 | 0.985 |
| DLH pre  | 0.721 | 0.194 | 0.695 | 1.717 | 1.519 | 0.801 | 1.158 |
| DLH post | 0.603 | 0.150 | 0.584 | 1.003 | 0.761 | 0.654 | 1.043 |

RMSE² = level² + anomaly², exactly. The structureless reference assigns each
night's observed citywide mean to every node; its RMSE is the pooled spatial
standard deviation and the score attainable with no spatial skill at all.

### Model cost

| Model | Parameters | Train (ms/step) | Inference (ms) |
|---|---|---|---|
| GAT + skip, k=8 | 35,075 | 9.9 | 3.28 |
| MLP | 33,603 | 4.6 | 1.07 |
| MLP + spatial lag | 33,763 | 5.4 | 0.98 |

---

## Data

| Feature | Source | Role |
|---|---|---|
| Night LST (MOD11A1, MYD11A1) | MODIS Terra + Aqua | **target**, 1 km |
| t2m, u10, v10, ssr, str, slhf | ERA5-Land | atmospheric predictors |
| NDVI, NDBI | Landsat | surface indices |
| Building height, building cover | Google Open Buildings 2.5D Temporal V1 | morphology |
| Tropospheric NO₂ | Sentinel-5P | anthropogenic proxy |
| Daytime LST | Landsat | validation only, never a predictor |

**Windows.** Pre-monsoon 1 March–30 April 2025; post-monsoon 15 November
2025–15 January 2026.

**Nodes.** 1 km grid in EPSG:32643, matched to the MODIS native resolution.
Sampling finer than the target manufactures apparent spatial skill, because a
model can then recover which source pixel a node occupies rather than any
property of the surface.

| City | Candidate nodes | Retained (pre / post) | Night passes (pre / post) |
|---|---|---|---|
| Bengaluru | 973 | 329 / 326 | 78 / 85 |
| Hyderabad | 770 | 477 / 480 | 92 / 104 |
| Delhi | 873 | 830 / 826 | 108 / 105 |

Retention is after a 60% coverage filter.

**Past LST is deliberately excluded from the predictor set.** With lagged
target values available, persistence carries the prediction and morphology
goes unused.

### Two data handling details that matter

**Overpass timestamps.** MOD11A1 and MYD11A1 are daily composites whose
`system:time_start` is midnight, not the acquisition. Matching reanalysis to
that timestamp assigns every pass identical midnight forcing. The extraction
script uses `Day_view_time` / `Night_view_time`, converted to UTC via the
domain's central longitude.

**Retrieval masking.** Loss and all metrics are masked to genuinely retrieved
cells. Interpolated cells enter the input sequence but are never scored.

---

## Repository Structure
```
Physics-Informed-UHI-Diagnostics/
│
├── README.md
├── requirements.txt
├── .gitattributes                      Git LFS tracking
├── .gitignore
│
├── data/                               extracted inputs (LFS)
│   ├── Blr_Nodes_1km.geojson           1 km nodes + building morphology
│   ├── Hyd_Nodes_1km.geojson
│   ├── Dlh_Nodes_1km.geojson
│   └── {Blr,Hyd,Dlh}_{Summer,Winter}_{April,December}.csv
│
├── src/
│   ├── gee/
│   │   ├── nodes_1km.js                node grid + Open Buildings morphology
│   │   └── modis_extraction.js         night LST, overpass-time reanalysis pairing
│   └── notebooks/
│       ├── 01_main.ipynb               pipeline reproducing the results
│       ├── 02_supplementary.ipynb      semivariogram, k sweep, spatial-lag test
│       └── 03_superseded_design.ipynb
│
└── results/
    ├── modis_night_results_FINAL.csv   main sweep, 4 arms x 6 configs
    ├── oracle_and_decomposition.csv    structureless reference + tree decomposition
    ├── neural_decomposition.csv        graph decomposition + signed level error
    ├── bootstrap_gaps.csv              CIs on the tree-vs-graph gaps
    ├── tree_baselines.csv              RF / HistGBR, with and without spatial lag
    ├── k_focused.csv                   message-passing degree sweep
    ├── mlp_lag_test.csv                spatial lag vs message passing
    ├── plots.py                        generates the two figures
    ├── fig1_ablation.pdf
    ├── fig2_variance_deficit.pdf
    └── {Bangalore,Hyderabad,Delhi}/
        ├── {April,December}_Models/    checkpoints: Full, NoGraph, NoMorph, NoAtmos
        └── *.png                       per-arm city maps, collapse, correlation,
                                        scatter + histogram
```

---

## Reproducing

```bash
git clone https://github.com/AyushG-1210/Physics-Informed-UHI-Diagnostics.git
cd Physics-Informed-UHI-Diagnostics
```

1. Run the two Earth Engine scripts in `gee/`. They produce six MODIS CSVs and
   three node GeoJSONs.
2. Set `DATA_ROOT` in section 2 of `01_main.ipynb` to the directory holding
   those nine files. Outside Colab, remove the `drive.mount` call.
3. Run `01_main.ipynb` top to bottom.

Sections 1–9 build the data, models and main sweep; 10–13 produce the
decomposition and bootstrap tables. Section 8 is the expensive step — 24 model
fits, roughly 3 hours on a Colab T4 — and writes `modis_night_results.csv`
unconditionally, so back that file up before re-running.

### Environment

```bash
pip install \
  "numpy==1.26.4" \
  "torch==2.3.0+cu121" \
  "torch-geometric==2.5.3" \
  "torch-scatter==2.1.2+pt23cu121" \
  "torch-sparse==0.6.18+pt23cu121" \
  "torch-cluster==1.6.3+pt23cu121" \
  "scikit-learn>=1.3" \
  "contextily==1.6.0" \
  "rasterio==1.3.10" \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

`numpy` is pinned below 2.0 because `ndarray.ptp()` was removed there. These
pins intentionally conflict with some Colab defaults; that is expected.

### Determinism

`set_seed()` fixes Python, NumPy and Torch seeds but does **not** make GPU runs
bit-reproducible — cuDNN kernel selection and non-deterministic reductions mean
small run-to-run differences remain. Reported numbers come from `seed=10`.

---

## Evaluation protocol

Each domain is divided into a 5 × 5 grid of spatial blocks assigned 60/15/25 to
train, validation and test. All time windows are available to every split.
Normalisation statistics come from training nodes only.

Random splits place neighbouring observations in both training and held-out
sets and inflate apparent skill, so blocked assignment is used throughout.

Spatial-lag features average neighbours' **predictors**, never their targets.
Predictor values are known everywhere at inference, so crossing block
boundaries is not leakage; borrowing neighbour targets would be.

Bootstrap intervals resample **nights** with replacement — nodes within a night
share that night's weather and are not independent. Both models are evaluated
on the same resampled nights each draw, so intervals are on the difference.

---

## Limitations

- Single spatial fold. Multi-seed runs elsewhere in the study varied within one
  standard deviation, but fold sensitivity is untested.
- No terrain variable. The residual correlates +0.544 with the neighbourhood
  mean of the target, consistent with cold-air drainage.
- Building morphology ends in 2023, two to three years before the target
  windows, with each annual layer centred on 30 June rather than on either
  season. Construction after 2023 is not represented.
- Heights are model-inferred from 10 m Sentinel-2, not measured. The product
  documentation notes false detections on features such as solar panels and
  limits on very small structures.
- The day–night contrast rests on one Landsat scene per configuration against
  78–108 MODIS night passes. The night result is robust; the daytime contrast
  is indicative.
- Coverage filtering retains 329–830 of 770–973 candidate nodes.
- All three cities are inland; coastal sea-breeze regimes are untested.

---

## Acknowledgements

Data from MODIS (NASA LP DAAC), ERA5-Land (Copernicus Climate Change Service),
Landsat (USGS), Google Open Buildings, and Sentinel-5P (ESA).