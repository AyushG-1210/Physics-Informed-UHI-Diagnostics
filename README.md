# Separating Level and Spatial Skill in Satellite Nocturnal Urban Heat Estimation

> Ayush Gouda, Aditya Prakash, Hema M S

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-orange)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.5.3-green)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code and data for a letter submitted to *IEEE Geoscience and Remote Sensing Letters* (2026).

---

## Overview

Nocturnal urban land surface temperature (LST) models are usually compared by
pooled RMSE. This study asks what that number actually measures.

Using MODIS Terra and Aqua night LST over Bengaluru, Hyderabad and Delhi in two
seasonal windows, each model's error is split, pass by pass, into a citywide
*level* component and an intra-urban *anomaly* component:

```
MSE = MSE_level + MSE_anomaly
```

The cross-term vanishes because spatial anomalies sum to zero within a pass, so
the split is exact. The citywide level swings more from night to night than
temperature varies within a city, so level tracking can dominate RMSE even
though it says nothing about intra-urban skill.

The study is about **evaluation**, not graph versus tree architectures. The two
models (a random forest and a graph attention network) are vehicles for showing
how evaluation design moves the two error components.

---

## Findings

**RMSE gaps overstate spatial gaps.** When test nights are seen in training
(same-night split), the forest has lower anomaly error than the GAT in all six
city–season configurations, with night-bootstrap intervals excluding zero. Yet
the RMSE gaps (0.46–1.09 °C) are 2.7–6.8 times the anomaly gaps (0.10–0.22 °C)
and do not follow them in size or order (Spearman ρ = 0.03, n = 6, descriptive).

**One normalization layer moves RMSE, not spatial skill.** Removing PairNorm
from the GAT cuts its RMSE by 0.23–0.58 °C while anomaly error changes by at
most 0.09 °C. The forest keeps lower anomaly error either way.

**Unseen nights change the picture.** Withholding test nights from training
raises the forest's level error from 0.16–0.48 °C to 1.06–2.53 °C. Level error
alone then exceeds the observed spatial variability for every model, and the
model RMSE favors shifts with routine GAT training settings while the spatial
ranking largely holds.

**Recommendations.** Studies comparing intra-urban LST models should report
(i) level and anomaly errors separately, (ii) spatial skill (R²_sp) against an
oracle-level reference, and (iii) whether test nights were seen in training.

### Error decomposition (°C)

| Split | Config | RF RMSE | RF level | RF anom. | GAT RMSE | GAT level | GAT anom. | Ref. RMSE |
|---|---|---|---|---|---|---|---|---|
| Same-night | BLR-Pre  | 0.635 | 0.212 | 0.599 | 1.396 | 1.140 | 0.805 | 0.845 |
| Same-night | BLR-Post | 0.854 | 0.475 | 0.709 | 1.554 | 1.324 | 0.812 | 0.853 |
| Same-night | HYD-Pre  | 0.694 | 0.329 | 0.612 | 1.270 | 0.962 | 0.829 | 0.951 |
| Same-night | HYD-Post | 0.722 | 0.403 | 0.599 | 1.311 | 1.095 | 0.721 | 0.985 |
| Same-night | DLH-Pre  | 0.660 | 0.167 | 0.639 | 1.754 | 1.543 | 0.834 | 1.158 |
| Same-night | DLH-Post | 0.581 | 0.160 | 0.558 | 1.045 | 0.757 | 0.721 | 1.043 |
| Unseen-night | BLR-Pre  | 2.661 | 2.436 | 1.071 | 2.308 | 2.067 | 1.028 | 1.106 |
| Unseen-night | BLR-Post | 1.617 | 1.489 | 0.630 | 1.653 | 1.513 | 0.664 | 0.702 |
| Unseen-night | HYD-Pre  | 1.693 | 1.522 | 0.741 | 1.604 | 1.405 | 0.775 | 0.853 |
| Unseen-night | HYD-Post | 1.311 | 1.064 | 0.766 | 1.692 | 1.451 | 0.869 | 1.012 |
| Unseen-night | DLH-Pre  | 2.680 | 2.535 | 0.870 | 2.856 | 2.682 | 0.981 | 1.188 |
| Unseen-night | DLH-Post | 1.361 | 1.258 | 0.520 | 1.430 | 1.285 | 0.628 | 0.922 |

RMSE² = level² + anomaly², exactly. The oracle-level reference assigns each
pass's observed citywide mean to every test node. It is not deployable; its
RMSE equals the observed spatial standard deviation and defines spatial skill
as R²_sp = 1 − MSE_anomaly / MSE_ref. Numbers are for one test-block
assignment; rotated assignments (`table1_all_folds.csv`) keep the same-night
spatial ranking.

---

## Data

| Feature | Source | Role |
|---|---|---|
| Night LST (MOD11A1, MYD11A1) | MODIS Terra + Aqua | **target**, 1 km |
| t2m, u10, v10, ssr, str, slhf | ERA5-Land, hourly at overpass time | atmospheric predictors |
| NDVI, NDBI | Landsat | static surface predictors |
| Building height, building cover | Google Open Buildings 2.5D Temporal V1 | static morphology predictors |
| Tropospheric NO₂ | Sentinel-5P TROPOMI, period mean | static anthropogenic proxy |

**Windows.** Files are named by month. `April` is the pre-monsoon window
(1 March–30 April 2025) and `December` is the post-monsoon window
(15 November 2025–15 January 2026).

**Passes.** Terra (≈22:30 local) and Aqua (≈01:30 local) are treated as
separate passes. Their citywide levels differ by 2.5–3.0 °C. All clear-sky
retrievals are kept, without quality-flag or view-angle filtering.

**Nodes.** 1 km grid in EPSG:32643, matched to the MODIS resolution. Sampling
finer than the target manufactures apparent spatial skill, because a model can
then recover which source pixel a node occupies rather than any property of the
surface.

| City | Candidate nodes | Retained (pre / post) | Night passes (pre / post) |
|---|---|---|---|
| Bengaluru | 973 | 329 / 326 | 78 / 85 |
| Hyderabad | 770 | 477 / 480 | 92 / 104 |
| Delhi | 873 | 830 / 826 | 108 / 105 |

Retention is after a 60% coverage filter.

**LST is never a predictor.** With lagged target values available, persistence
carries the prediction and the surface predictors go unused.

### Two data-handling details that matter

**Overpass timestamps.** MOD11A1 and MYD11A1 are daily composites whose
`system:time_start` is midnight, not the acquisition. Matching reanalysis to
that timestamp assigns every pass identical midnight forcing. The extraction
script uses `Day_view_time` / `Night_view_time`, converted to UTC via the
domain's central longitude.

**Retrieval masking.** Training loss and all metrics are masked to genuinely
retrieved cells. Interpolated cells enter the input sequence but are never
trained on or scored.

---

## Models and evaluation

**Splits.** Each domain is divided into a 5 × 5 grid of spatial blocks,
assigned roughly 60/15/25 to train, validation and test. Normalisation
statistics come from training nodes only.

- *Same-night split:* test locations are withheld, but every night appears in
  training (the usual spatial gap-filling setup).
- *Unseen-night split:* contiguous four-night blocks are also withheld, so test
  nodes share neither location nor timing with training data
  (leave-location-and-time-out).

**Random forest.** 300 trees, minimum leaf size tuned on validation blocks,
with 8-neighbour averages of the static **predictors**. Neighbour targets are
never used: predictor values are known everywhere at inference, so crossing
block boundaries is not leakage, while borrowing neighbour targets would be.

**GAT.** Graph attention network with skip connections, PairNorm, an LSTM over
the 5-pass predictor window, separate heads for atmospheric and static inputs,
and an 8-neighbour graph restricted to downwind directions. Huber loss plus
advection-smoothness and building/vegetation margin penalties. Settings were
fixed in advance (learning rate 8×10⁻⁴, loss weights 1.0 and 0.03), apart from
a sensitivity sweep over learning rate and regularization.

**Decomposition.** Computed per pass and pooled with node-count weights.
Passes with fewer than five valid test nodes are excluded.

**Bootstrap.** Intervals resample calendar **nights** with replacement, keeping
a night's Terra and Aqua passes together (2000 draws, 95% percentile). Both
models are scored on the same resampled nights each draw, so intervals are on
the difference.

---

## Repository structure

```
├── README.md
├── LICENSE
├── requirements.txt
├── .gitattributes                         Git LFS tracking
├── .gitignore
│
├── data/                                  extracted inputs (LFS)
│   ├── {Blr,Hyd,Dlh}_Nodes_1km.geojson    1 km nodes + building morphology
│   └── {Blr,Hyd,Dlh}_{April,December}_MODIS_60d.csv
│
├── src/
│   ├── gee/
│   │   ├── nodes_1km.js                   node grid + Open Buildings morphology
│   │   └── modis_extraction.js            night LST, overpass-time reanalysis pairing
│   └── notebooks/
│       ├── 04_final_run.ipynb             the letter's pipeline (start here)
│       ├── 01_main.ipynb                  earlier exploratory sweep (see below)
│       ├── 02_supplementary.ipynb         semivariogram, k sweep, spatial-lag test
│       └── 03_superseded_design.ipynb     superseded design, kept for reference
│
└── results/
    ├── table1_all_folds.csv               level/anomaly decomposition, all test-block assignments
    ├── bootstrap_by_night.csv             night-bootstrap intervals on model gaps
    ├── gat_diagnostics.csv                PairNorm ablation and GAT level-error diagnostics
    ├── gat_sensitivity.csv                GAT settings sweep (Table II)
    ├── per_sensor.csv                     Terra and Aqua analysed separately
    ├── spread.csv                         predicted vs observed spatial spread
    ├── data_checks.csv                    pass counts, node retention, quality-flag fractions
    ├── ablation_static.csv                static-predictor ablation (not reported in the letter)
    ├── run_log.csv                        run metadata
    ├── paper_numbers.txt                  every number quoted in the letter
    ├── fig_decoupling_{spatial,spatiotemporal}.{pdf,png}
    │                                      RMSE-gap vs anomaly-gap plots for the two splits
    ├── preds/                             saved model predictions
    ├── context/                           saved per-pass context for the analysis stage
    └── {Bangalore,Hyderabad,Delhi}/       earlier exploratory sweep (see below)
        ├── {April,December}_Models/       checkpoints: Full, NoGraph, NoMorph, NoAtmos
        └── *.png                          per-arm city maps, collapse, correlation,
                                           scatter + histogram
```

---

## Reproducing

```bash
git clone https://github.com/AyushG-1210/Physics-Informed-UHI-Diagnostics.git
cd Physics-Informed-UHI-Diagnostics
git lfs pull
```

1. The nine files in `data/` are included. To regenerate them, run the two
   Earth Engine scripts in `src/gee/`.
2. Open `src/notebooks/04_final_run.ipynb` and set the data path. Outside
   Colab, remove the `drive.mount` call.
3. `SMOKE_TEST` defaults to `True` for a quick check. Set it to `False` for the
   full run. `RUN_STAGE_B = False` skips the extra test-block assignments.
4. Run top to bottom. The notebook is resumable and writes the files in
   `results/`. Compare against `paper_numbers.txt`.

`results/preds/` and `results/context/` hold the saved predictions and context,
so the analysis stage (decomposition, bootstrap, tables) can be rerun without
retraining any model. `04_final_run.ipynb` writes numbers and CSVs rather than
plots. For model maps and diagnostic plots, see the earlier sweep below.

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
pins intentionally conflict with some Colab defaults.

### Determinism

`set_seed()` fixes Python, NumPy and Torch seeds but does **not** make GPU runs
bit-reproducible. cuDNN kernel selection and non-deterministic reductions leave
small run-to-run differences. Reported numbers come from `seed=10`.

---

## Earlier exploratory work

Notebooks `01`–`03` and the per-city folders under `results/` are from the
study's first pass. They run four ablation arms per configuration (Full,
NoGraph, NoMorph, NoAtmos) and produce prediction maps, scatter and histogram
plots, correlation plots and variance-collapse diagnostics, plus checkpoints.
They are kept as a code and visualization reference.

**They are not the source of any number in the letter.** They predate the final
run, which fixed a bug in the GAT loss masking (the loss had been trained on
test-node targets), so their absolute numbers are superseded and will not match
the tables above. Use `04_final_run.ipynb` and the top-level files in `results/`
for anything you intend to cite.

---