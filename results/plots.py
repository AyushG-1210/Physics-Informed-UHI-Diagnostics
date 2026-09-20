"""Figures 1 and 2 for the GRSL letter.

Fig 1  morphology ablation: spatial r collapses, RMSE does not
Fig 2  variance deficit: every model family sits below the 1:1 line

Both are single-column (3.5 in) vector PDFs.

Run from a directory containing:
    modis_night_results.csv          the neural sweep (cell 9)
    oracle_and_decomposition.csv     tree runs + structureless reference
    neural_decomposition.csv         optional; adds the Full arm to Fig 2
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# IEEE styling
# ---------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    # Type 42 embeds TrueType fonts. Matplotlib's PDF default is Type 3,
    # which IEEE PDF eXpress routinely rejects for non-embedded fonts.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COL = 3.5           # IEEE single column, inches
BLUE, ORANGE = "#0072B2", "#D55E00"     # Okabe-Ito, colourblind-safe
GREY = "#333333"

SEASON = {"April": "Pre", "December": "Post"}      # window -> season label
CITY = {"Bangalore": "BLR", "Hyderabad": "HYD", "Delhi": "DLH"}
ORDER = ["BLR Pre", "BLR Post", "HYD Pre", "HYD Post", "DLH Pre", "DLH Post"]


def label_from(city, month):
    """'Bangalore', 'April' -> 'BLR Pre'.

    The window labels in the data are month names but the paper describes
    them as pre- and post-monsoon. Mapping explicitly rather than by
    substring: a substring test for 'pre' matches neither month and
    silently labels everything 'Post'.
    """
    if month not in SEASON:
        raise KeyError(f"unmapped window {month!r}; add it to SEASON")
    return f"{CITY.get(city, city[:3].upper())} {SEASON[month]}"


# ---------------------------------------------------------------
# Figure 1 — morphology ablation
# ---------------------------------------------------------------
def figure1(path="modis_night_results_FINAL.csv", out="fig1_ablation.pdf"):
    df = pd.read_csv(path)
    df["cfg"] = [label_from(c, m) for c, m in zip(df["city"], df["month"])]

    full = df[df["arm"] == "Full"].set_index("cfg")
    nom = df[df["arm"] == "NoMorph"].set_index("cfg")
    for name, frame in (("Full", full), ("NoMorph", nom)):
        dupes = frame.index[frame.index.duplicated()].tolist()
        if dupes:
            raise ValueError(f"{name} arm has duplicate configs: {dupes}")
    cfgs = [c for c in ORDER if c in full.index and c in nom.index]
    full, nom = full.loc[cfgs], nom.loc[cfgs]

    x = np.arange(len(cfgs))
    w = 0.36

    fig, ax1 = plt.subplots(figsize=(COL, 2.4), layout="constrained")
    ax1.bar(x - w / 2, full["spatial_r_mean"], w, color=BLUE,
            label="Spatial $r$, full")
    ax1.bar(x + w / 2, nom["spatial_r_mean"], w, color=ORANGE, alpha=0.85,
            label="Spatial $r$, no morphology")
    ax1.set_ylabel("Spatial correlation $r$")
    ax1.set_ylim(0, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cfgs, rotation=25, ha="right")
    ax1.tick_params(axis="y", labelcolor=BLUE)

    ax2 = ax1.twinx()
    ax2.plot(x, full["rmse"], color=GREY, marker="o", ms=3.5,
             label="RMSE, full")
    ax2.plot(x, nom["rmse"], color=GREY, marker="s", ms=3.5, ls="--",
             label="RMSE, no morphology")
    ax2.set_ylabel(r"RMSE ($^\circ$C)", color=GREY)
    lo = min(full["rmse"].min(), nom["rmse"].min())
    hi = max(full["rmse"].max(), nom["rmse"].max())
    pad = max(0.25, (hi - lo) * 1.5)
    ax2.set_ylim(lo - pad, hi + pad)     # keeps the RMSE lines visibly flat

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", ncol=2, frameon=False,
               handlelength=1.4, columnspacing=1.0)

    fig.savefig(out, format="pdf")   # no tight bbox: keeps width at COL
    plt.close(fig)

    drop = (full["spatial_r_mean"] - nom["spatial_r_mean"])
    d_rmse = (nom["rmse"] - full["rmse"])
    print(f"  {out}")
    print(f"    spatial r drop: mean {drop.mean():.3f}, "
          f"range {drop.min():.3f}-{drop.max():.3f}")
    print(f"    RMSE change:    mean {d_rmse.mean():+.3f} degC, "
          f"max |shift| {d_rmse.abs().max():.3f}, "
          f"improved in {(d_rmse < 0).sum()} of {len(d_rmse)}")
    print("    ^ these are your [G8] numbers -- use them in the caption")


# ---------------------------------------------------------------
# Figure 2 — variance deficit
# ---------------------------------------------------------------
def figure2(out="fig2_variance_deficit.pdf"):
    """Pools every model family. The spread ratios quoted in the letter
    come from the tree runs, so the neural sweep alone cannot carry this
    figure."""
    pts = []

    if os.path.exists("modis_night_results_FINAL.csv"):
        d = pd.read_csv("modis_night_results_FINAL.csv")
        d = d[d["arm"] == "Full"]
        pts.append(pd.DataFrame({
            "obs": d["actual_spatial_sd"], "pred": d["pred_spatial_sd"],
            "family": "Graph network"}))

    if os.path.exists("grsl_supp/oracle_and_decomposition.csv"):
        d = pd.read_csv("grsl_supp/oracle_and_decomposition.csv")
        d = d[d["model"] != "oracle (level only)"].copy()
        d["family"] = np.where(d["model"].str.startswith("RandomForest"),
                               "Random forest", "Gradient boosting")
        pts.append(pd.DataFrame({
            "obs": d["sd_obs"], "pred": d["sd_pred"], "family": d["family"]}))

    if os.path.exists("grsl_supp/neural_decomposition.csv"):
        d = pd.read_csv("grsl_supp/neural_decomposition.csv")
        pts.append(pd.DataFrame({
            "obs": d["sd_obs"], "pred": d["sd_pred"],
            "family": "Graph network"}))

    if not pts:
        raise FileNotFoundError("no results CSVs found")
    P = pd.concat(pts, ignore_index=True).dropna()

    fig, ax = plt.subplots(figsize=(COL, 2.6), layout="constrained")
    styles = {"Random forest": ("o", BLUE),
              "Gradient boosting": ("s", ORANGE),
              "Graph network": ("^", GREY)}
    for fam, (mk, c) in styles.items():
        sub = P[P["family"] == fam]
        if len(sub):
            ax.scatter(sub["obs"], sub["pred"], marker=mk, s=22,
                       facecolors="none", edgecolors=c, linewidths=0.9,
                       label=fam)

    lo = min(P["obs"].min(), P["pred"].min()) - 0.08
    hi = max(P["obs"].max(), P["pred"].max()) + 0.08
    ax.plot([lo, hi], [lo, hi], color="k", ls="--", lw=0.8, alpha=0.6,
            label="1:1")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_box_aspect(1)          # square axes, figure stays COL wide
    ax.set_xlabel(r"Observed spatial SD ($^\circ$C)")
    ax.set_ylabel(r"Predicted spatial SD ($^\circ$C)")
    ax.legend(frameon=False, loc="upper left", handlelength=1.2)

    fig.savefig(out, format="pdf")   # no tight bbox: keeps width at COL
    plt.close(fig)

    r = (P["pred"] / P["obs"])
    print(f"  {out}")
    print(f"    {len(P)} points, {P['family'].nunique()} families")
    print(f"    spread ratio {r.min():.3f}-{r.max():.3f}, "
          f"{(r < 1).sum()} of {len(r)} below 1:1")


if __name__ == "__main__":
    figure1()
    figure2()
    print("\nCaptions must stand alone and must not restate the axes.")
    print("Use the printed numbers above -- do not quote a shift you")
    print("have not measured.")

    '''
        spatial r drop: mean 0.160, range 0.086-0.287
    RMSE change:    mean -0.060 degC, max |shift| 0.148, improved in 4 of 6
    ^ these are your [G8] numbers -- use them in the caption
  fig2_variance_deficit.pdf
    36 points, 3 families
    spread ratio 0.615-0.904, 36 of 36 below 1:1

Captions must stand alone and must not restate the axes.
Use the printed numbers above -- do not quote a shift you
have not measured.
'''