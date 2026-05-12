# -*- coding: utf-8 -*-
"""
Generate revised Figures 2–6 for the NIR-based automated textile sorting paper.

This version deliberately moves away from repeated horizontal bar-chart panels.
It uses overlap, nesting, matrix, flow, and archetype visuals to make the sorting
paper visually and analytically distinct from the disruptor paper.

Expected input files from 7_rule_evaluation_and_sensitivity.py:
- 7_rule_flags_row_level.csv
- 7_rule_summary.csv
- 7_any_barrier_summary.csv
- 7_rule_diagnostic_material.csv
- 7_rule_diagnostic_category.csv

Main visual logic
-----------------
Figure 2: rule-specific violation shares, sensitivity, and baseline intersections
Figure 3: supported recognition scope and blend-complexity diagnostics
Figure 4: minor-component detectability and black-colour proxy diagnostics
Figure 5: concealed structures and S5 surface-hidden material mismatch
Figure 6: category-level profiles of sorting-relevant design barriers

Notes
-----
- Outputs are saved as PNG and PDF.
- PDF font type is set to 42 so text remains editable in vector-editing software.
- The script avoids seaborn and uses only pandas, numpy and matplotlib.
"""

from pathlib import Path
import re
import textwrap
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.colors import LinearSegmentedColormap

# =============================================================================
# 0. User settings
# =============================================================================

DATA_DIR = Path("rule_eval_outputs")
# Allow the script to run either from the project root with rule_eval_outputs/
# or directly from a folder containing the CSV outputs.
if not (DATA_DIR / "7_rule_flags_row_level.csv").exists():
    DATA_DIR = Path(".")

OUT_DIR = Path("figures_sorting_v9")
OUT_DIR.mkdir(exist_ok=True)

DPI = 400
TOP_N = 10
MIN_CATEGORY_N = 100

# Distinct from disruptor-paper orange/blue emphasis: use a restrained purple/teal logic.
PURPLE = "#8d79c6"
PURPLE_DARK = "#5f4a9b"
PURPLE_LIGHT = "#d9d1ef"
TEAL = "#67b8a9"
TEAL_DARK = "#2f8378"
TEAL_LIGHT = "#d6eee9"
GOLD = "#e0b45b"
GOLD_LIGHT = "#f4dfad"
GREY = "#d8d8d8"
GREY_DARK = "#6b6b6b"
LIGHT_GREY = "#f2f2f2"
TEXT = "#303030"
RED = "#c96a5d"

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "DejaVu Sans"

# =============================================================================
# 1. Load data
# =============================================================================

row = pd.read_csv(DATA_DIR / "7_rule_flags_row_level.csv")
summary = pd.read_csv(DATA_DIR / "7_rule_summary.csv")
any_barrier = pd.read_csv(DATA_DIR / "7_any_barrier_summary.csv")
diag_material = pd.read_csv(DATA_DIR / "7_rule_diagnostic_material.csv")
diag_category = pd.read_csv(DATA_DIR / "7_rule_diagnostic_category.csv")

N_TOTAL = len(row)

RULE_COLS = {
    "S1": "s1_violate_central",
    "S2": "s2_violate",
    "S3": "s3_violate_central",
    "S4": "s4_violate_central",
    "S5": "s5_violate_central",
}
RULE_LABELS = {
    "S1": "S1\nrecognition\nscope",
    "S2": "S2\n>2 fibres",
    "S3": "S3\nminor\ncomponent",
    "S4": "S4\nblack\nproxy",
    "S5": "S5\nsurface–hidden\nmismatch",
}
RULE_SHORT = {
    "S1": "Recognition scope",
    "S2": "Blend complexity",
    "S3": "Minor component",
    "S4": "Black-colour proxy",
    "S5": "Surface–hidden mismatch",
}

# Ensure rule columns are boolean-like for calculations.
for c in RULE_COLS.values():
    row[c] = row[c].astype(bool)
for c in ["s1_violate_conservative", "s1_violate_expanded", "s3_violate_conservative",
          "s3_violate_expanded", "s4_violate_expanded", "s5_violate_conservative",
          "s5_violate_expanded", "any_barrier_conservative", "any_barrier_central",
          "any_barrier_expanded"]:
    if c in row.columns:
        row[c] = row[c].astype(bool)

# =============================================================================
# 2. Helpers
# =============================================================================

def pct(x):
    return 100 * float(x)


def wrap(s, width=24):
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False))


def clean_label(s):
    return str(s).replace("_", " ")


def save_fig(fig, name):
    png = OUT_DIR / f"{name}.png"
    pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def panel_label(ax, label, x=-0.08, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="top", ha="left", color=TEXT)


def panel_title(ax, label, title, y=1.07, fontsize=10.5):
    """Place panel letter and panel title as one left-aligned text block."""
    ax.text(
        0.0, y, f"{label} {title}",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=fontsize, color=TEXT
    )


def clean_ax(ax, grid_axis=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#aaaaaa")
    ax.spines["bottom"].set_color("#aaaaaa")
    ax.tick_params(axis="both", colors=TEXT, labelsize=8)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#e8e8e8", linewidth=0.7)
        ax.set_axisbelow(True)


def share_from_summary(rule, setting):
    m = (summary["rule"] == rule) & (summary["setting"] == setting)
    if not m.any():
        return np.nan
    return float(summary.loc[m, "share_violate"].iloc[0])


def any_share(summary_type):
    m = any_barrier["summary_type"] == summary_type
    return float(any_barrier.loc[m, "share_with_any_barrier"].iloc[0])


def readable_sig(s):
    s = str(s)
    s = s.replace("_", " ").replace("+", " + ")
    s = s.replace("__", " → ")
    return s


def parse_s5_signature(sig):
    """Return surface_component, surface_material, hidden_component, hidden_material."""
    sig = str(sig)
    parts = sig.split("__")
    if len(parts) != 2:
        return None, None, None, None

    def parse_part(part):
        m = re.match(r"(.+?)\[(.+?)\]", part)
        if m:
            comp = m.group(1).replace("_", " ")
            mat = m.group(2).replace("_", " ").replace("+", " + ")
            return comp, mat
        return part.replace("_", " "), ""

    s_comp, s_mat = parse_part(parts[0])
    h_comp, h_mat = parse_part(parts[1])
    return s_comp, s_mat, h_comp, h_mat


def draw_rect_split(ax, labels, values, colors, x=0, y=0, w=1, h=1, horizontal=True,
                    min_label_share=0.055, fontsize=8):
    """Simple treemap-like one-level rectangle split."""
    vals = np.array(values, dtype=float)
    total = vals.sum()
    if total <= 0:
        return
    shares = vals / total
    cur = x
    cury = y
    for lab, val, share, color in zip(labels, vals, shares, colors):
        if horizontal:
            ww = w * share
            rect = Rectangle((cur, y), ww, h, facecolor=color, edgecolor="white", linewidth=1.2)
            ax.add_patch(rect)
            cx, cy = cur + ww / 2, y + h / 2
            cur += ww
        else:
            hh = h * share
            rect = Rectangle((x, cury), w, hh, facecolor=color, edgecolor="white", linewidth=1.2)
            ax.add_patch(rect)
            cx, cy = x + w / 2, cury + hh / 2
            cury += hh
        if share >= min_label_share:
            ax.text(cx, cy, f"{lab}\n{share*100:.1f}%", ha="center", va="center",
                    fontsize=fontsize, color=TEXT)
        else:
            ax.text(cx, cy, f"{share*100:.1f}%", ha="center", va="center",
                    fontsize=max(6, fontsize-1), color=TEXT)


def draw_bezier_flow(ax, x0, y0, x1, y1, width, color, alpha=0.55):
    verts = [
        (x0, y0),
        (x0 + (x1-x0)*0.45, y0),
        (x0 + (x1-x0)*0.55, y1),
        (x1, y1),
    ]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    path = MplPath(verts, codes)
    patch = PathPatch(path, facecolor="none", edgecolor=color, lw=width,
                      alpha=alpha, capstyle="round")
    ax.add_patch(patch)


def top_material_diag(rule, scenario, n=TOP_N):
    d = diag_material[(diag_material["rule"] == rule) & (diag_material["scenario"] == scenario)].copy()
    return d.sort_values("n", ascending=False).head(n)


def top_category_diag(rule, scenario, n=TOP_N):
    d = diag_category[(diag_category["rule"] == rule) & (diag_category["scenario"] == scenario)].copy()
    return d.sort_values("n", ascending=False).head(n)


def rule_sensitivity_table():
    """Return plotting data for the unified overview sensitivity chart."""
    return pd.DataFrame([
        {
            "label": "Any barrier",
            "family": "Aggregate",
            "low": any_share("Lower-bound any-barrier"),
            "base": any_share("Baseline any-barrier"),
            "high": any_share("Upper-bound any-barrier"),
            "low_text": "lower",
            "base_text": "baseline",
            "high_text": "upper",
            "color": PURPLE_DARK,
        },
        {
            "label": "S1 supported recognition scope",
            "family": "Composition-recognition",
            "low": share_from_summary("S1", "Extended library"),
            "base": share_from_summary("S1", "Baseline library"),
            "high": share_from_summary("S1", "Restricted library"),
            "low_text": "extended",
            "base_text": "baseline",
            "high_text": "restricted",
            "color": TEAL_DARK,
        },
        {
            "label": "S2 blend complexity",
            "family": "Composition-recognition",
            "low": share_from_summary("S2", "Fixed"),
            "base": share_from_summary("S2", "Fixed"),
            "high": share_from_summary("S2", "Fixed"),
            "low_text": "fixed",
            "base_text": "fixed",
            "high_text": "fixed",
            "color": TEAL_DARK,
        },
        {
            "label": "S3 minor component",
            "family": "Composition-recognition",
            "low": share_from_summary("S3", "3% threshold"),
            "base": share_from_summary("S3", "5% threshold"),
            "high": share_from_summary("S3", "10% threshold"),
            "low_text": "3%",
            "base_text": "5%",
            "high_text": "10%",
            "color": TEAL_DARK,
        },
        {
            "label": "S4 black-colour proxy",
            "family": "Optical/surface signal",
            "low": share_from_summary("S4", "Exact black"),
            "base": share_from_summary("S4", "Exact black"),
            "high": share_from_summary("S4", "Contains 'black'"),
            "low_text": "exact black",
            "base_text": "baseline exact",
            "high_text": "contains “black”",
            "color": GREY_DARK,
        },
        {
            "label": "S5 surface–hidden mismatch",
            "family": "Garment construction",
            "low": share_from_summary("S5", ">5% concealed-material mismatch"),
            "base": share_from_summary("S5", ">5% concealed-material mismatch"),
            "high": share_from_summary("S5", "Any hidden layer"),
            "low_text": ">5 wt% mismatch",
            "base_text": "baseline >5 wt%",
            "high_text": "any hidden layer",
            "color": PURPLE,
        },
    ])


def draw_unified_sensitivity(ax, sens, xmax=68):
    """Draw a unified range-and-baseline chart for all rules."""
    sens = sens.copy()
    sens["y"] = np.arange(len(sens))[::-1]
    for _, r in sens.iterrows():
        y = r["y"]
        # Thin grey range line for alternative rule definitions.
        ax.plot([r["low"] * 100, r["high"] * 100], [y, y], color=GREY, lw=3.2,
                solid_capstyle="round", zorder=1)
        # Endpoints.
        ax.scatter([r["low"] * 100, r["high"] * 100], [y, y], s=45,
                   color="#c9c9c9", edgecolor="white", linewidth=0.7, zorder=2)
        # Baseline/fixed point.
        ax.scatter(r["base"] * 100, y, s=96, color=r["color"],
                   edgecolor="white", linewidth=0.9, zorder=3)
        ax.text(r["base"] * 100 + 1.0, y, f"{r['base'] * 100:.1f}%",
                ha="left", va="center", fontsize=8.0, color=TEXT)
        # Endpoint labels only when the range is visually meaningful.
        # When the lower endpoint equals the baseline (S4/S5), offset labels
        # horizontally to avoid the common exact/baseline label collision.
        if abs(r["high"] - r["low"]) > 0.015:
            if abs(r["low"] - r["base"]) < 1e-9:
                ax.text(r["base"] * 100 - 0.8, y + 0.26, f"{r['low_text']}\n{r['low']*100:.1f}%",
                        ha="right", va="bottom", fontsize=6.5, color=GREY_DARK)
                ax.text(r["high"] * 100 + 0.8, y + 0.26, f"{r['high_text']}\n{r['high']*100:.1f}%",
                        ha="left", va="bottom", fontsize=6.5, color=GREY_DARK)
            else:
                ax.text(r["low"] * 100, y + 0.26, f"{r['low_text']}\n{r['low']*100:.1f}%",
                        ha="center", va="bottom", fontsize=6.5, color=GREY_DARK)
                ax.text(r["high"] * 100, y + 0.26, f"{r['high_text']}\n{r['high']*100:.1f}%",
                        ha="center", va="bottom", fontsize=6.5, color=GREY_DARK)
        else:
            ax.text(r["base"] * 100, y + 0.26, r["base_text"],
                    ha="center", va="bottom", fontsize=6.5, color=GREY_DARK)

    ax.set_yticks(sens["y"])
    ax.set_yticklabels(sens["label"], fontsize=8.0)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.65, len(sens) - 0.30)
    ax.set_xlabel("Share of garment variants flagged (%)", fontsize=8, color=TEXT)
    clean_ax(ax, grid_axis="x")



# =============================================================================
# Derived data used across figures
# =============================================================================

baseline_flags = row[list(RULE_COLS.values())].astype(int)
baseline_flags.columns = list(RULE_COLS.keys())
row["n_barriers_baseline"] = baseline_flags.sum(axis=1)
row["baseline_pattern"] = baseline_flags.apply(
    lambda r: " + ".join([k for k in RULE_COLS if r[k] == 1]) if r.sum() else "None",
    axis=1,
)


# =============================================================================
# Figure 2: Unified overview of rule prevalence, sensitivity, and overlap
# =============================================================================

fig = plt.figure(figsize=(15.4, 7.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 1.38], wspace=0.44)

# Panel (a): unified rule-specific sensitivity chart
ax = fig.add_subplot(gs[0, 0])
sens = rule_sensitivity_table()
draw_unified_sensitivity(ax, sens, xmax=68)
panel_title(ax, "(a)", "Rule-specific violation shares and scenario sensitivity", y=1.04, fontsize=10.8)
ax.text(
    0.0, -0.16,
    "Grey endpoints indicate alternative rule definitions; coloured dots indicate baseline or fixed rule specifications.\n"
    "The aggregate any-barrier range combines boundary specifications across rules and is not the sum of individual rows.",
    transform=ax.transAxes,
    fontsize=7.1,
    color=GREY_DARK,
    ha="left",
    va="top",
)

# Panel (b): custom UpSet plot for top baseline intersections
patterns = row.loc[row["baseline_pattern"] != "None", "baseline_pattern"].value_counts().head(10)
pattern_names = list(patterns.index)
pattern_counts = patterns.values
x = np.arange(len(pattern_names))

sub = gs[0, 1].subgridspec(2, 1, height_ratios=[0.70, 0.48], hspace=0.04)
ax_bar = fig.add_subplot(sub[0])
ax_bar.vlines(x, 0, pattern_counts, color=TEAL_DARK, lw=8, alpha=0.90)
for i, c in enumerate(pattern_counts):
    ax_bar.text(i, c + max(pattern_counts) * 0.035, f"{c:,}", ha="center", va="bottom", fontsize=7.2, rotation=90)
ax_bar.set_ylabel("Variants", fontsize=8, color=TEXT)
ax_bar.set_xticks([])
panel_title(ax_bar, "(b)", "Top baseline rule-intersection patterns", y=1.12, fontsize=10.8)
clean_ax(ax_bar, grid_axis="y")
ax_bar.set_xlim(-0.6, len(x) - 0.4)
ax_bar.set_ylim(0, max(pattern_counts) * 1.25)


ax_mat = fig.add_subplot(sub[1], sharex=ax_bar)
rule_order = ["S1", "S2", "S3", "S4", "S5"]
ypos = np.arange(len(rule_order))[::-1]
for i, pat in enumerate(pattern_names):
    active = pat.split(" + ")
    active_y = []
    for j, rule in enumerate(rule_order):
        y = ypos[j]
        is_active = rule in active
        ax_mat.scatter(
            i, y,
            s=70 if is_active else 32,
            color=TEAL_DARK if is_active else "#dddddd",
            edgecolor="white" if is_active else "#dddddd",
            zorder=3,
        )
        if is_active:
            active_y.append(y)
    if active_y:
        ax_mat.plot([i, i], [min(active_y), max(active_y)], color=TEAL_DARK, lw=1.8, zorder=2)
ax_mat.set_yticks(ypos)
ax_mat.set_yticklabels([RULE_LABELS[r] for r in rule_order], fontsize=7.4)
ax_mat.set_xticks(x)
ax_mat.set_xticklabels([str(i + 1) for i in x], fontsize=7.5)
ax_mat.set_xlabel("Intersection rank", fontsize=8, color=TEXT)
ax_mat.set_xlim(-0.6, len(x) - 0.4)
ax_mat.set_ylim(-0.8, len(rule_order) - 0.2)
for spine in ax_mat.spines.values():
    spine.set_visible(False)
ax_mat.tick_params(axis="both", length=0, colors=TEXT)

fig.suptitle("Figure 2. Prevalence, sensitivity, and overlap of sorting-relevant design barriers", fontsize=12.3, y=0.995, color=TEXT)
save_fig(fig, "figure2_rule_sensitivity_overlap")
plt.close(fig)

# =============================================================================
# Figure 3: Recognition-scope mismatch, blend complexity, and material signatures
# =============================================================================

fig = plt.figure(figsize=(15.2, 9.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], height_ratios=[1.0, 1.08], hspace=0.58, wspace=0.44)

# Panel (a): decomposition of S1 baseline failures
ax = fig.add_subplot(gs[0, 0])
s1_fail = row[row["s1_violate_central"]].copy()
def s1_reason(r):
    sig = str(r.get("s1_signature_central", ""))
    n = int(r.get("n_distinct_fibres", 0))
    if sig == "no_material":
        return "No usable\nmaterial"
    if n == 1:
        return "Unsupported\nmono-material"
    if n == 2:
        return "Unsupported\nbinary blend"
    if n >= 3:
        return "More-than-two-fibre\ncomposition"
    return "Other"
reason_counts = s1_fail.apply(s1_reason, axis=1).value_counts()
reason_order = ["More-than-two-fibre\ncomposition", "Unsupported\nbinary blend", "Unsupported\nmono-material", "No usable\nmaterial"]
reason_counts = reason_counts.reindex(reason_order, fill_value=0)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Draw the decomposition manually so that small groups are labelled outside
# the bar rather than squeezed into unreadable narrow segments.
vals = reason_counts.values.astype(float)
shares = vals / vals.sum() if vals.sum() else vals
colors = [PURPLE_DARK, PURPLE, PURPLE_LIGHT, GREY]
x_start, y_bar, w_total, h_bar = 0.02, 0.22, 0.96, 0.56
x_cur = x_start
small_label_y = 0.84
small_callout_x = [0.83, 0.965]
small_i = 0
for lab, val, share, col in zip(reason_counts.index, vals, shares, colors):
    ww = w_total * share
    ax.add_patch(Rectangle((x_cur, y_bar), ww, h_bar, facecolor=col, edgecolor="white", linewidth=1.2))
    cx = x_cur + ww / 2
    if share >= 0.065:
        ax.text(cx, y_bar + h_bar / 2, f"{lab}\n{share*100:.1f}%",
                ha="center", va="center", fontsize=7.7, color=TEXT)
    else:
        # Callouts for small segments: show both category and percentage,
        # separated horizontally so labels do not overlap.
        label_map = {
            "Unsupported\nmono-material": "Unsupported mono-material",
            "No usable\nmaterial": "No usable material",
        }
        label = label_map.get(lab, lab.replace("\n", " "))
        tx = small_callout_x[min(small_i, len(small_callout_x) - 1)]
        small_i += 1
        ax.plot([cx, tx], [y_bar + h_bar, small_label_y - 0.025], color=GREY_DARK, lw=0.7)
        ax.text(tx, small_label_y, f"{label}\n{share*100:.1f}%",
                ha="center", va="bottom", fontsize=6.2, color=TEXT)
    x_cur += ww
ax.add_patch(Rectangle((x_start, y_bar), w_total, h_bar, fill=False, edgecolor="#bbbbbb", linewidth=0.8))
panel_title(ax, "(a)", "Decomposition of S1 supported recognition-scope violations", y=0.92, fontsize=10.5)
ax.text(0.02, 0.08, f"S1 baseline violations: {len(s1_fail):,} variants", ha="left", va="bottom", fontsize=7.5, color=GREY_DARK)

# Panel (b): S1 pass/fail mosaic by fibre count
ax = fig.add_subplot(gs[0, 1])
row["fibre_count_group"] = np.select(
    [row["n_distinct_fibres"] <= 1, row["n_distinct_fibres"] == 2, row["n_distinct_fibres"] >= 3],
    ["1 fibre", "2 fibres", "3+ fibres"], default="unknown"
)
group_order = ["1 fibre", "2 fibres", "3+ fibres"]
counts = row.groupby(["fibre_count_group", "s1_violate_central"]).size().unstack(fill_value=0).reindex(group_order).fillna(0)
x0 = 0.02
available_w = 0.96
for g in group_order:
    total = counts.loc[g].sum()
    w = available_w * total / len(row)
    fail = counts.loc[g, True] if True in counts.columns else 0
    fail_h = fail / total if total else 0
    ax.add_patch(Rectangle((x0, 0.15), w, 0.70*(1-fail_h), facecolor=TEAL_LIGHT, edgecolor="white", linewidth=1.1))
    ax.add_patch(Rectangle((x0, 0.15 + 0.70*(1-fail_h)), w, 0.70*fail_h, facecolor=PURPLE, edgecolor="white", linewidth=1.1))
    ax.add_patch(Rectangle((x0, 0.15), w, 0.70, fill=False, edgecolor="#bbbbbb", linewidth=0.7))
    ax.text(x0 + w/2, 0.08, f"{g}\n{int(total):,}", ha="center", va="top", fontsize=7.3, color=TEXT)
    if fail_h > 0.08:
        ax.text(x0 + w/2, 0.15 + 0.70*(1-fail_h) + 0.70*fail_h/2, f"violates S1\n{fail_h*100:.0f}%", ha="center", va="center", fontsize=7, color="white")
    if (1-fail_h) > 0.12:
        ax.text(x0 + w/2, 0.15 + 0.70*(1-fail_h)/2, f"passes S1\n{(1-fail_h)*100:.0f}%", ha="center", va="center", fontsize=7, color=TEXT)
    x0 += w
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
panel_title(ax, "(b)", "S1 rule outcome by readable fibre-count group", y=1.05, fontsize=10.5)

# Panel (c): lollipop list of top unsupported S1 baseline compositions
ax = fig.add_subplot(gs[1, 0])
s1_top = top_material_diag("S1", "Baseline library", n=8).copy()
s1_top["label"] = s1_top["material_diag"].apply(readable_sig)
s1_top = s1_top.sort_values("n", ascending=True)
y = np.arange(len(s1_top))
ax.hlines(y, 0, s1_top["n"], color="#e2e2e2", lw=1.5)
point_cols = [TEAL_DARK if i == len(s1_top) - 1 else GREY_DARK for i in range(len(s1_top))]
ax.scatter(s1_top["n"], y, s=62, color=point_cols, edgecolor="white", linewidth=0.8, zorder=3)
s1_denom = int(row["s1_violate_central"].sum())
for i, (_, r) in enumerate(s1_top.iterrows()):
    ax.text(
        r["n"] + s1_top["n"].max()*0.035, i,
        f"{int(r['n']):,} ({r['n']/s1_denom*100:.1f}%)",
        va="center", ha="left", fontsize=7.0, color=TEXT,
    )
ax.set_yticks(y)
ax.set_yticklabels([wrap(v, 31) for v in s1_top["label"]], fontsize=7.2)
ax.set_xlabel("Count among S1 baseline violations", fontsize=8, color=TEXT)
panel_title(ax, "(c)", "Top unsupported readable compositions under S1 baseline", y=1.04, fontsize=10.5)
ax.set_xlim(0, s1_top["n"].max()*1.48)
ax.margins(x=0)
clean_ax(ax, grid_axis="x")

# Panel (d): lollipop list of top S2 more-than-two-fibre compositions
ax = fig.add_subplot(gs[1, 1])
s2_top = top_material_diag("S2", "Fixed", n=8).copy()
s2_top["label"] = s2_top["material_diag"].apply(readable_sig)
s2_top = s2_top.sort_values("n", ascending=True)
y = np.arange(len(s2_top))
ax.hlines(y, 0, s2_top["n"], color="#e2e2e2", lw=1.5)
point_cols = [TEAL_DARK if i == len(s2_top) - 1 else GREY_DARK for i in range(len(s2_top))]
ax.scatter(s2_top["n"], y, s=62, color=point_cols, edgecolor="white", linewidth=0.8, zorder=3)
s2_denom = int(row["s2_violate"].sum())
for i, (_, r) in enumerate(s2_top.iterrows()):
    ax.text(
        r["n"] + s2_top["n"].max()*0.035, i,
        f"{int(r['n']):,} ({r['n']/s2_denom*100:.1f}%)",
        va="center", ha="left", fontsize=7.0, color=TEXT,
    )
ax.set_yticks(y)
ax.set_yticklabels([wrap(v, 34) for v in s2_top["label"]], fontsize=7.2)
ax.set_xlabel("Count among S2 violations", fontsize=8, color=TEXT)
panel_title(ax, "(d)", "Top more-than-two-fibre compositions under S2", y=1.04, fontsize=10.5)
ax.set_xlim(0, s2_top["n"].max()*1.45)
ax.margins(x=0)
clean_ax(ax, grid_axis="x")

fig.suptitle("Figure 3. Supported recognition scope and blend-complexity diagnostics", fontsize=12.3, y=1.01, color=TEXT)
save_fig(fig, "figure3_recognition_scope_mechanisms")
plt.close(fig)

# =============================================================================
# Figure 4: Minor-component detectability and black-colour proxy diagnostics
# =============================================================================

fig = plt.figure(figsize=(14.2, 7.8))
gs = fig.add_gridspec(
    2, 2,
    hspace=0.58,
    wspace=0.46,
    height_ratios=[1.00, 1.02]
)

# -------------------------------------------------------------------------
# Panel (a): S3 material dominance as a split rectangle
# -------------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])

total_s3_diag = int(
    diag_material[
        (diag_material["rule"] == "S3") &
        (diag_material["scenario"] == "5% threshold")
    ]["n"].sum()
)

elastane_n = int(
    diag_material[
        (diag_material["rule"] == "S3") &
        (diag_material["scenario"] == "5% threshold") &
        (diag_material["material_diag"] == "elastane")
    ]["n"].sum()
)

other_n = total_s3_diag - elastane_n

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

draw_rect_split(
    ax,
    ["elastane", "all other\nminor components"],
    [elastane_n, other_n],
    [GOLD, LIGHT_GREY],
    x=0.05,
    y=0.26,
    w=0.90,
    h=0.48,
    horizontal=True,
    min_label_share=0.04,
    fontsize=8.5,
)
ax.add_patch(Rectangle((0.05, 0.26), 0.90, 0.48, fill=False, edgecolor="#bbbbbb", linewidth=0.8))
panel_title(ax, "(a)", "S3 minor-component violations dominated by elastane", y=0.92, fontsize=10.5)
ax.text(0.05, 0.10, f"Baseline S3 diagnostic counts: elastane {elastane_n:,} of {total_s3_diag:,}",
        ha="left", va="bottom", fontsize=7.2, color=GREY_DARK)


# -------------------------------------------------------------------------
# Panel (b): top non-elastane minor materials
# -------------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])

minor = diag_material[
    (diag_material["rule"] == "S3") &
    (diag_material["scenario"] == "5% threshold")
].copy()
minor = (
    minor[minor["material_diag"] != "elastane"]
    .sort_values("n", ascending=False)
    .head(9)
    .sort_values("n", ascending=True)
)
y = np.arange(len(minor))
ax.hlines(y, 0, minor["n"], color="#e5e5e5", lw=1.5)
ax.scatter(minor["n"], y, s=72, color=GOLD_LIGHT, edgecolor=GOLD, linewidth=1.0, zorder=3)
for yy, (_, r) in zip(y, minor.iterrows()):
    ax.text(r["n"] + minor["n"].max() * 0.04, yy, f"{int(r['n']):,}", va="center", fontsize=7.5, color=TEXT)
ax.set_yticks(y)
ax.set_yticklabels([clean_label(v) for v in minor["material_diag"]], fontsize=7.5)
ax.set_xlabel("Diagnostic count", fontsize=8, color=TEXT)
panel_title(ax, "(b)", "Other S3 violating minor components", y=1.04, fontsize=10.5)
ax.set_xlim(0, minor["n"].max() * 1.18)
ax.margins(x=0)
clean_ax(ax, grid_axis="x")

# -------------------------------------------------------------------------
# Panel (c): exact black versus additional contains-black proxy
# -------------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])

s4_exp = diag_material[(diag_material["rule"] == "S4") & (diag_material["scenario"] == "Contains 'black'")].copy()
exact_black_n = int(s4_exp.loc[s4_exp["material_diag"].astype(str).str.lower() == "black", "n"].sum())
contains_black_total = int(s4_exp["n"].sum())
added_n = contains_black_total - exact_black_n
s4_vals = pd.Series({"exact black": exact_black_n, "additional\ncontains-“black”": added_n})

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
draw_rect_split(ax, s4_vals.index, s4_vals.values, [GREY_DARK, GREY],
                x=0.05, y=0.26, w=0.90, h=0.48,
                horizontal=True, min_label_share=0.04, fontsize=8.2)
ax.add_patch(Rectangle((0.05, 0.26), 0.90, 0.48, fill=False, edgecolor="#bbbbbb", linewidth=0.8))
panel_title(ax, "(c)", "Exact-black share within the S4 contains-black proxy", y=0.92, fontsize=10.5)
ax.text(0.05, 0.10, f"Contains-“black” total: {contains_black_total:,}; additional beyond exact black: {added_n:,}",
        ha="left", va="bottom", fontsize=7.2, color=GREY_DARK)


# -------------------------------------------------------------------------
# Panel (d): what the broader S4 black proxy adds
# -------------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])

s4_mixed = s4_exp[s4_exp["material_diag"].astype(str).str.lower() != "black"].copy()
s4_mixed = s4_mixed.sort_values("n", ascending=False).head(8).sort_values("n", ascending=True)
y = np.arange(len(s4_mixed))
xmax_mixed = max(s4_mixed["n"]) if not s4_mixed.empty else 1
ax.hlines(y, 0, s4_mixed["n"], color="#e5e5e5", lw=1.5)
ax.scatter(s4_mixed["n"], y, s=72, color=GREY_DARK, edgecolor="white", linewidth=0.8, zorder=3)
for yy, (_, r) in zip(y, s4_mixed.iterrows()):
    ax.text(r["n"] + xmax_mixed * 0.04, yy, f"{int(r['n']):,}", va="center", ha="left", fontsize=7.5, color=TEXT)
ax.set_yticks(y)
ax.set_yticklabels([clean_label(v) for v in s4_mixed["material_diag"]], fontsize=7.5)
ax.set_xlabel("Additional diagnostic count beyond exact black", fontsize=8, color=TEXT)
panel_title(ax, "(d)", "Mixed and descriptive labels added by the S4 contains-black proxy", y=1.04, fontsize=10.5)
ax.set_xlim(0, xmax_mixed * 1.38)
ax.margins(x=0)
clean_ax(ax, grid_axis="x")

fig.suptitle("Figure 4. Minor-component detectability and black-colour proxy diagnostics", fontsize=12.3, y=1.01, color=TEXT)
save_fig(fig, "figure4_minor_fibre_black_mechanisms")
plt.close(fig)

# =============================================================================
# Figure 5: Concealed structures and S5 surface–hidden material mismatch
# =============================================================================

fig = plt.figure(figsize=(14.6, 7.8))
gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.35], height_ratios=[0.95, 1.05],
                      hspace=0.55, wspace=0.45)

# Panel (a): S5 scenario sensitivity / nested logic
ax = fig.add_subplot(gs[0, 0])
any_hidden_n = int(summary.loc[(summary["rule"] == "S5") & (summary["setting"] == "Any hidden layer"), "n_violate"].iloc[0])
dom_n = int(summary.loc[(summary["rule"] == "S5") & (summary["setting"] == "Dominant-material mismatch"), "n_violate"].iloc[0])
central_n = int(summary.loc[(summary["rule"] == "S5") & (summary["setting"] == ">5% concealed-material mismatch"), "n_violate"].iloc[0])
levels = [
    ("Any hidden layer", any_hidden_n, TEAL_LIGHT),
    ("dominant-material\nmismatch", dom_n, TEAL),
    (">5 wt% concealed-material\nmismatch", central_n, TEAL_DARK),
]
maxn = any_hidden_n
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
for i, (lab, n, col) in enumerate(levels):
    w = 0.92 * (n / maxn)
    x0 = 0.04 + (0.92 - w) / 2
    y0 = 0.70 - i * 0.25
    ax.add_patch(Rectangle((x0, y0), w, 0.16, facecolor=col, edgecolor="white", linewidth=1.2))
    ax.text(0.50, y0 + 0.08, f"{lab}\n{n:,} ({n/N_TOTAL*100:.1f}%)", ha="center", va="center", fontsize=7.4, color="white" if i == 2 else TEXT)
ax.text(0.04, 0.96, "Three S5 rule scenarios", ha="left", va="top", fontsize=10.5, color=TEXT)
ax.text(0.04, 0.04, f"The baseline >5 wt% mismatch rule captures {central_n/any_hidden_n*100:.1f}% of garments with any hidden layer.", ha="left", va="bottom", fontsize=7.3, color=GREY_DARK)
panel_label(ax, "(a)")


# Panel (b): Sankey-like top mismatch flows
ax = fig.add_subplot(gs[:, 1])
s5_base = diag_material[(diag_material["rule"] == "S5") & (diag_material["scenario"] == ">5% concealed-material mismatch")].copy()
s5_total = int(s5_base["n"].sum())
s5_top = s5_base.sort_values("n", ascending=False).head(8).copy()
flows = []
for _, r in s5_top.iterrows():
    s_comp, s_mat, h_comp, h_mat = parse_s5_signature(r["material_diag"])
    if s_comp is None:
        continue
    flows.append((f"{s_comp}: {s_mat}", f"{h_comp}: {h_mat}", int(r["n"])))
left_weight, right_weight = Counter(), Counter()
for l, rr, n in flows:
    left_weight[l] += n
    right_weight[rr] += n
left_labels = [k for k, _ in left_weight.most_common()]
right_labels = [k for k, _ in right_weight.most_common()]
left_y = {lab: 0.86 - i * (0.72 / max(1, len(left_labels)-1)) for i, lab in enumerate(left_labels)}
right_y = {lab: 0.86 - i * (0.72 / max(1, len(right_labels)-1)) for i, lab in enumerate(right_labels)}
max_flow = max([n for _, _, n in flows]) if flows else 1

# Count labels are deliberately de-overlapped. In earlier versions, two flows
# could share almost the same midpoint (e.g. n=89 and n=80), causing one label
# to hide behind the other even though both flows were drawn.
raw_label_y = [(left_y[l] + right_y[rr]) / 2 + 0.02 for l, rr, n in flows]
label_y = list(raw_label_y)
min_gap = 0.045
if label_y:
    order = sorted(range(len(label_y)), key=lambda i: label_y[i], reverse=True)
    # First pass: push labels downward to maintain a minimum gap.
    prev_y = None
    for i in order:
        if prev_y is None:
            label_y[i] = min(label_y[i], 0.90)
        else:
            label_y[i] = min(label_y[i], prev_y - min_gap)
        prev_y = label_y[i]
    # If the lowest label is too low, shift all labels up together.
    min_y = min(label_y)
    if min_y < 0.14:
        shift = 0.14 - min_y
        label_y = [y + shift for y in label_y]

for idx, (l, rr, n) in enumerate(flows):
    draw_bezier_flow(
        ax, 0.30, left_y[l], 0.70, right_y[rr],
        width=1.5 + 8.5*n/max_flow, color=PURPLE_DARK, alpha=0.42
    )
    ax.text(
        0.50, label_y[idx], f"n={n:,}",
        ha="center", va="center", fontsize=6.7, color=TEXT,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.78),
        zorder=5,
    )
for lab in left_labels:
    ax.scatter([0.27], [left_y[lab]], s=42, color=PURPLE, edgecolor="white", zorder=3)
    ax.text(0.24, left_y[lab], wrap(lab, 24), ha="right", va="center", fontsize=6.6, color=TEXT)
for lab in right_labels:
    ax.scatter([0.73], [right_y[lab]], s=42, color=TEAL_DARK, edgecolor="white", zorder=3)
    ax.text(0.76, right_y[lab], wrap(lab, 24), ha="left", va="center", fontsize=6.6, color=TEXT)
ax.text(0.18, 0.98, "surface reference", ha="center", va="top", fontsize=7.5, color=GREY_DARK)
ax.text(0.82, 0.98, "concealed layer", ha="center", va="top", fontsize=7.5, color=GREY_DARK)
ax.text(0.02, 0.02, f"Flow widths and labels show diagnostic counts within baseline S5-violating cases (n={s5_total:,}), not shares of all garments.", ha="left", va="bottom", fontsize=7.1, color=GREY_DARK)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
panel_title(ax, "(b)", "Top S5 surface–hidden material mismatch signatures", y=1.02, fontsize=10.5)

# Panel (c): hidden component type dominance
ax = fig.add_subplot(gs[1, 0])
s5_all = diag_material[(diag_material["rule"] == "S5") & (diag_material["scenario"] == ">5% concealed-material mismatch")].copy()
parsed = s5_all["material_diag"].apply(parse_s5_signature)
s5_all["hidden_component"] = parsed.apply(lambda x: x[2] if x[2] else "other")
comp_counts = s5_all.groupby("hidden_component")["n"].sum().sort_values(ascending=False)
comp_plot = comp_counts.head(8).sort_values(ascending=True)
y = np.arange(len(comp_plot))
ax.hlines(y, 0, comp_plot.values, color="#e5e5e5", lw=1.5)
ax.scatter(comp_plot.values, y, s=70, color=TEAL_DARK, edgecolor="white", linewidth=0.8, zorder=3)
for yy, (lab, val) in zip(y, comp_plot.items()):
    ax.text(val + comp_plot.max() * 0.035, yy, f"{int(val):,} ({val / comp_counts.sum() * 100:.1f}%)",
            va="center", ha="left", fontsize=7.2, color=TEXT)
ax.set_yticks(y)
ax.set_yticklabels([clean_label(v) for v in comp_plot.index], fontsize=7.5)
ax.set_xlabel("Diagnostic count within baseline S5 violations", fontsize=8, color=TEXT)
panel_title(ax, "(c)", "Concealed component types in baseline S5 violations", y=1.04, fontsize=10.5)
ax.set_xlim(0, comp_plot.max() * 1.45)
ax.margins(x=0)
clean_ax(ax, grid_axis="x")


fig.suptitle("Figure 5. Concealed structures and S5 surface–hidden material mismatch", fontsize=12.3, y=1.01, color=TEXT)
save_fig(fig, "figure5_hidden_layer_mismatch")
plt.close(fig)

# =============================================================================
# Figure 6: Category-level barrier profiles
# =============================================================================

# Single heatmap only. No additional aggregation or rule-family comparison is
# used here, so the figure remains a direct descriptive category profile.
fig = plt.figure(figsize=(9.6, 9.4))
ax = fig.add_subplot(111)

cat = row.groupby("detail_category").agg(
    n=("row_id", "size"),
    anyb=("any_barrier_central", "mean"),
    s1=("s1_violate_central", "mean"),
    s2=("s2_violate", "mean"),
    s3=("s3_violate_central", "mean"),
    s4=("s4_violate_central", "mean"),
    s5=("s5_violate_central", "mean"),
).reset_index()
cat = cat[cat["n"] >= MIN_CATEGORY_N].copy()
cat = cat.sort_values("anyb", ascending=False).reset_index(drop=True)

cols = ["anyb", "s1", "s2", "s3", "s4", "s5"]
labels_cols = ["Any", "S1\nrecognition", "S2\nblend", "S3\nminor", "S4\nblack", "S5\nsurface–hidden"]
data = cat[cols].values * 100

cmap = LinearSegmentedColormap.from_list(
    "heat",
    ["#f7fbfa", TEAL_LIGHT, GOLD_LIGHT, PURPLE_DARK]
)
im = ax.imshow(data, aspect="auto", vmin=0, vmax=100, cmap=cmap)

ax.set_xticks(np.arange(len(cols)))
ax.set_xticklabels(labels_cols, fontsize=9)
ax.set_yticks(np.arange(len(cat)))
ax.set_yticklabels([clean_label(v) for v in cat["detail_category"]], fontsize=7.3)

# Show all cell values. Use text colour contrast so low and high values remain readable.
for i in range(len(cat)):
    for j in range(len(cols)):
        value = data[i, j]
        text_color = "white" if value >= 62 else TEXT
        ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=6.2, color=text_color)

ax.set_title("Category-level baseline rule-violation shares", loc="left", fontsize=11.0, color=TEXT)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(axis="both", length=0, colors=TEXT)

cbar = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.025)
cbar.set_label("Share of variants (%)", fontsize=8)
cbar.ax.tick_params(labelsize=7)

ax.text(
    0.0, -0.075,
    f"Rows include detailed categories with at least {MIN_CATEGORY_N:,} garment variants and are sorted by the baseline any-barrier share.",
    transform=ax.transAxes,
    fontsize=7.4,
    color=GREY_DARK,
    ha="left",
    va="top",
)

fig.suptitle("Figure 6. Category-level profiles of sorting-relevant design barriers", fontsize=12.3, y=0.995, color=TEXT)
save_fig(fig, "figure6_category_profiles")
plt.close(fig)

print("\nAll revised sorting figures saved in:", OUT_DIR.resolve())
