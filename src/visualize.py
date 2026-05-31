import os
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from config import PROCESSED_DATA_PATH

warnings.filterwarnings("ignore")


# =============================================================================
# 1. GLOBAL STYLE
# =============================================================================

mpl.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     12,
    "axes.titleweight":  "bold",
    "axes.labelsize":     10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":          True,
    "grid.color":        "#e8e8e8",
    "grid.linewidth":     0.6,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
})

ACTION_COLORS = {
    "URGENT_RESTOCK":    "#E24B4A",
    "PRIORITY_RESTOCK":  "#EF9F27",
    "CONSIDER_MARKDOWN": "#378ADD",
    "OK":                "#639922",
}

FAMILY_COLORS = {
    "DAIRY":   "#534AB7",
    "MEATS":   "#D85A30",
    "PRODUCE": "#1D9E75",
}

CHARTS_DIR = os.path.join(PROCESSED_DATA_PATH, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

def needs_cols(df, *cols, chart_name="chart"):
    """Return True if all cols exist in df. Print a warning and return False otherwise."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"       ⚠  {chart_name}: skipped — missing columns: {missing}")
        print(f"          Available: {list(df.columns)}")
        return False
    return True


def safe_fam_col(df, dim_fam):
    """
    Try to attach family names to df via family_id merge.
    Returns (merged_df, col_name) where col_name is 'family' if the merge
    succeeded, or 'family_id' as a fallback.
    """
    if "family_id" in df.columns and not dim_fam.empty:
        merged = df.merge(dim_fam, on="family_id", how="left")
        return merged, "family" if "family" in merged.columns else "family_id"
    if "family" in df.columns:
        return df, "family"
    return df, None   # no family column at all


def load_all_data():
    files = {
        "recs":      "perishable_recommendations.csv",
        "preds":     "perishable_demand_predictions.csv",
        "sales":     "fact_sales.csv",
        "dim_fam":   "dim_family.csv",
        "inv":       "inventory_batches.csv",
        "per_cfg":   "perishable_config.csv",
        "dim_store": "dim_store.csv",
    }
    data = {}
    for key, filename in files.items():
        path = os.path.join(PROCESSED_DATA_PATH, filename)
        try:
            data[key] = pd.read_csv(path)
            print(f"  ✓  {filename}  ({len(data[key]):,} rows)")
        except FileNotFoundError:
            print(f"  ✗  {filename} not found — run the pipeline first.")
            data[key] = pd.DataFrame()

    if not data["sales"].empty and "sales_date" in data["sales"].columns:
        data["sales"]["sales_date"] = pd.to_datetime(data["sales"]["sales_date"])
    if not data["inv"].empty:
        for col in ["received_date", "expiry_date"]:
            if col in data["inv"].columns:
                data["inv"][col] = pd.to_datetime(data["inv"][col])
    if not data["preds"].empty and "sales_date" in data["preds"].columns:
        data["preds"]["sales_date"] = pd.to_datetime(data["preds"]["sales_date"])

    return data


def save_fig(fig, pdf, filename):
    png_path = os.path.join(CHARTS_DIR, filename)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"       → {filename}")

def make_cover_page(pdf):
    fig = plt.figure(figsize=(11, 8.5))

    banner = fig.add_axes([0, 0.88, 1, 0.12])
    banner.set_facecolor("#1D1D2E")
    banner.set_axis_off()

    fig.text(0.5, 0.78, "Perishable Inventory\nRecommendation Report",
             ha="center", va="center",
             fontsize=28, fontweight="bold", color="#1D1D2E", linespacing=1.4)
    fig.text(0.5, 0.63,
             "Demand Forecasting  ·  Inventory Analysis  ·  Action Recommendations",
             ha="center", fontsize=12, color="#666666")

    fig.text(0.5, 0.50, "Action key:", ha="center", fontsize=10, color="#555")
    x_positions = [0.18, 0.38, 0.60, 0.80]
    for (action, color), x in zip(ACTION_COLORS.items(), x_positions):
        patch = mpatches.FancyBboxPatch(
            (x - 0.085, 0.40), 0.17, 0.055,
            boxstyle="round,pad=0.01", facecolor=color, edgecolor="none",
            transform=fig.transFigure, clip_on=False,
        )
        fig.add_artist(patch)
        fig.text(x, 0.427, action.replace("_", " ").title(),
                 ha="center", va="center", fontsize=8.5,
                 color="white", fontweight="bold",
                 transform=fig.transFigure)

    fig.text(0.5, 0.10, "Generated by visualize.py",
             ha="center", fontsize=9, color="#aaaaaa")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# CHART 1 — KPI SUMMARY
# =============================================================================

def chart_kpi_summary(recs, pdf):
    if recs.empty or not needs_cols(recs, "recommended_action", chart_name="KPI summary"):
        return

    counts = recs["recommended_action"].value_counts()
    total  = len(recs)

    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    fig.suptitle("Recommendation summary — all store × family combinations",
                 fontweight="bold", fontsize=13, y=1.02)

    _kpi_box(axes[0], total, "Total evaluated", "#444444", "#f5f5f5")
    for ax, (action, color) in zip(axes[1:], ACTION_COLORS.items()):
        n   = counts.get(action, 0)
        pct = n / total * 100 if total else 0
        _kpi_box(ax, n, action.replace("_", "\n"), color, color + "22",
                 subtitle=f"{pct:.1f}% of total")

    fig.tight_layout()
    save_fig(fig, pdf, "01_kpi_summary.png")


def _kpi_box(ax, value, label, text_color, bg_color, subtitle=""):
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.65, f"{value:,}",
            ha="center", va="center", fontsize=24, fontweight="bold", color=text_color)
    ax.text(0.5, 0.35, label,
            ha="center", va="center", fontsize=9, color=text_color, linespacing=1.3)
    if subtitle:
        ax.text(0.5, 0.12, subtitle,
                ha="center", va="center", fontsize=8, color=text_color + "bb")
    ax.axvline(0.04, color=text_color, lw=3, ymin=0.1, ymax=0.9)


# =============================================================================
#  CHART 2 — ACTION DISTRIBUTION
# =============================================================================

def chart_action_distribution(recs, pdf):
    if recs.empty or not needs_cols(recs, "recommended_action",
                                    chart_name="Action distribution"):
        return

    counts = (
        recs["recommended_action"]
            .value_counts()
            .reindex(ACTION_COLORS.keys(), fill_value=0)
    )

    fig, (ax_d, ax_b) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("How are recommendations distributed?", fontweight="bold")

    # Donut
    wedges, _, autotexts = ax_d.pie(
        counts.values,
        colors=[ACTION_COLORS[k] for k in counts.index],
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        pctdistance=0.78,
    )
    for at in autotexts:
        at.set_fontsize(9); at.set_color("white"); at.set_fontweight("bold")
    ax_d.text(0, 0, f"{counts.sum():,}\ntotal",
              ha="center", va="center", fontsize=11, fontweight="bold", color="#333")
    handles = [
        mpatches.Patch(facecolor=ACTION_COLORS[k],
                       label=f"{k.replace('_', ' ')}  ({v:,})")
        for k, v in counts.items()
    ]
    ax_d.legend(handles=handles, loc="lower center",
                bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8, frameon=False)
    ax_d.set_title("Overall mix", fontsize=11)

    # Horizontal bar
    bars = ax_b.barh(counts.index, counts.values,
                     color=[ACTION_COLORS[k] for k in counts.index],
                     edgecolor="none", height=0.55)
    for bar, val in zip(bars, counts.values):
        ax_b.text(val + counts.values.max() * 0.01,
                  bar.get_y() + bar.get_height() / 2,
                  f"{val:,}", va="center", fontsize=9)
    ax_b.set_xlabel("Count")
    ax_b.set_yticklabels([k.replace("_", "\n") for k in counts.index], fontsize=9)
    ax_b.set_title("Count per action", fontsize=11)
    ax_b.invert_yaxis()

    fig.tight_layout()
    save_fig(fig, pdf, "02_action_distribution.png")


# =============================================================================
#  CHART 3 — ACTIONS BY PRODUCT FAMILY
# =============================================================================

def chart_actions_by_family(recs, dim_fam, pdf):
    if recs.empty or not needs_cols(recs, "recommended_action",
                                    chart_name="Actions by family"):
        return

    merged, fam_col = safe_fam_col(recs, dim_fam)

    # Guard: if there is genuinely no family information at all, skip
    if fam_col is None or fam_col not in merged.columns:
        print("       ⚠  Actions by family: skipped — no family column found in recs.")
        return

    pivot = (
        merged.groupby([fam_col, "recommended_action"])
              .size()
              .unstack(fill_value=0)
              .reindex(columns=list(ACTION_COLORS.keys()), fill_value=0)
    )

    n_fam    = len(pivot)
    n_act    = len(ACTION_COLORS)
    x        = np.arange(n_fam)
    width    = 0.18

    fig, ax = plt.subplots(figsize=(max(8, n_fam * 2.5), 5))
    fig.suptitle("Recommended actions by product family", fontweight="bold")

    for i, (action, color) in enumerate(ACTION_COLORS.items()):
        if action not in pivot.columns:
            continue
        offset = (i - n_act / 2 + 0.5) * width
        bars   = ax.bar(x + offset, pivot[action], width,
                        color=color, edgecolor="none",
                        label=action.replace("_", " "))
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + pivot.values.max() * 0.01,
                        str(int(h)), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontsize=10)
    ax.set_ylabel("Count")
    ax.set_ylim(0, pivot.values.max() * 1.18)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper right")

    fig.tight_layout()
    save_fig(fig, pdf, "03_actions_by_family.png")


# =============================================================================
#  CHART 4 — STORE-LEVEL HEATMAP
# =============================================================================

def chart_store_heatmap(recs, pdf):
    if recs.empty:
        return
    if not needs_cols(recs, "store_id", "recommended_action",
                      chart_name="Store heatmap"):
        return

    pivot = (
        recs.groupby(["store_id", "recommended_action"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=list(ACTION_COLORS.keys()), fill_value=0)
    )
    top_stores = pivot.sum(axis=1).nlargest(40).index
    pivot = pivot.loc[top_stores].sort_values("URGENT_RESTOCK", ascending=False)

    fig_h = max(5, len(pivot) * 0.32)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.suptitle("Store-level heatmap — action counts per store", fontweight="bold")

    sns.heatmap(
        pivot, ax=ax, cmap="YlOrRd",
        linewidths=0.3, linecolor="#f0f0f0",
        annot=len(pivot) <= 25, fmt="d",
        cbar_kws={"shrink": 0.5, "label": "Row count"},
    )
    ax.set_xlabel("Recommended action")
    ax.set_ylabel("Store ID")
    ax.set_xticklabels(
        [t.get_text().replace("_", "\n") for t in ax.get_xticklabels()],
        fontsize=9, rotation=0,
    )
    ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    save_fig(fig, pdf, "04_store_heatmap.png")


# =============================================================================
#  CHART 5 — PREDICTED vs ACTUAL
# =============================================================================

def chart_predicted_vs_actual(preds, pdf):
    if preds.empty:
        return
    if not needs_cols(preds, "actual_sales", "predicted_sales",
                      chart_name="Predicted vs actual"):
        return

    sample    = preds.sample(min(500, len(preds)), random_state=42)
    residuals = sample["predicted_sales"] - sample["actual_sales"]
    mae       = np.abs(residuals).mean()
    rmse      = np.sqrt((residuals ** 2).mean())
    ss_res    = (residuals ** 2).sum()
    ss_tot    = ((sample["actual_sales"] - sample["actual_sales"].mean()) ** 2).sum()
    r2        = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model accuracy — predicted vs actual sales", fontweight="bold")

    # Scatter
    ax1.scatter(sample["actual_sales"], sample["predicted_sales"],
                alpha=0.35, s=16, color=ACTION_COLORS["OK"], edgecolors="none", zorder=2)
    lim_max = max(sample["actual_sales"].max(), sample["predicted_sales"].max()) * 1.05
    lim_min = min(0, sample["actual_sales"].min(), sample["predicted_sales"].min())
    ax1.plot([lim_min, lim_max], [lim_min, lim_max],
             "--", color="#888", lw=1.2, label="Perfect fit  y = x", zorder=1)
    stats_txt = f"MAE  = {mae:.2f}\nRMSE = {rmse:.2f}\nR²   = {r2:.3f}"
    ax1.text(0.04, 0.96, stats_txt, transform=ax1.transAxes,
             va="top", fontsize=9, color="#333",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="#ddd", alpha=0.9))
    ax1.set_xlabel("Actual sales"); ax1.set_ylabel("Predicted sales")
    ax1.set_title("Scatter: actual vs predicted (test set)")
    ax1.legend(fontsize=8, frameon=False)

    # Residuals
    ax2.hist(residuals, bins=40, color=ACTION_COLORS["PRIORITY_RESTOCK"],
             edgecolor="white", linewidth=0.4, alpha=0.85)
    ax2.axvline(0, color="#333", lw=1.2, linestyle="--", label="Zero error")
    ax2.axvline(residuals.mean(), color=ACTION_COLORS["URGENT_RESTOCK"], lw=1.2,
                linestyle=":", label=f"Mean = {residuals.mean():.1f}")
    ax2.set_xlabel("Residual  (predicted − actual)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residual distribution")
    ax2.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    save_fig(fig, pdf, "05_model_accuracy.png")


# =============================================================================
#  CHART 6 — SALES TREND OVER TIME
# =============================================================================

def chart_sales_trend(sales, dim_fam, per_cfg, pdf):
    if sales.empty or dim_fam.empty:
        return
    if not needs_cols(sales, "sales_date", "family_id", "sales",
                      chart_name="Sales trend"):
        return

    merged, fam_col = safe_fam_col(sales, dim_fam)
    if fam_col is None:
        return

    if not per_cfg.empty and "family_id" in per_cfg.columns:
        merged = merged[merged["family_id"].isin(per_cfg["family_id"].unique())]

    if merged.empty:
        print("       ⚠  Sales trend: no perishable rows after filter.")
        return

    weekly = (
        merged.groupby([pd.Grouper(key="sales_date", freq="W"), fam_col])["sales"]
              .mean()
              .reset_index()
    )
    families = weekly[fam_col].unique()
    n        = len(families)

    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle("Weekly average sales — perishable families", fontweight="bold")

    for ax, fam in zip(axes, families):
        fam_data = weekly[weekly[fam_col] == fam]
        color    = FAMILY_COLORS.get(str(fam), "#378ADD")
        ax.plot(fam_data["sales_date"], fam_data["sales"],
                color=color, lw=1.5)
        ax.fill_between(fam_data["sales_date"], fam_data["sales"],
                        alpha=0.12, color=color)
        rolling = fam_data["sales"].rolling(4, min_periods=1).mean()
        ax.plot(fam_data["sales_date"], rolling,
                color=color, lw=2.5, alpha=0.6, linestyle="--", label="4-wk avg")
        ax.set_ylabel("Avg daily sales")
        ax.set_title(str(fam), fontweight="bold", color=color, pad=4)
        ax.legend(fontsize=8, frameon=False, loc="upper left")

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    save_fig(fig, pdf, "06_sales_trend.png")


# =============================================================================
#  CHART 7 — PROMOTION IMPACT
# =============================================================================

def chart_promotion_impact(sales, dim_fam, per_cfg, pdf):
    if sales.empty:
        return
    if not needs_cols(sales, "onpromotion", "family_id", "sales",
                      chart_name="Promotion impact"):
        return

    merged, fam_col = safe_fam_col(sales, dim_fam)
    if fam_col is None:
        return

    if not per_cfg.empty and "family_id" in per_cfg.columns:
        merged = merged[merged["family_id"].isin(per_cfg["family_id"].unique())]

    promo_effect = (
        merged.groupby([fam_col, "onpromotion"])["sales"]
              .mean()
              .reset_index()
    )
    families = promo_effect[fam_col].unique()
    x        = np.arange(len(families))
    width    = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.suptitle("Average sales: on promotion vs not on promotion", fontweight="bold")

    for i, (promo_val, label, color) in enumerate([
        (0, "Not on promotion", "#B4B2A9"),
        (1, "On promotion",     "#534AB7"),
    ]):
        vals = []
        for fam in families:
            row = promo_effect[
                (promo_effect[fam_col] == fam) &
                (promo_effect["onpromotion"] == promo_val)
            ]
            vals.append(float(row["sales"].values[0]) if len(row) else 0.0)

        bars = ax.bar(x + i * width, vals, width,
                      color=color, edgecolor="none", label=label)
        max_val = max(vals) if vals else 1
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_val * 0.01,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(families, fontsize=10)
    ax.set_ylabel("Average daily sales")
    ax.legend(fontsize=9, frameon=False)

    fig.tight_layout()
    save_fig(fig, pdf, "07_promotion_impact.png")


# =============================================================================
#  CHART 8 — INVENTORY ON-HAND vs PREDICTED DEMAND
# =============================================================================

def chart_inventory_vs_demand(recs, pdf):
    if recs.empty:
        return
    if not needs_cols(recs, "predicted_sales", "on_hand_qty", "recommended_action",
                      chart_name="Inventory vs demand"):
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("On-hand quantity vs predicted demand", fontweight="bold")

    for action, color in ACTION_COLORS.items():
        sub = recs[recs["recommended_action"] == action]
        if sub.empty:
            continue
        ax.scatter(sub["predicted_sales"], sub["on_hand_qty"],
                   c=color, s=50, alpha=0.65, edgecolors="none",
                   label=action.replace("_", " "), zorder=2)

    lim = max(recs["predicted_sales"].max(), recs["on_hand_qty"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "--", color="#888", lw=1.2,
            label="y = x  (perfectly matched)", zorder=1)
    ax.fill_between([0, lim], [0, 0], [0, lim],
                    alpha=0.05, color="#E24B4A", label="Understocked zone")

    ax.set_xlabel("Predicted sales")
    ax.set_ylabel("On-hand quantity")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    fig.tight_layout()
    save_fig(fig, pdf, "08_inventory_vs_demand.png")


# =============================================================================
#  CHART 9 — EXPIRY RISK WINDOW
# =============================================================================

def chart_expiry_risk(inv, dim_fam, pdf):
    if inv.empty:
        return
    if not needs_cols(inv, "expiry_date", "received_qty", "family_id",
                      chart_name="Expiry risk"):
        return

    today      = inv["expiry_date"].max() - pd.Timedelta(days=30)
    window_end = today + pd.Timedelta(days=7)
    expiring   = inv[(inv["expiry_date"] >= today) &
                     (inv["expiry_date"] <= window_end)].copy()

    if expiring.empty:
        print("       ⚠  Expiry risk: no batches expire in the 7-day window.")
        return

    expiring["days_to_expiry"] = (expiring["expiry_date"] - today).dt.days
    expiring, fam_col = safe_fam_col(expiring, dim_fam)
    if fam_col is None:
        fam_col = "family_id"

    daily = (
        expiring.groupby([fam_col, "days_to_expiry"])["received_qty"]
                .sum()
                .reset_index()
    )
    families = daily[fam_col].unique()
    days     = sorted(daily["days_to_expiry"].unique())
    x        = np.arange(len(days))
    n_fam    = len(families)
    width    = 0.7 / max(n_fam, 1)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.suptitle("Units expiring in the next 7 days — by family", fontweight="bold")

    for i, fam in enumerate(families):
        fam_data = daily[daily[fam_col] == fam]
        day_vals = [
            int(fam_data[fam_data["days_to_expiry"] == d]["received_qty"].sum())
            for d in days
        ]
        color  = FAMILY_COLORS.get(str(fam), "#888")
        offset = (i - n_fam / 2 + 0.5) * width
        ax.bar(x + offset, day_vals, width,
               color=color, edgecolor="none", label=str(fam), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Day +{d}" for d in days], fontsize=9)
    ax.set_xlabel("Days until expiry"); ax.set_ylabel("Units expiring")
    ax.legend(fontsize=9, frameon=False)

    fig.tight_layout()
    save_fig(fig, pdf, "09_expiry_risk.png")


# =============================================================================
#  CHART 10 — FEATURE CORRELATION WITH ERROR
# =============================================================================

def chart_feature_importance(preds, pdf):
    if preds.empty:
        return
    if not needs_cols(preds, "actual_sales", "predicted_sales",
                      chart_name="Feature importance"):
        return

    feature_cols = ["transactions", "shelf_life_days",
                    "lag_1_sales", "lag_7_sales", "rolling_7d_mean"]
    available = [c for c in feature_cols if c in preds.columns]
    if not available:
        print("       ⚠  Feature importance: no feature columns in preds.")
        return

    residuals_abs = (preds["predicted_sales"] - preds["actual_sales"]).abs()
    importances   = {col: abs(preds[col].corr(residuals_abs, method="spearman"))
                     for col in available}
    imp = pd.Series(importances).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Feature influence on prediction error\n(|Spearman r| with absolute residual)",
                 fontweight="bold")

    colors = [ACTION_COLORS["OK"] if v < imp.median()
              else ACTION_COLORS["PRIORITY_RESTOCK"]
              for v in imp.values]
    bars = ax.barh(imp.index, imp.values, color=colors, edgecolor="none", height=0.55)

    for bar, val in zip(bars, imp.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.set_xlabel("|Spearman r|")
    ax.set_xlim(0, imp.max() * 1.25)

    fig.tight_layout()
    save_fig(fig, pdf, "10_feature_importance.png")


# =============================================================================
#  CHART 11 — ACTION MIX OVER TIME
# =============================================================================

def chart_action_timeline(recs, sales, pdf):
    if recs.empty or sales.empty:
        return
    if not needs_cols(recs, "store_id", "recommended_action",
                      chart_name="Action timeline"):
        return
    if not needs_cols(sales, "store_id", "sales_date",
                      chart_name="Action timeline (sales)"):
        return

    latest_dates = (
        sales.groupby("store_id")["sales_date"]
             .max().reset_index()
             .rename(columns={"sales_date": "approx_date"})
    )
    recs_dated = recs.merge(latest_dates, on="store_id", how="left")
    recs_dated = recs_dated.dropna(subset=["approx_date"])
    recs_dated["approx_date"] = pd.to_datetime(recs_dated["approx_date"])
    recs_dated["month"] = recs_dated["approx_date"].dt.to_period("M")

    monthly = (
        recs_dated.groupby(["month", "recommended_action"])
                  .size().unstack(fill_value=0)
                  .reindex(columns=list(ACTION_COLORS.keys()), fill_value=0)
    )
    if len(monthly) < 2:
        print("       ⚠  Action timeline: fewer than 2 periods — skipped.")
        return

    monthly.index  = monthly.index.astype(str)
    monthly_pct    = monthly.div(monthly.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Action mix over time — proportion per month", fontweight="bold")

    x      = np.arange(len(monthly_pct))
    bottom = np.zeros(len(monthly_pct))
    for action, color in ACTION_COLORS.items():
        if action not in monthly_pct.columns:
            continue
        vals = monthly_pct[action].values
        ax.bar(x, vals, bottom=bottom, color=color, edgecolor="none",
               label=action.replace("_", " "), width=0.8)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(monthly_pct.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% of recommendations")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(fontsize=8, frameon=False,
              bbox_to_anchor=(1.01, 1), loc="upper left")

    fig.tight_layout()
    save_fig(fig, pdf, "11_action_timeline.png")

#=============================================================================

def main():
    print("\n" + "=" * 60)
    print("  PERISHABLE INVENTORY REPORT GENERATOR")
    print("=" * 60)

    print("\n[1/3]  Loading data...")
    d = load_all_data()

    report_path = os.path.join(PROCESSED_DATA_PATH, "report.pdf")
    print(f"\n[2/3]  Building charts → {report_path}")

    with PdfPages(report_path) as pdf:
        info = pdf.infodict()
        info["Title"]   = "Perishable Inventory Recommendation Report"
        info["Subject"] = "Demand Forecasting and Inventory Action Recommendations"

        print("\n     Cover page")
        make_cover_page(pdf)

        print("     Chart 1  — KPI summary")
        chart_kpi_summary(d["recs"], pdf)

        print("     Chart 2  — Action distribution")
        chart_action_distribution(d["recs"], pdf)

        print("     Chart 3  — Actions by product family")
        chart_actions_by_family(d["recs"], d["dim_fam"], pdf)

        print("     Chart 4  — Store-level heatmap")
        chart_store_heatmap(d["recs"], pdf)

        print("     Chart 5  — Predicted vs actual")
        chart_predicted_vs_actual(d["preds"], pdf)

        print("     Chart 6  — Sales trend over time")
        chart_sales_trend(d["sales"], d["dim_fam"], d["per_cfg"], pdf)

        print("     Chart 7  — Promotion impact")
        chart_promotion_impact(d["sales"], d["dim_fam"], d["per_cfg"], pdf)

        print("     Chart 8  — Inventory vs demand")
        chart_inventory_vs_demand(d["recs"], pdf)

        print("     Chart 9  — Expiry risk window")
        chart_expiry_risk(d["inv"], d["dim_fam"], pdf)

        print("     Chart 10 — Feature importance")
        chart_feature_importance(d["preds"], pdf)

        print("     Chart 11 — Action mix over time")
        chart_action_timeline(d["recs"], d["sales"], pdf)

    print(f"\n[3/3]  Done.")
    print(f"\n  PDF report  →  {report_path}")
    print(f"  PNG charts  →  {CHARTS_DIR}/")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()