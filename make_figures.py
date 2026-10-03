"""
make_figures.py

Presentation figures built from the committed result files (no numbers typed in
by hand except where noted). Output: results/figures/*.png

Usage:
    python make_figures.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.config as cfg

OUT = cfg.RESULTS_DIR / "figures"
OUT.mkdir(parents=True, exist_ok=True)
EXP = cfg.EXPERIMENTS_DIR

# Validated reference palette (light mode)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 12, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def rec(exp_id):
    return json.load(open(EXP / exp_id / "experiment_record.json"))


def mean_f1(ids):
    return float(np.mean([rec(i)["macro_f1"] for i in ids]))


def label_bars(ax, bars, fmt="{:.3f}", inside=False):
    for b in bars:
        w = b.get_width()
        ax.text(w + 0.01 if not inside else w - 0.01, b.get_y() + b.get_height() / 2, fmt.format(w),
                va="center", ha="left" if not inside else "right", color=INK, fontsize=11)


def save(fig, name, footnote=None, top_note=None):
    bottom = 0.12 if footnote else 0.0
    top = 0.92 if top_note else 1.0
    fig.tight_layout(rect=[0, bottom, 1, top])
    if footnote:
        fig.text(0.01, 0.02, footnote, color=INK2, fontsize=10, ha="left", va="bottom", wrap=True)
    if top_note:
        fig.text(0.01, 0.97, top_note, color=INK2, fontsize=11, ha="left", va="top")
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("wrote", OUT / name)


def fig_fake_vs_honest():
    stages = ["Original pipeline\n(leaky)", "Empty-field fix", "Strict features\n(no packet IDs)",
              "Strict, no MQTT\n(final)"]
    xgb = [1.0, rec("C01_xgb_binary_clean")["macro_f1"], rec("S01_xgb_binary_strict")["macro_f1"],
           rec("N01_xgb_binary_strict_nomqtt")["macro_f1"]]
    cnn = [rec("E05_cnn_gru_binary")["macro_f1"], rec("C02_cnn_gru_binary_clean")["macro_f1"],
           rec("S02_cnn_gru_binary_strict")["macro_f1"], rec("F_strict_no_mqtt_binary_best_s42")["macro_f1"]]
    y = np.arange(len(stages))[::-1]
    h = 0.34
    fig, ax = plt.subplots(figsize=(10, 5.2))
    b1 = ax.barh(y + h / 2, cnn, h * 0.9, color=BLUE, label="CNN-GRU (ours)")
    b2 = ax.barh(y - h / 2, xgb, h * 0.9, color=ORANGE, label="XGBoost (baseline)")
    label_bars(ax, b1)
    label_bars(ax, b2)
    ax.set_yticks(y, stages)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Macro-F1, normal vs attack (Edge-IIoTset test set)")
    ax.set_title("The 100% score was a data shortcut; honest score is about 0.86")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    save(fig, "fig1_fake_vs_honest.png")


def fig_shap_fidelity():
    fid = json.load(open(cfg.RESULTS_DIR / "shap_fidelity.json"))
    names = ["Full model", "Remove 5 random features", "Remove top-5 SHAP features"]
    vals = [fid["baseline_macro_f1"], fid["FID_drop_random5"]["macro_f1"], fid["FID_drop_top5_shap"]["macro_f1"]]
    y = np.arange(3)[::-1]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(y, vals, 0.5, color=[BLUE, BLUE, ORANGE])
    label_bars(ax, bars)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Macro-F1 after retraining without those features")
    ax.set_title("SHAP is faithful: removing its top features breaks the model")
    ax.grid(axis="y", visible=False)
    top = ", ".join(f.split("__")[-1] for f in fid["top5_shap"])
    save(fig, "fig2_shap_fidelity.png", footnote=f"Top-5 SHAP features removed: {top}")


def fig_model_comparison():
    seeds = [42, 7, 2024]
    models = [
        ("MLP", [f"A_mlp_s{s}" for s in seeds]),
        ("1D-CNN", [f"A_cnn1d_s{s}" for s in seeds]),
        ("GRU", [f"A_gru_s{s}" for s in seeds]),
        ("CNN-GRU (tuned)", [f"F_strict_no_mqtt_binary_best_s{s}" for s in seeds]),
        ("XGBoost", ["N01_xgb_binary_strict_nomqtt"]),
    ]
    f1 = [mean_f1(ids) for _, ids in models]
    size_kb = [rec(ids[0])["model_size_mb"] * 1024 for _, ids in models]
    names = [m for m, _ in models]
    y = np.arange(len(models))[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, gridspec_kw={"width_ratios": [1.1, 1]})
    a1.scatter(f1, y, s=90, color=[ORANGE if n == "XGBoost" else BLUE for n in names], zorder=3)
    for v, yy in zip(f1, y):
        a1.text(v + 0.0015, yy, f"{v:.3f}", va="center", color=INK, fontsize=11)
    a1.set_yticks(y, names)
    a1.set_xlim(0.84, 0.88)
    a1.set_xlabel("Macro-F1 (mean of 3 seeds; zoomed axis)")
    a1.set_title("Accuracy: all networks tie")
    bars = a2.barh(y, size_kb, 0.5, color=[ORANGE if n == "XGBoost" else BLUE for n in names])
    for b, v in zip(bars, size_kb):
        a2.text(v + 8, b.get_y() + b.get_height() / 2, f"{v:.0f} KB", va="center", color=INK, fontsize=11)
    a2.tick_params(axis="y", left=False, labelleft=False)
    a2.set_xlim(0, max(size_kb) * 1.25)
    a2.set_xlabel("Model file size (KB)")
    a2.set_title("Size: the MLP is smallest")
    for a in (a1, a2):
        a.grid(axis="y", visible=False)
    save(fig, "fig3_model_comparison.png", top_note="Edge-IIoTset, normal vs attack, strict features")


def fig_datasets():
    seeds = [42, 7, 2024]
    groups = ["Normal vs attack", "Attack type / category"]
    edge = [rec("F_strict_no_mqtt_binary_best_s42")["macro_f1"],
            mean_f1([f"L_strict_sqrt_weighted_s{s}" for s in seeds])]
    cic = [rec("CS02_cnn_gru_binary_strict")["macro_f1"], rec("CS05_cnn_gru_category_sqrt_strict")["macro_f1"]]
    x = np.arange(2)
    w = 0.34
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, edge, w * 0.9, color=BLUE, label="Edge-IIoTset (single packets)")
    b2 = ax.bar(x + w / 2, cic, w * 0.9, color=AQUA, label="CICIoT2023 (flows with timing)")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.2f}",
                ha="center", color=INK, fontsize=12)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro-F1 (CNN-GRU, test set)")
    ax.set_title("Flow-level data with timing makes attack types learnable")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    save(fig, "fig4_datasets.png")


def fig_ceiling():
    ceil = json.load(open(cfg.AUDIT_DIR / "detection_ceiling.json"))["Edge-IIoTset (strict, no MQTT)"]
    ops = pd.read_csv(cfg.RESULTS_DIR / "operating_points.csv")
    default = ops[ops["operating_point"] == "default (0.5)"]
    edge = default[default["dataset"].str.startswith("Edge")].set_index("model")["detection_rate"]
    names = ["Best possible\n(perfect lookup)", "XGBoost", "CNN-GRU (ours)"]
    vals = [ceil["lookup_detection_rate_on_seen"], edge["XGBoost"], edge["CNN-GRU"]]
    y = np.arange(3)[::-1]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(y, [v * 100 for v in vals], 0.5, color=[INK2, ORANGE, BLUE])
    for b in bars:
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}%",
                va="center", color=INK, fontsize=11)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Attacks detected on Edge-IIoTset test set (%)")
    ax.set_title("Our model is at the ceiling of what packet features allow")
    ax.grid(axis="y", visible=False)
    save(fig, "fig5_detection_ceiling.png",
         footnote=f"{ceil['attacks_with_mostly_normal_pattern'] * 100:.1f}% of attack packets have exactly the same "
                  f"features as mostly-normal packets ({ceil['distinct_train_patterns']:,} distinct patterns in "
                  f"{ceil['train_rows']:,} training rows), so no packet-level model can detect them.")


if __name__ == "__main__":
    fig_fake_vs_honest()
    fig_shap_fidelity()
    fig_model_comparison()
    fig_datasets()
    fig_ceiling()
