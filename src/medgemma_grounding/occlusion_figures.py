"""Review figures and plain tables for the occlusion experiment."""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def preparation_figure(directory, config, masks, model_size):
    names = list(masks)
    fig, axes = plt.subplots(3, 5, figsize=(18, 11))
    original = np.asarray(Image.open(directory / "model_original.png"))
    for row in range(3):
        axes[row, 0].imshow(original)
        axes[row, 0].set_title("Original" if row == 0 else "Original reference")
    for column, name in enumerate(names, 1):
        model_mask = np.asarray(Image.fromarray(masks[name]).resize(model_size, Image.Resampling.NEAREST))
        axes[0, column].imshow(original)
        axes[0, column].contour(model_mask, levels=[.5], colors=["#FFD43B" if name == "target" else "#33C8FF"], linewidths=1.3)
        if name != "target":
            target = np.asarray(Image.fromarray(masks["target"]).resize(model_size, Image.Resampling.NEAREST))
            axes[0, column].contour(target, levels=[.5], colors=["#FFD43B"], linewidths=.8)
        axes[0, column].set_title(f"{name.replace('_', ' ').title()} outline")
        for row, fill in enumerate(("mean", "blur"), 1):
            axes[row, column].imshow(Image.open(directory / f"model_{fill}_{name}.png"))
            axes[row, column].set_title(f"{fill.title()} / {name.replace('_', ' ')}")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{config['case_id']} | Pre-inference mask review\nYellow: expert region. Blue: equal-shape translated control. Actual processor-space image views.", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(directory / "mask_review.png", dpi=120)
    plt.close(fig)


def results_figure(prepared, output, maps, summary):
    config = json.loads((prepared / "manifest.json").read_text())
    size = tuple(config["model_size_width_height"])
    original = np.asarray(Image.open(prepared / "model_original.png"))
    mask = np.asarray(Image.open(prepared / "mask_target.png").resize(size, Image.Resampling.NEAREST)) > 0
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    limit = max(max(float(np.abs(m).max()) for m in maps.values()), 1e-8)
    region_values = [v for region in summary["regions"].values()
                     for v in [region["target_drop"], *region["control_drops"]]]
    low, high = min(0, min(region_values)), max(0, max(region_values))
    padding = max((high - low) * .15, 1e-8)
    for row, fill in enumerate(("mean", "blur")):
        axes[row, 0].imshow(original)
        axes[row, 0].contour(mask, levels=[.5], colors=["#FFD43B"], linewidths=1.5)
        axes[row, 0].set_title("Original + radiologist effusion outline")
        axes[row, 1].imshow(original)
        artist = axes[row, 1].imshow(maps[fill], cmap="RdBu_r", vmin=-limit, vmax=limit,
                                    alpha=.7, interpolation="nearest", extent=(-.5, size[0]-.5, size[1]-.5, -.5))
        axes[row, 1].contour(mask, levels=[.5], colors=["#FFD43B"], linewidths=1.2)
        axes[row, 1].set_title(f"{fill.title()} fill: signed Yes-score drop ({config['config']['grid']}×{config['config']['grid']})")
        fig.colorbar(artist, ax=axes[row, 1], fraction=.046, pad=.04)
        region = summary["regions"][fill]
        values = [region["target_drop"], *region["control_drops"]]
        axes[row, 2].bar(["Target", "Control 1", "Control 2", "Control 3"], values,
                         color=["#B07800", "#3188AD", "#3188AD", "#3188AD"])
        axes[row, 2].axhline(0, color="black", linewidth=.7)
        axes[row, 2].set_ylim(low - padding, high + padding)
        axes[row, 2].set_ylabel("Original score − occluded score")
        axes[row, 2].set_title(f"{fill.title()} fill: full-region occlusions")
        axes[row, 2].tick_params(axis="x", rotation=20)
        for ax in axes[row, :2]:
            ax.axis("off")
    fig.suptitle(f"{summary['case_id']} | One discovery-case pilot\nRed: score decreased after masking. Blue: score increased. Shared heatmap scale; no per-image normalization.", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(output / "results.png", dpi=130)
    plt.close(fig)


def write_tables(output, rows, original):
    columns = ["condition", "kind", "fill", "yes_score", "signed_drop", "response", "prediction",
               "changed_model_pixels", "row", "column", "bbox_xyxy", "seconds"]
    with (output / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, signed_drop=original - row["yes_score"]))
