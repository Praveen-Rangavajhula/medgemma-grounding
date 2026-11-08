"""Section 2.5: descriptive, matched-position residual-stream comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile

import numpy as np

from medgemma_grounding.activation_pilot import read_condition, verify_collection
from medgemma_grounding.grounding_baseline import json_text
from medgemma_grounding.grounding_protocol import sha256


def distances(reference: np.ndarray, changed: np.ndarray) -> dict:
    """Group Frobenius distances and mean aligned-token cosine, in float64."""
    if (reference.ndim != 2 or reference.shape != changed.shape or not reference.size
            or not np.isfinite(reference).all() or not np.isfinite(changed).all()):
        raise ValueError("Expected finite, matching, nonempty [tokens, width] arrays")
    a, b = reference.astype(np.float64), changed.astype(np.float64)
    delta = np.linalg.norm(b - a)
    norm = np.linalg.norm(a)
    denominators = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = denominators > 0
    cosine = np.clip(np.sum(a[valid] * b[valid], axis=1) / denominators[valid], -1, 1)
    return {"l2": float(delta), "reference_l2": float(norm),
            "relative_l2": float(delta / norm) if norm > 0 else None,
            "rmse": float(delta / np.sqrt(a.size)),
            "mean_token_cosine": float(cosine.mean()) if valid.any() else None,
            "mean_token_cosine_distance": float(1 - cosine.mean()) if valid.any() else None,
            "cosine_valid_tokens": int(valid.sum()), "tokens": len(a)}


def contrasts(rows: list[dict]) -> list[dict]:
    """Compare target distance with every matched control, separately per fill."""
    groups = sorted({(r["fill"], r["group"], r["layer"]) for r in rows})
    result = []
    for fill, group, layer in groups:
        selected = {r["condition"]: r for r in rows
                    if (r["fill"], r["group"], r["layer"]) == (fill, group, layer)}
        if set(selected) != {"target", "control_1", "control_2", "control_3"}:
            raise ValueError("Each layer/group/fill requires target and three controls")
        entry = {"fill": fill, "group": group, "layer": layer}
        for metric in ("relative_l2", "mean_token_cosine_distance"):
            target = selected["target"][metric]
            controls = [selected[f"control_{i}"][metric] for i in (1, 2, 3)]
            valid = target is not None and all(v is not None for v in controls)
            entry.update({f"target_{metric}": target, f"controls_{metric}": controls,
                          f"mean_control_{metric}": float(np.mean(controls)) if valid else None,
                          f"target_minus_mean_control_{metric}": float(target - np.mean(controls)) if valid else None})
        result.append(entry)
    return result


def candidate_layers(comparisons: list[dict]) -> dict:
    """Exploratory ranking, chosen rule before inspection of activation distances."""
    result = {}
    for group in sorted({r["group"] for r in comparisons}):
        ranked = []
        for layer in sorted({r["layer"] for r in comparisons}):
            by_fill = {r["fill"]: r["target_minus_mean_control_relative_l2"] for r in comparisons
                       if r["group"] == group and r["layer"] == layer}
            if set(by_fill) != {"mean", "blur"}:
                raise ValueError("Candidate comparison requires both fills")
            if all(value is not None and value > 0 for value in by_fill.values()):
                ranked.append({"layer": layer, "contrasts_by_fill": by_fill,
                               "minimum_contrast_across_fills": min(by_fill.values())})
        ranked.sort(key=lambda r: (-r["minimum_contrast_across_fills"], r["layer"]))
        result[group] = ranked[:3]
    return result


def plot_profiles(output: Path, rows: list[dict], comparisons: list[dict], case_id: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = ("decision", "image", "text")
    names = {"decision": "Final prefill position", "image": "Image tokens", "text": "Prompt text tokens"}
    colors = {"target": "#b51f35", "control_1": "#277da1", "control_2": "#43aa8b", "control_3": "#7953a4"}
    for metric, label, filename in (("relative_l2", "Relative L2 change", "layer_profiles.png"),
                                    ("mean_token_cosine_distance", "Mean token cosine distance", "cosine_profiles.png")):
        fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, sharey="row", layout="constrained")
        for i, group in enumerate(groups):
            for j, fill in enumerate(("mean", "blur")):
                ax = axes[i, j]
                for condition, color in colors.items():
                    selected = sorted([r for r in rows if r["group"] == group and r["fill"] == fill
                                       and r["condition"] == condition], key=lambda r: r["layer"])
                    ax.plot([r["layer"] for r in selected], [np.nan if r[metric] is None else r[metric] for r in selected],
                            color=color, label=condition.replace("_", " "), linewidth=2 if condition == "target" else 1.2)
                ax.set_title(f"{names[group]} · {fill} fill")
                ax.set_ylim(bottom=0)
                ax.grid(alpha=.2)
                if j == 0:
                    ax.set_ylabel(label)
                if i == 2:
                    ax.set_xlabel("Decoder layer (zero-based; post-residual output)")
        axes[0, 1].legend(fontsize=9, ncol=2)
        fig.suptitle(f"{case_id} — original vs. masked activations\nOne discovery case; descriptive sensitivity, not causal layer attribution", fontsize=13)
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for ax, group in zip(axes, groups):
        for fill, color in (("mean", "#b51f35"), ("blur", "#277da1")):
            selected = sorted([r for r in comparisons if r["group"] == group and r["fill"] == fill], key=lambda r: r["layer"])
            ax.plot([r["layer"] for r in selected],
                    [np.nan if r["target_minus_mean_control_relative_l2"] is None else r["target_minus_mean_control_relative_l2"] for r in selected],
                    color=color, label=fill)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_title(names[group])
        ax.set_xlabel("Decoder layer (zero-based)")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Target − mean control relative L2 change")
    axes[-1].legend()
    fig.suptitle("Positive: target mask changes activations more than the average control\nControls differ anatomically and are not independent replicates", fontsize=12)
    fig.savefig(output / "target_control_contrasts.png", dpi=150)
    plt.close(fig)


def analyze(activations: Path, output: Path) -> dict:
    run, conditions = verify_collection(activations)
    summary_path = activations / "summary.json"
    saved = json.loads(summary_path.read_text())
    if (saved["status"] != "single_case_activation_pilot_complete"
            or saved["run_sha256"] != sha256(activations / "run.json")
            or len(conditions) != len(run["identity"]["condition_ids"])
            or saved["condition_metadata_sha256"] != {r["id"]: sha256(activations / r["id"] / "condition.json") for r in conditions}):
        raise ValueError("Need a complete, unchanged section 2.4 collection")
    identity = {"activation_summary_sha256": sha256(summary_path), "source_sha256": sha256(Path(__file__)),
                "activation_path": str(activations.resolve())}
    if output.exists():
        previous = json.loads((output / "summary.json").read_text())
        if previous["identity"] != identity:
            raise ValueError("Analysis output exists with different inputs/code; use a new directory")
        for name, digest in previous["artifacts_sha256"].items():
            if sha256(output / name) != digest:
                raise ValueError("Saved analysis artifact changed")
        return previous
    _, baseline, _ = read_condition(activations / run["identity"]["condition_ids"][0], run["activation_shape"])
    rows, sanity = [], []
    baseline_score = conditions[0]["yes_score"]
    for meta in conditions[1:]:
        _, values, _ = read_condition(activations / meta["id"], run["activation_shape"])
        dest = rows if meta["kind"] == "region" else sanity
        for layer, hook in enumerate(run["hooks"]):
            for group, positions in run["tokens"]["positions"].items():
                dest.append({"id": meta["id"], "condition": meta["condition"], "fill": meta["fill"],
                             "layer": layer, "hook": hook, "group": group,
                             "yes_score": meta["yes_score"], "signed_score_drop": baseline_score - meta["yes_score"],
                             **distances(baseline[layer, positions], values[layer, positions])})
        del values
    comparisons = contrasts(rows)
    result = {"section": "2.5", "status": "single_case_layer_comparison_complete", "identity": identity,
              "case_id": run["case"]["case_id"], "layers": len(run["hooks"]), "region_conditions": len(conditions) - 3,
              "groups": {k: len(v) for k, v in run["tokens"]["positions"].items()},
              "candidate_rule": "Per token group: retain layers with positive target-minus-mean-control relative L2 in BOTH fills; rank by the smaller of the two contrasts, break ties by ascending layer, return up to three. Empty list is a valid result. Exploratory one-case candidates only.",
              "candidate_layers": candidate_layers(comparisons),
              "sanity_max_l2": max(r["l2"] for r in sanity),
              "metric_definitions": {"relative_l2": "||H_masked-H_original||_F / ||H_original||_F within each layer and token group; null for zero reference norm.",
                                     "mean_token_cosine": "Mean cosine of corresponding token vectors; exclude zero-norm pairs and report their count. No pooling before cosine.",
                                     "contrast": "Target distance minus arithmetic mean of the three control distances; controls are descriptive comparisons, not independent samples."},
              "limitations": ["One discovery image, no cohort or confirmation inference.",
                              "Large activation changes do not imply causal importance, pathology specificity, or a lower Yes score.",
                              "Residual scale varies by layer; relative L2 and cosine accompany raw L2/RMSE.",
                              "Image token summaries are not pixel localization maps; prompt text positions can overlap the final prefill position.",
                              "No probes, clustering, SAE features, p-values, or cross-layer score correlations are inferred from one case."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".layers-", dir=output.parent) as temporary:
        stage = Path(temporary) / "analysis"
        stage.mkdir()
        for name, table in (("layer_metrics.csv", rows), ("sanity_metrics.csv", sanity)):
            with (stage / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
        (stage / "target_control_contrasts.json").write_text(json_text(comparisons))
        plot_profiles(stage, rows, comparisons, run["case"]["case_id"])
        result["artifacts_sha256"] = {p.name: sha256(p) for p in stage.iterdir()}
        (stage / "summary.json").write_text(json_text(result))
        stage.rename(output)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, default=Path("outputs/part2/activations-v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/part2/layers-v1"))
    args = parser.parse_args(argv)
    try:
        result = analyze(args.activations, args.output)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Layer comparison failed: {error}\n")
    print(json_text({k: result[k] for k in ("status", "candidate_layers", "sanity_max_l2")}))
    print(f"Layer profiles saved to {args.output}")


if __name__ == "__main__":
    main()
