
"""
Pleural Effusion Classification Benchmark using MedGemma 1.5

Author: Braveenan Sritharan
Created: 2026-11-02

Description:
    Evaluate MedGemma 1.5 on frontal CheXpert validation images
    for binary pleural effusion classification.

    The script:
    1. Loads frontal chest X-rays with binary ground-truth labels.
    2. Generates responses using a fixed Yes/No prompt.
    3. Extracts binary predictions from generated responses.
    4. Computes continuous first-token Yes-vs-No scores.
    5. Prints and incrementally saves image-level predictions.
    6. Computes classification metrics and saves them to a TXT file.
    7. Generates confusion matrix, ROC, PR, and score distribution plots.

Classification Metrics:
    TP = True Positive
    TN = True Negative
    FP = False Positive
    FN = False Negative

    Accuracy    = (TP + TN) / (TP + TN + FP + FN)
    Precision   = TP / (TP + FP)
    Recall      = TP / (TP + FN)
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    TPR         = TP / (TP + FN)
    FPR         = FP / (FP + TN)
    F1-score    = 2TP / (2TP + FP + FN)

    ROC-AUC     = Area under the ROC curve
    AP          = Average Precision (PR-curve summary)

Outputs:
    results/classification/predictions.csv
    results/classification/metrics.txt
    results/classification/plots/confusion_matrix.png
    results/classification/plots/roc_curve.png
    results/classification/plots/pr_curve.png
    results/classification/plots/score_distribution.png

Notes:
    - Binary metrics use predictions parsed from generated text.
    - Unparsed responses are excluded from binary metrics and counted.
    - ROC-AUC and AP use continuous Yes-vs-No scores.
    - The continuous score is not a calibrated disease probability.
"""

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
)

from medgemma_grounding.inference import (
    load_model,
    load_image,
    prepare_inputs,
    generate,
)


LABELS_CSV = Path("data/CheXpert/val_labels.csv")
IMAGE_ROOT = Path("data/CheXpert/val")
OUTPUT_DIR = Path("results/classification")
PLOTS_DIR = OUTPUT_DIR / "plots"

PREDICTIONS_CSV = OUTPUT_DIR / "predictions.csv"
METRICS_TXT = OUTPUT_DIR / "metrics.txt"

PROMPT = "Is pleural effusion present in this chest X-ray? Answer Yes or No."

MAX_NEW_TOKENS = 16

PREDICTION_COLUMNS = [
    "image_path",
    "ground_truth",
    "response",
    "prediction",
    "yes_score",
]


def load_dataset():
    """Load frontal validation images with binary pleural effusion labels."""

    df = pd.read_csv(LABELS_CSV)

    df = df[df["Frontal/Lateral"] == "Frontal"].copy()
    df = df[df["Pleural Effusion"].isin([0.0, 1.0])].copy()

    def resolve_image_path(path):
        relative = Path(path)
        return IMAGE_ROOT / Path(*relative.parts[2:])

    df["image_path"] = df["Path"].apply(resolve_image_path)

    missing = [
        str(path)
        for path in df["image_path"]
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images are missing. First: {missing[0]}"
        )

    return df.reset_index(drop=True)


def parse_prediction(response):
    """Extract a binary pleural effusion prediction from generated text."""

    text = response.lower().strip()

    # Recognize explicit Yes/No answers.
    if re.match(r"^yes\b", text):
        return 1

    if re.match(r"^no\b", text):
        return 0

    # Avoid assigning a label to explicitly uncertain responses.
    uncertainty_patterns = [
        r"\b(?:cannot|can't|unable to) determine\b",
        r"\b(?:cannot|can't) tell\b",
        r"\b(?:uncertain|indeterminate|inconclusive)\b",
        r"\b(?:possible|possibly|may be|might be) pleural effusion\b",
        r"\b(?:cannot|can't) (?:confirm|exclude|rule out)\b",
    ]

    if any(re.search(pattern, text) for pattern in uncertainty_patterns):
        return None

    negative_patterns = [
        r"\bno (?:obvious |evidence of |significant )?pleural effusion\b",
        r"\bno evidence of (?:a |any )?pleural effusion\b",
        r"\bpleural effusion is (?:not present|absent)\b",
        r"\bwithout pleural effusion\b",
        r"\bnegative for pleural effusion\b",
        r"\babsence of pleural effusion\b",
    ]

    positive_patterns = [
        r"\bpleural effusion is present\b",
        r"\b(?:there is|there are) (?:a |an )?"
        r"(?:small |moderate |large )?pleural effusion\b",
        r"\bpositive for pleural effusion\b",
        r"\b(?:small |moderate |large )pleural effusion\b",
    ]

    negative = any(
        re.search(pattern, text)
        for pattern in negative_patterns
    )

    positive = any(
        re.search(pattern, text)
        for pattern in positive_patterns
    )

    if negative and not positive:
        return 0

    if positive and not negative:
        return 1

    # Unknown or conflicting response.
    return None


def compute_yes_score(bundle, image, prompt):
    """Compute a forced-choice Yes-vs-No score from next-token logits."""

    inputs = prepare_inputs(bundle, image, prompt)

    tokenizer = bundle.processor.tokenizer

    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)

    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError("Yes and No must each be a single token.")

    with bundle.torch.inference_mode():
        outputs = bundle.model(**inputs)

    # Logits for the first generated token.
    next_token_logits = outputs.logits[0, -1, :]

    # Select the No and Yes candidate logits.
    candidate_logits = bundle.torch.stack([
        next_token_logits[no_ids[0]],
        next_token_logits[yes_ids[0]],
    ]).float()

    # Normalize over only the two candidate tokens.
    probabilities = bundle.torch.softmax(
        candidate_logits,
        dim=0,
    )

    return probabilities[1].item()


def compute_metrics(results):
    """Calculate classification metrics from saved predictions."""

    y_true = results["ground_truth"].astype(int)
    y_score = results["yes_score"].astype(float)

    # Continuous-score metrics.
    if y_true.nunique() == 2:
        roc_auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
    else:
        roc_auc = float("nan")
        ap = float("nan")

    # Binary metrics from parsed generated responses.
    valid = results.dropna(subset=["prediction"]).copy()

    if valid.empty:
        raise ValueError("No generated responses could be parsed.")

    y_true_binary = valid["ground_truth"].astype(int)
    y_pred = valid["prediction"].astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_binary,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if tp + fn > 0
        else float("nan")
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else float("nan")
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn > 0
        else float("nan")
    )

    metrics = {
        "total_images": len(results),
        "parsed_images": len(valid),
        "unparsed_images": len(results) - len(valid),
        "roc_auc": roc_auc,
        "average_precision": ap,
        "accuracy": accuracy_score(y_true_binary, y_pred),
        "precision": precision_score(
            y_true_binary,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true_binary,
            y_pred,
            zero_division=0,
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tpr": sensitivity,
        "fpr": fpr,
        "f1": f1_score(
            y_true_binary,
            y_pred,
            zero_division=0,
        ),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

    return metrics


def save_metrics(metrics):
    """Save classification metrics to a readable text file."""

    with METRICS_TXT.open("w") as file:
        file.write("Pleural Effusion Classification Benchmark\n")
        file.write("Model: MedGemma 1.5\n")
        file.write(f"Prompt: {PROMPT}\n")
        file.write("=" * 55 + "\n\n")

        file.write("Classification Metrics\n")
        file.write("-" * 55 + "\n")

        for name, value in metrics.items():
            if isinstance(value, float):
                file.write(f"{name}: {value:.6f}\n")
            else:
                file.write(f"{name}: {value}\n")


def plot_results(results, metrics):
    """Generate and save classification evaluation plots."""

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    valid = results.dropna(subset=["prediction"]).copy()

    # 1. Confusion matrix
    if not valid.empty:
        cm = confusion_matrix(
            valid["ground_truth"].astype(int),
            valid["prediction"].astype(int),
            labels=[0, 1],
        )

        fig, ax = plt.subplots(figsize=(6, 5))

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Absent (0)", "Present (1)"],
        )

        display.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
        )

        ax.set_title("Pleural Effusion - Confusion Matrix")

        fig.tight_layout()
        fig.savefig(
            PLOTS_DIR / "confusion_matrix.png",
            dpi=300,
        )
        plt.close(fig)

    y_true = results["ground_truth"].astype(int)
    y_score = results["yes_score"].astype(float)

    if y_true.nunique() == 2:
        # 2. ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_score)

        fig, ax = plt.subplots(figsize=(6, 5))

        ax.plot(
            fpr,
            tpr,
            label=f"ROC-AUC = {metrics['roc_auc']:.4f}",
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Chance",
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Pleural Effusion - ROC Curve")
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            PLOTS_DIR / "roc_curve.png",
            dpi=300,
        )
        plt.close(fig)

        # 3. Precision-Recall curve
        precision, recall, _ = precision_recall_curve(
            y_true,
            y_score,
        )

        fig, ax = plt.subplots(figsize=(6, 5))

        ax.plot(
            recall,
            precision,
            label=f"AP = {metrics['average_precision']:.4f}",
        )

        ax.axhline(
            y=y_true.mean(),
            linestyle="--",
            label="Positive prevalence",
        )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Pleural Effusion - Precision-Recall Curve")
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            PLOTS_DIR / "pr_curve.png",
            dpi=300,
        )
        plt.close(fig)

    # 4. Score distribution
    fig, ax = plt.subplots(figsize=(7, 5))

    for label, name in [
        (0, "Ground truth: Absent"),
        (1, "Ground truth: Present"),
    ]:
        scores = results.loc[
            results["ground_truth"] == label,
            "yes_score",
        ].astype(float)

        ax.hist(
            scores,
            bins=20,
            alpha=0.6,
            label=name,
        )

    ax.set_xlabel("Yes-vs-No Score")
    ax.set_ylabel("Number of Images")
    ax.set_title("Pleural Effusion - Score Distribution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / "score_distribution.png",
        dpi=300,
    )
    plt.close(fig)


def main():
    # Load validation dataset.
    df = load_dataset()

    print("Total frontal images:", len(df))

    print("\nPleural Effusion labels:")
    print(df["Pleural Effusion"].value_counts())

    print("\nFixed prompt:")
    print(PROMPT)

    # Create output directory.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Prevent accidental overwriting of existing predictions.
    if PREDICTIONS_CSV.exists():
        raise FileExistsError(
            f"{PREDICTIONS_CSV} already exists. "
            "Remove or rename it before starting a new run."
        )

    # Load MedGemma once.
    bundle = load_model()

    # Run inference and save each completed image.
    with PREDICTIONS_CSV.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=PREDICTION_COLUMNS,
        )

        writer.writeheader()
        file.flush()

        for index, sample in df.iterrows():
            image = load_image(sample["image_path"])

            response = generate(
                bundle,
                image,
                PROMPT,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            prediction = parse_prediction(response)

            yes_score = compute_yes_score(
                bundle,
                image,
                PROMPT,
            )

            result = {
                "image_path": str(sample["image_path"]),
                "ground_truth": int(sample["Pleural Effusion"]),
                "response": response,
                "prediction": prediction,
                "yes_score": yes_score,
            }

            # Save each completed prediction.
            writer.writerow(result)
            file.flush()

            # Print image-level results.
            print(f"\nImage {index + 1}/{len(df)}")
            print("Image:", result["image_path"])
            print("Ground truth:", result["ground_truth"])
            print("MedGemma response:", result["response"])
            print("Parsed prediction:", result["prediction"])
            print("Yes-vs-No score:", result["yes_score"])

    # Read completed predictions.
    results = pd.read_csv(PREDICTIONS_CSV)

    # Compute classification metrics.
    metrics = compute_metrics(results)

    # Save metrics to TXT.
    save_metrics(metrics)

    # Generate plots.
    plot_results(results, metrics)

    # Print final metrics.
    print("\nClassification Metrics")
    print("-" * 40)

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.6f}")
        else:
            print(f"{name}: {value}")

    print("\nPredictions saved to:", PREDICTIONS_CSV)
    print("Metrics saved to:", METRICS_TXT)
    print("Plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
