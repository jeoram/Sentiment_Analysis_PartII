"""
Model Evaluation Script
Evaluates model performance on the app reviews dataset
"""

import argparse
import json
import os
import sys

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.model import SentimentModel, score_to_sentiment

# Paths
DATA_DIR = os.environ.get("DATA_PATH", "data")
DATASET_FILE = os.path.join(DATA_DIR, "dataset.csv")


def load_test_data():
    """Load test data from dataset"""
    if os.path.exists(DATASET_FILE):
        print(f"Loading dataset from {DATASET_FILE}")
        df = pd.read_csv(DATASET_FILE)
        df = df.dropna(subset=["content", "score"])
        df["content"] = df["content"].astype(str)
        df["score"] = df["score"].astype(int)
        df["sentiment"] = df["score"].apply(score_to_sentiment)

        # Use 20% for testing
        _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["sentiment"])

        return test_df["content"].tolist(), test_df["sentiment"].tolist()
    else:
        # Fallback sample data
        print("Dataset not found, using sample test data")
        return [
            ("I love this product! It's amazing!", "positive"),
            ("This is the best thing ever!", "positive"),
            ("Great quality and fast delivery", "positive"),
            ("Absolutely wonderful experience", "positive"),
            ("I hate this, it's terrible", "negative"),
            ("Worst purchase I ever made", "negative"),
            ("Very disappointed with the quality", "negative"),
            ("This product is awful", "negative"),
            ("It's okay, nothing special", "neutral"),
            ("Average product, does the job", "neutral"),
            ("Not bad, not great either", "neutral"),
            ("It's fine I guess", "neutral"),
        ]


def evaluate_model(output_file: str = "metrics.json") -> dict:
    """
    Evaluate the sentiment model and save metrics

    Args:
        output_file: Path to save metrics JSON

    Returns:
        Dictionary of metrics
    """
    # Initialize and load model
    model = SentimentModel()
    model.load()

    # Load test data
    test_data = load_test_data()

    if isinstance(test_data[0], tuple):
        # Sample data format
        texts = [t[0] for t in test_data]
        true_labels = [t[1] for t in test_data]
    else:
        # Dataset format
        texts, true_labels = test_data

    print(f"Evaluating on {len(texts)} samples...")

    # Get predictions
    predicted_labels = []
    for text in texts:
        result = model.predict(text)
        predicted_labels.append(result["sentiment"])

    # Calculate metrics
    metrics = {
        "accuracy": round(accuracy_score(true_labels, predicted_labels), 4),
        "f1_score": round(f1_score(true_labels, predicted_labels, average="weighted"), 4),
        "precision": round(precision_score(true_labels, predicted_labels, average="weighted"), 4),
        "recall": round(recall_score(true_labels, predicted_labels, average="weighted"), 4),
        "num_samples": len(texts),
        "predictions": {
            "correct": sum(1 for t, p in zip(true_labels, predicted_labels) if t == p),
            "incorrect": sum(1 for t, p in zip(true_labels, predicted_labels) if t != p),
        },
    }

    # Print detailed report
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"Samples:   {metrics['num_samples']}")
    print("=" * 60)

    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels))

    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predicted_labels))
    print()

    # Save metrics to file
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {output_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate sentiment model")
    parser.add_argument("--output", "-o", default="metrics.json", help="Output file for metrics")
    args = parser.parse_args()

    metrics = evaluate_model(args.output)

    # Exit with error if accuracy is too low
    if metrics["accuracy"] < 0.5:
        print("[WARNING] Model accuracy is below 50%!")
        sys.exit(1)

    print(f"[PASS] Model evaluation passed with {metrics['accuracy']*100:.1f}% accuracy")
    sys.exit(0)


if __name__ == "__main__":
    main()
