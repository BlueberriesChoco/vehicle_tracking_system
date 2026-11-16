#!/usr/bin/env python
"""Train a Transformer trajectory predictor on behavior vector CSVs.

The model predicts the next ``--future-len`` positions from the past
``--past-len`` positions. Prediction residual is used as a trajectory anomaly
signal (research point 2, path 1).

Usage:
    python scripts/train_transformer.py \
        --input data/outputs/vectors/admin_in \
        --output data/references/traj_admin_in \
        --epochs 40
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.anomaly import TrajectoryPredictionDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


from src.utils.vectors_io import load_vector_rows as load_rows


def main():
    parser = argparse.ArgumentParser(description="Train Transformer trajectory predictor")
    parser.add_argument("--input", required=True, help="A vec_*.csv file or directory")
    parser.add_argument("--output", required=True, help="Output model prefix (writes .pt and .json)")
    parser.add_argument("--past-len", type=int, default=5, help="Past positions used as input")
    parser.add_argument("--future-len", type=int, default=5, help="Future positions to predict")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--contamination", type=float, default=0.05, help="Expected anomaly fraction for residual threshold")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        logger.error("No vector rows found: %s", args.input)
        sys.exit(1)

    detector = TrajectoryPredictionDetector(
        past_len=args.past_len,
        future_len=args.future_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        contamination=args.contamination,
        device=args.device,
        random_state=args.seed,
    )
    detector.fit(rows)
    detector.save(args.output)

    history = detector.training_history
    first = history[0] if history else {}
    last = history[-1] if history else {}
    print(f"Model saved to: {args.output}.pt / {args.output}.json")
    print(f"Windows trained, epochs: {len(history)}")
    print(f"Train loss: {first.get('train_loss')} -> {last.get('train_loss')}")
    print(f"Val loss:   {first.get('val_loss')} -> {last.get('val_loss')}")


if __name__ == "__main__":
    main()
