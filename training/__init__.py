"""
Training Module

Contains training, evaluation, and visualization functionality.
"""

from training.train import DeepSupervisionTrainer, train_model
from training.evaluate import evaluate_model, compute_metrics
from training.visualize import plot_training_curves, plot_refinement_trajectory

__all__ = [
    "DeepSupervisionTrainer",
    "train_model",
    "evaluate_model",
    "compute_metrics",
    "plot_training_curves",
    "plot_refinement_trajectory",
]