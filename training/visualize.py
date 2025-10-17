"""
Visualization Script

Creates visualizations of training progress and reasoning trajectories.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict
import logging
from pathlib import Path

from src.refinement_engine import ReasoningResult

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_training_curves(
    training_history: List[Dict],
    output_path: str = "results/training_curves.png"
):
    """
    Plot training and validation loss curves.
    
    Args:
        training_history: List of epoch metrics
        output_path: Path to save figure
    """
    epochs = [h['epoch'] for h in training_history]
    train_losses = [h['train_loss'] for h in training_history]
    val_losses = [h['val_loss'] for h in training_history]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, train_losses, label='Training Loss', marker='o', linewidth=2)
    ax.plot(epochs, val_losses, label='Validation Loss', marker='s', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved training curves: {output_path}")


def plot_refinement_trajectory(
    result: ReasoningResult,
    ground_truth: np.ndarray,
    output_path: str = "results/refinement_trajectory.png"
):
    """
    Plot how solution evolves through refinement steps.
    
    Args:
        result: Reasoning result containing refinement history
        ground_truth: True solution for comparison
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data from refinement history
    steps = [state.step for state in result.refinement_history]
    confidences = [state.confidence for state in result.refinement_history]
    valid = [state.symbolic_valid for state in result.refinement_history]
    
    # Compute errors
    errors = []
    for state in result.refinement_history:
        hypothesis = state.hypothesis.cpu().numpy().flatten()
        error = np.mean((hypothesis - ground_truth) ** 2)
        errors.append(error)
    
    # Plot 1: Confidence over steps
    ax1 = axes[0, 0]
    ax1.plot(steps, confidences, marker='o', linewidth=2, markersize=8)
    ax1.axhline(y=0.9, color='r', linestyle='--', label='Threshold')
    ax1.set_xlabel('Refinement Step', fontsize=11)
    ax1.set_ylabel('Neural Confidence', fontsize=11)
    ax1.set_title('Confidence Evolution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error over steps
    ax2 = axes[0, 1]
    ax2.plot(steps, errors, marker='s', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Refinement Step', fontsize=11)
    ax2.set_ylabel('Mean Squared Error', fontsize=11)
    ax2.set_title('Error Reduction', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Symbolic validation
    ax3 = axes[1, 0]
    colors = ['green' if v else 'red' for v in valid]
    ax3.bar(steps, [1]*len(steps), color=colors, alpha=0.6)
    ax3.set_xlabel('Refinement Step', fontsize=11)
    ax3.set_ylabel('Valid', fontsize=11)
    ax3.set_title('Symbolic Verification', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1.2])
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Invalid', 'Valid'])
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Solution comparison (first few dimensions)
    ax4 = axes[1, 1]
    final_solution = result.refinement_history[-1].hypothesis.cpu().numpy().flatten()[:5]
    ground_truth_subset = ground_truth[:5]
    
    x = np.arange(len(final_solution))
    width = 0.35
    
    ax4.bar(x - width/2, ground_truth_subset, width, label='Ground Truth', alpha=0.8)
    ax4.bar(x + width/2, final_solution, width, label='Final Solution', alpha=0.8)
    ax4.set_xlabel('Dimension', fontsize=11)
    ax4.set_ylabel('Value', fontsize=11)
    ax4.set_title('Solution Comparison (First 5 Dims)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved refinement trajectory: {output_path}")


def plot_evaluation_summary(
    metrics: Dict,
    output_path: str = "results/evaluation_summary.png"
):
    """
    Plot summary of evaluation metrics.
    
    Args:
        metrics: Dictionary of evaluation metrics
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Success rate
    ax1 = axes[0, 0]
    success_rate = metrics['success_rate']
    ax1.bar(['Success'], [success_rate], color='green', alpha=0.7)
    ax1.bar(['Failure'], [1 - success_rate], bottom=[success_rate], color='red', alpha=0.7)
    ax1.set_ylabel('Rate', fontsize=11)
    ax1.set_title('Success Rate', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.text(0, success_rate/2, f'{success_rate:.1%}', ha='center', fontsize=14, fontweight='bold')
    
    # Plot 2: Average steps
    ax2 = axes[0, 1]
    refinement_stats = metrics.get('refinement_stats', {})
    if refinement_stats:
        avg_steps = refinement_stats.get('avg_steps', 0)
        ax2.bar(['Avg Steps'], [avg_steps], color='blue', alpha=0.7)
        ax2.set_ylabel('Number of Steps', fontsize=11)
        ax2.set_title('Average Refinement Steps', fontsize=12, fontweight='bold')
        ax2.text(0, avg_steps/2, f'{avg_steps:.2f}', ha='center', fontsize=14, fontweight='bold')
    
    # Plot 3: Convergence reasons
    ax3 = axes[1, 0]
    convergence_reasons = metrics.get('convergence_reasons', {})
    if convergence_reasons:
        reasons = list(convergence_reasons.keys())
        counts = list(convergence_reasons.values())
        ax3.barh(reasons, counts, color='purple', alpha=0.7)
        ax3.set_xlabel('Count', fontsize=11)
        ax3.set_title('Convergence Reasons', fontsize=12, fontweight='bold')
    
    # Plot 4: Component statistics
    ax4 = axes[1, 1]
    symbolic_stats = metrics.get('symbolic_stats', {})
    if symbolic_stats:
        valid_rate = symbolic_stats.get('valid_rate', 0)
        avg_violations = symbolic_stats.get('avg_violations', 0)
        
        x = ['Valid Rate', 'Avg Violations']
        y = [valid_rate, avg_violations / 5]  # Normalize violations for display
        colors = ['green', 'orange']
        
        ax4.bar(x, y, color=colors, alpha=0.7)
        ax4.set_ylabel('Value', fontsize=11)
        ax4.set_title('Symbolic Verification Stats', fontsize=12, fontweight='bold')
        ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved evaluation summary: {output_path}")


def plot_metacognitive_adaptation(
    metacognitive_stats: Dict,
    output_path: str = "results/metacognitive_adaptation.png"
):
    """
    Plot how metacognitive controller adapted its weights over time.
    
    Args:
        metacognitive_stats: Statistics from metacognitive controller
        output_path: Path to save figure
    """
    if not metacognitive_stats:
        logger.warning("No metacognitive statistics available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract weight evolution if available
    # This would require storing weight history during evaluation
    # For now, show final weights
    neural_weight = metacognitive_stats.get('current_neural_weight', 0.5)
    symbolic_weight = metacognitive_stats.get('current_symbolic_weight', 0.5)
    
    weights = [neural_weight, symbolic_weight]
    labels = ['Neural', 'Symbolic']
    colors = ['blue', 'green']
    
    ax.bar(labels, weights, color=colors, alpha=0.7)
    ax.set_ylabel('Weight', fontsize=12)
    ax.set_title('Final Component Weights', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    
    for i, (label, weight) in enumerate(zip(labels, weights)):
        ax.text(i, weight/2, f'{weight:.3f}', ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved metacognitive adaptation: {output_path}")