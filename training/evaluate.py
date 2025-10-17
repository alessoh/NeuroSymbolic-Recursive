"""
Evaluation Script

Evaluates trained models on test data with full reasoning process.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List
import logging
import argparse
from pathlib import Path

from src.neural_component import NeuralReasoner
from src.symbolic_component import SymbolicVerifier
from src.refinement_engine import RefinementEngine, ReasoningResult
from src.metacognitive_controller import MetacognitiveController
from src.utils import load_checkpoint, save_metrics, get_device, setup_logging

logger = logging.getLogger(__name__)


def compute_metrics(results: List[ReasoningResult], solutions: torch.Tensor) -> Dict[str, float]:
    """
    Compute evaluation metrics from reasoning results.
    
    Args:
        results: List of reasoning results
        solutions: Ground truth solutions [num_samples, output_dim]
        
    Returns:
        Dictionary of metrics
    """
    num_samples = len(results)
    
    # Success rate
    success_count = sum(1 for r in results if r.success)
    success_rate = success_count / num_samples
    
    # Average steps taken
    avg_steps = sum(r.steps_taken for r in results) / num_samples
    
    # Average final confidence
    avg_confidence = sum(r.final_confidence for r in results) / num_samples
    
    # Solution accuracy (MSE)
    predictions = torch.stack([r.solution for r in results])
    mse = torch.mean((predictions - solutions) ** 2).item()
    
    # Convergence reasons distribution
    convergence_reasons = {}
    for r in results:
        reason = r.convergence_reason.split(':')[0]  # Get first part
        convergence_reasons[reason] = convergence_reasons.get(reason, 0) + 1
    
    return {
        'success_rate': success_rate,
        'avg_steps': avg_steps,
        'avg_confidence': avg_confidence,
        'mse': mse,
        'rmse': mse ** 0.5,
        'convergence_reasons': convergence_reasons
    }


def evaluate_model(
    checkpoint_path: str,
    test_loader: DataLoader,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    max_iterations: int = 5,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Evaluate a trained model on test data.
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_loader: DataLoader for test data
        input_dim: Model input dimension
        hidden_dim: Model hidden dimension
        output_dim: Model output dimension
        max_iterations: Maximum refinement iterations
        verbose: Whether to log detailed progress
        
    Returns:
        Dictionary containing evaluation results and metrics
    """
    device = get_device()
    
    # Initialize model
    model = NeuralReasoner(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    ).to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    model.eval()
    
    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Initialize components
    symbolic_verifier = SymbolicVerifier()
    refinement_engine = RefinementEngine(
        neural_model=model,
        symbolic_verifier=symbolic_verifier,
        max_iterations=max_iterations
    )
    metacognitive = MetacognitiveController()
    
    # Collect results
    all_results = []
    all_solutions = []
    
    logger.info("Starting evaluation...")
    
    with torch.no_grad():
        for batch_idx, (puzzles, solutions, targets) in enumerate(test_loader):
            puzzles = puzzles.to(device)
            targets = targets.to(device)
            
            # Run reasoning for each sample in batch
            for i in range(puzzles.size(0)):
                puzzle = puzzles[i:i+1]
                target = targets[i:i+1]
                solution = solutions[i]
                
                # Execute reasoning
                result = refinement_engine.reason(
                    puzzle,
                    target,
                    verbose=False
                )
                
                all_results.append(result)
                all_solutions.append(solution)
                
                # Update metacognitive controller
                metacognitive.update_weights(
                    neural_confidence=result.final_confidence,
                    symbolic_valid=result.refinement_history[-1].symbolic_valid,
                    actual_correct=None  # No ground truth comparison here
                )
            
            if verbose and (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {(batch_idx + 1) * test_loader.batch_size} samples")
    
    # Compute metrics
    all_solutions_tensor = torch.stack(all_solutions)
    metrics = compute_metrics(all_results, all_solutions_tensor)
    
    # Add component statistics
    metrics['refinement_stats'] = refinement_engine.get_statistics()
    metrics['symbolic_stats'] = symbolic_verifier.get_statistics()
    metrics['metacognitive_stats'] = metacognitive.get_statistics()
    
    # Log results
    logger.info("="*60)
    logger.info("Evaluation Results")
    logger.info("="*60)
    logger.info(f"Success Rate: {metrics['success_rate']:.2%}")
    logger.info(f"Average Steps: {metrics['avg_steps']:.2f}")
    logger.info(f"Average Confidence: {metrics['avg_confidence']:.3f}")
    logger.info(f"RMSE: {metrics['rmse']:.4f}")
    logger.info(f"Convergence Reasons: {metrics['convergence_reasons']}")
    
    return {
        'metrics': metrics,
        'results': all_results,
        'checkpoint_info': checkpoint
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NeuroSymbolic-Recursive model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=16,
        help="Input dimension"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension"
    )
    parser.add_argument(
        "--output_dim",
        type=int,
        default=9,
        help="Output dimension"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of test samples"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=5,
        help="Maximum refinement iterations"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Create test data
    from training.train import create_synthetic_data
    _, test_loader = create_synthetic_data(
        num_samples=args.num_samples,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        train_split=0.0  # Use all as test
    )
    
    # Evaluate
    evaluation = evaluate_model(
        checkpoint_path=args.checkpoint,
        test_loader=test_loader,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        max_iterations=args.max_iterations
    )
    
    # Save results
    save_metrics(evaluation['metrics'], "evaluation_results.json")