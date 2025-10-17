"""
Simple Example

Demonstrates basic usage of the NeuroSymbolic-Recursive system.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neural_component import NeuralReasoner
from src.symbolic_component import SymbolicVerifier
from src.refinement_engine import RefinementEngine
from src.metacognitive_controller import MetacognitiveController
from src.utils import setup_logging


def create_sample_puzzle(input_dim=16, output_dim=9):
    """Create a simple sample puzzle"""
    puzzle = torch.randn(1, input_dim)
    target = torch.tensor([[4.5]])  # Sum should equal 4.5
    return puzzle, target


def main():
    """Run simple example demonstration"""
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    
    print("="*70)
    print("NeuroSymbolic-Recursive System - Simple Example")
    print("="*70)
    print()
    
    # Configuration
    input_dim = 16
    hidden_dim = 64
    output_dim = 9
    
    print("Configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Output dimension: {output_dim}")
    print()
    
    # Initialize components
    print("Initializing components...")
    
    neural_model = NeuralReasoner(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    print(f"  ✓ Neural Reasoner ({neural_model.count_parameters():,} parameters)")
    
    symbolic_verifier = SymbolicVerifier()
    print(f"  ✓ Symbolic Verifier ({len(symbolic_verifier.rules)} rules)")
    
    refinement_engine = RefinementEngine(
        neural_model=neural_model,
        symbolic_verifier=symbolic_verifier,
        max_iterations=5,
        confidence_threshold=0.9
    )
    print(f"  ✓ Refinement Engine (max 5 iterations)")
    
    metacognitive = MetacognitiveController()
    print(f"  ✓ Metacognitive Controller")
    print()
    
    # Create sample puzzle
    print("Creating sample puzzle...")
    puzzle, target = create_sample_puzzle(input_dim, output_dim)
    print(f"  Puzzle shape: {puzzle.shape}")
    print(f"  Target constraint: sum should equal {target.item():.2f}")
    print()
    
    # Execute reasoning
    print("="*70)
    print("Executing Recursive Reasoning Process")
    print("="*70)
    print()
    
    result = refinement_engine.reason(puzzle, target, verbose=True)
    
    print()
    print("="*70)
    print("Reasoning Results")
    print("="*70)
    print(f"Success: {result.success}")
    print(f"Steps taken: {result.steps_taken}")
    print(f"Final confidence: {result.final_confidence:.3f}")
    print(f"Convergence reason: {result.convergence_reason}")
    print()
    
    # Show refinement progression
    print("Refinement Progression:")
    print("-"*70)
    for state in result.refinement_history:
        valid_str = "✓" if state.symbolic_valid else "✗"
        violations_str = f" ({len(state.violations)} violations)" if state.violations else ""
        print(f"Step {state.step}: confidence={state.confidence:.3f}, "
              f"valid={valid_str}{violations_str}")
    print()
    
    # Display final solution
    final_solution = result.solution.flatten().detach().numpy()
    print("Final Solution (first 5 values):")
    print(f"  {final_solution[:5]}")
    print(f"  Sum: {final_solution.sum():.3f} (target: {target.item():.2f})")
    print()
    
    # Component statistics
    print("="*70)
    print("Component Statistics")
    print("="*70)
    
    refinement_stats = refinement_engine.get_statistics()
    print(f"Refinement Engine:")
    print(f"  Total runs: {refinement_stats.get('total_runs', 1)}")
    print(f"  Success rate: {refinement_stats.get('success_rate', result.success):.1%}")
    print(f"  Avg steps: {refinement_stats.get('avg_steps', result.steps_taken):.2f}")
    print()
    
    symbolic_stats = symbolic_verifier.get_statistics()
    if symbolic_stats:
        print(f"Symbolic Verifier:")
        print(f"  Total verifications: {symbolic_stats.get('total_verifications', 0)}")
        print(f"  Valid rate: {symbolic_stats.get('valid_rate', 0):.1%}")
        print(f"  Avg violations: {symbolic_stats.get('avg_violations', 0):.2f}")
        print()
    
    metacognitive_stats = metacognitive.get_statistics()
    if metacognitive_stats:
        print(f"Metacognitive Controller:")
        print(f"  Neural weight: {metacognitive.neural_weight:.3f}")
        print(f"  Symbolic weight: {metacognitive.symbolic_weight:.3f}")
        print()
    
    print("="*70)
    print("Example Complete!")
    print("="*70)


if __name__ == "__main__":
    main()