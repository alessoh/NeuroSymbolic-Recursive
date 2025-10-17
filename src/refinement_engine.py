"""
Refinement Engine Module

Coordinates recursive refinement between neural and symbolic components.
"""

import torch
from typing import Optional, List
from dataclasses import dataclass
import logging

from src.neural_component import NeuralReasoner
from src.symbolic_component import SymbolicVerifier

logger = logging.getLogger(__name__)


@dataclass
class RefinementState:
    """
    Represents the state at a refinement step.
    
    Attributes:
        hypothesis: Current solution hypothesis
        confidence: Neural confidence score
        symbolic_valid: Whether symbolic verification passed
        violations: List of constraint violations
        step: Refinement step number
    """
    hypothesis: torch.Tensor
    confidence: float
    symbolic_valid: bool
    violations: List[str]
    step: int


@dataclass
class ReasoningResult:
    """
    Final result of the reasoning process.
    
    Attributes:
        solution: Final solution tensor
        steps_taken: Number of refinement steps executed
        final_confidence: Final neural confidence score
        refinement_history: List of all refinement states
        success: Whether reasoning succeeded
        convergence_reason: Why reasoning stopped
    """
    solution: torch.Tensor
    steps_taken: int
    final_confidence: float
    refinement_history: List[RefinementState]
    success: bool
    convergence_reason: str


class RefinementEngine:
    """
    Coordinates recursive refinement between neural and symbolic components.
    
    This engine implements the core reasoning loop that alternates between
    neural proposal and symbolic verification, progressively refining solutions.
    
    Args:
        neural_model: Neural reasoner component
        symbolic_verifier: Symbolic verifier component
        max_iterations: Maximum number of refinement cycles
        confidence_threshold: Minimum confidence for early stopping
        convergence_tolerance: Tolerance for detecting convergence
    """
    
    def __init__(
        self,
        neural_model: NeuralReasoner,
        symbolic_verifier: SymbolicVerifier,
        max_iterations: int = 5,
        confidence_threshold: float = 0.9,
        convergence_tolerance: float = 0.01
    ):
        self.neural = neural_model
        self.symbolic = symbolic_verifier
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.convergence_tolerance = convergence_tolerance
        
        # Statistics tracking
        self.reasoning_history = []
        
        logger.info(
            f"Initialized RefinementEngine: max_iterations={max_iterations}, "
            f"confidence_threshold={confidence_threshold}, "
            f"convergence_tolerance={convergence_tolerance}"
        )
    
    def reason(
        self,
        puzzle: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> ReasoningResult:
        """
        Execute the recursive reasoning process.
        
        Args:
            puzzle: Input puzzle [batch_size, input_dim]
            target: Target constraints [batch_size, constraint_dim]
            verbose: Whether to log reasoning steps
            
        Returns:
            ReasoningResult containing solution and reasoning trace
        """
        batch_size = puzzle.size(0)
        device = self.neural.get_device()
        
        # Move inputs to model device
        puzzle = puzzle.to(device)
        if target is not None:
            target = target.to(device)
        
        refinement_history = []
        
        # Step 0: Generate initial hypothesis
        with torch.no_grad():
            hypothesis, confidence = self.neural(puzzle)
        
        confidence_val = confidence.mean().item()
        
        # Verify initial hypothesis
        valid, violations = self.symbolic.verify(hypothesis, puzzle, target)
        
        if verbose:
            logger.info(f"Step 0: Initial hypothesis")
            logger.info(f"  Confidence: {confidence_val:.3f}")
            logger.info(f"  Valid: {valid}")
            if violations:
                logger.info(f"  Violations: {len(violations)}")
        
        # Record initial state
        refinement_history.append(RefinementState(
            hypothesis=hypothesis.detach().clone(),
            confidence=confidence_val,
            symbolic_valid=valid,
            violations=violations.copy(),
            step=0
        ))
        
        # Check if initial hypothesis is already good enough
        if valid and confidence_val >= self.confidence_threshold:
            convergence_reason = "Initial hypothesis satisfies all constraints with high confidence"
            if verbose:
                logger.info(f"Converged immediately: {convergence_reason}")
            
            return ReasoningResult(
                solution=hypothesis,
                steps_taken=1,
                final_confidence=confidence_val,
                refinement_history=refinement_history,
                success=True,
                convergence_reason=convergence_reason
            )
        
        # Refinement loop
        previous_hypothesis = hypothesis
        convergence_reason = "Maximum iterations reached"
        
        for step in range(1, self.max_iterations + 1):
            # Encode violations for neural feedback
            violation_encoding = self.symbolic.encode_violations(violations)
            violation_batch = violation_encoding.unsqueeze(0).expand(batch_size, -1).to(device)
            
            # Generate refined hypothesis
            with torch.no_grad():
                hypothesis, confidence = self.neural(
                    puzzle,
                    previous_hypothesis.detach(),
                    violation_batch
                )
            
            confidence_val = confidence.mean().item()
            
            # Verify refined hypothesis
            valid, violations = self.symbolic.verify(hypothesis, puzzle, target)
            
            if verbose:
                logger.info(f"Step {step}: Refined hypothesis")
                logger.info(f"  Confidence: {confidence_val:.3f}")
                logger.info(f"  Valid: {valid}")
                if violations:
                    logger.info(f"  Violations: {violations[:2]}")
            
            # Record refinement state
            refinement_history.append(RefinementState(
                hypothesis=hypothesis.detach().clone(),
                confidence=confidence_val,
                symbolic_valid=valid,
                violations=violations.copy(),
                step=step
            ))
            
            # Check stopping criteria
            
            # 1. Success: Valid and confident
            if valid and confidence_val >= self.confidence_threshold:
                convergence_reason = f"Converged at step {step}: valid with high confidence"
                if verbose:
                    logger.info(convergence_reason)
                break
            
            # 2. Convergence: Hypothesis not changing significantly
            hypothesis_change = torch.abs(hypothesis - previous_hypothesis).mean().item()
            if hypothesis_change < self.convergence_tolerance:
                convergence_reason = f"Converged at step {step}: hypothesis stabilized"
                if verbose:
                    logger.info(convergence_reason)
                break
            
            # 3. Improvement: Valid but low confidence - continue refining
            if valid and confidence_val < self.confidence_threshold:
                if verbose:
                    logger.debug(f"Valid but low confidence, continuing refinement")
            
            # Update for next iteration
            previous_hypothesis = hypothesis
        
        # Determine success
        final_valid = refinement_history[-1].symbolic_valid
        final_confidence = refinement_history[-1].confidence
        success = final_valid and final_confidence >= self.confidence_threshold
        
        # Store in reasoning history
        self.reasoning_history.append({
            'steps': len(refinement_history),
            'success': success,
            'final_confidence': final_confidence,
            'final_valid': final_valid
        })
        
        if verbose:
            logger.info(f"\nReasoning complete:")
            logger.info(f"  Success: {success}")
            logger.info(f"  Steps: {len(refinement_history)}")
            logger.info(f"  Final confidence: {final_confidence:.3f}")
            logger.info(f"  Reason: {convergence_reason}")
        
        return ReasoningResult(
            solution=hypothesis,
            steps_taken=len(refinement_history),
            final_confidence=final_confidence,
            refinement_history=refinement_history,
            success=success,
            convergence_reason=convergence_reason
        )
    
    def get_statistics(self) -> dict:
        """Get statistics about reasoning performance"""
        if not self.reasoning_history:
            return {}
        
        total_runs = len(self.reasoning_history)
        successful_runs = sum(1 for h in self.reasoning_history if h['success'])
        avg_steps = sum(h['steps'] for h in self.reasoning_history) / total_runs
        avg_confidence = sum(h['final_confidence'] for h in self.reasoning_history) / total_runs
        
        return {
            'total_runs': total_runs,
            'success_rate': successful_runs / total_runs,
            'avg_steps': avg_steps,
            'avg_final_confidence': avg_confidence
        }
    
    def reset_statistics(self):
        """Reset reasoning history statistics"""
        self.reasoning_history = []
        logger.info("Reset refinement engine statistics")