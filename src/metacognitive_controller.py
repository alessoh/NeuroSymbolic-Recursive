"""
Metacognitive Controller Module

Makes strategic decisions about the reasoning process.
"""

from enum import Enum
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ReasoningStrategy(Enum):
    """Enumeration of available reasoning strategies"""
    NEURAL_FIRST = "neural_first"
    SYMBOLIC_FIRST = "symbolic_first"
    BALANCED = "balanced"


class MetacognitiveController:
    """
    High-level controller that makes strategic decisions about reasoning.
    
    This controller monitors the reasoning process and adaptively adjusts
    how much to rely on neural versus symbolic components based on
    performance and problem characteristics.
    
    Args:
        initial_strategy: Starting reasoning strategy
        learning_rate: Rate of adaptation for component weights
    """
    
    def __init__(
        self,
        initial_strategy: ReasoningStrategy = ReasoningStrategy.BALANCED,
        learning_rate: float = 0.05
    ):
        self.strategy = initial_strategy
        self.learning_rate = learning_rate
        
        # Component weights (how much to trust each component)
        self.neural_weight = 0.5
        self.symbolic_weight = 0.5
        
        # Performance tracking
        self.decision_history = []
        self.performance_stats = {
            'neural_correct': 0,
            'symbolic_correct': 0,
            'both_correct': 0,
            'both_wrong': 0,
            'total_decisions': 0
        }
        
        logger.info(
            f"Initialized MetacognitiveController: strategy={initial_strategy.value}, "
            f"learning_rate={learning_rate}"
        )
    
    def select_strategy(
        self,
        puzzle_complexity: float,
        time_budget: float,
        problem_type: Optional[str] = None
    ) -> ReasoningStrategy:
        """
        Select appropriate reasoning strategy based on context.
        
        Args:
            puzzle_complexity: Estimated complexity (0-1)
            time_budget: Available computation time (0-1)
            problem_type: Optional problem type identifier
            
        Returns:
            Selected reasoning strategy
        """
        # Simple heuristic-based selection for initial implementation
        
        if puzzle_complexity > 0.7:
            # Complex problems benefit from explicit symbolic reasoning
            if time_budget > 0.5:
                strategy = ReasoningStrategy.SYMBOLIC_FIRST
            else:
                # Limited time, use balanced approach
                strategy = ReasoningStrategy.BALANCED
        
        elif puzzle_complexity < 0.3:
            # Simple problems can rely on neural pattern recognition
            strategy = ReasoningStrategy.NEURAL_FIRST
        
        else:
            # Medium complexity, use balanced approach
            strategy = ReasoningStrategy.BALANCED
        
        # Override based on learned preferences for problem type
        if problem_type and problem_type in self._get_problem_preferences():
            preferred = self._get_problem_preferences()[problem_type]
            strategy = preferred
        
        self.strategy = strategy
        logger.debug(
            f"Selected strategy: {strategy.value} "
            f"(complexity={puzzle_complexity:.2f}, budget={time_budget:.2f})"
        )
        
        return strategy
    
    def _get_problem_preferences(self) -> Dict[str, ReasoningStrategy]:
        """Get learned preferences for different problem types"""
        # This would be learned over time; using defaults for now
        return {
            'logic_puzzle': ReasoningStrategy.SYMBOLIC_FIRST,
            'pattern_recognition': ReasoningStrategy.NEURAL_FIRST,
            'hybrid_task': ReasoningStrategy.BALANCED
        }
    
    def update_weights(
        self,
        neural_confidence: float,
        symbolic_valid: bool,
        actual_correct: Optional[bool] = None
    ):
        """
        Adaptively adjust component weights based on performance.
        
        Args:
            neural_confidence: Confidence from neural component
            symbolic_valid: Whether symbolic verification passed
            actual_correct: Ground truth correctness (if available)
        """
        # Track which components were correct
        neural_high_conf = neural_confidence > 0.8
        
        # Update statistics
        self.performance_stats['total_decisions'] += 1
        
        if actual_correct is not None:
            # We have ground truth - update based on correctness
            if actual_correct and neural_high_conf:
                self.performance_stats['neural_correct'] += 1
                # Increase neural weight
                self.neural_weight = min(0.95, self.neural_weight + self.learning_rate)
            
            if actual_correct and symbolic_valid:
                self.performance_stats['symbolic_correct'] += 1
                # Increase symbolic weight
                self.symbolic_weight = min(0.95, self.symbolic_weight + self.learning_rate)
            
            if actual_correct and neural_high_conf and symbolic_valid:
                self.performance_stats['both_correct'] += 1
            
            if not actual_correct:
                # Wrong answer - decrease weights of components that were confident
                if neural_high_conf:
                    self.neural_weight = max(0.05, self.neural_weight - self.learning_rate)
                if symbolic_valid:
                    self.symbolic_weight = max(0.05, self.symbolic_weight - self.learning_rate)
        
        else:
            # No ground truth - use agreement as proxy
            if neural_high_conf and symbolic_valid:
                # Both agree - slightly increase both weights
                self.performance_stats['both_correct'] += 1
                self.neural_weight = min(0.95, self.neural_weight + self.learning_rate * 0.5)
                self.symbolic_weight = min(0.95, self.symbolic_weight + self.learning_rate * 0.5)
            
            elif neural_high_conf and not symbolic_valid:
                # Disagreement - favor symbolic for safety
                self.symbolic_weight = min(0.95, self.symbolic_weight + self.learning_rate)
                self.neural_weight = max(0.05, self.neural_weight - self.learning_rate * 0.5)
        
        # Normalize weights to sum to 1
        total = self.neural_weight + self.symbolic_weight
        self.neural_weight /= total
        self.symbolic_weight /= total
        
        # Record decision
        self.decision_history.append({
            'neural_weight': self.neural_weight,
            'symbolic_weight': self.symbolic_weight,
            'neural_confidence': neural_confidence,
            'symbolic_valid': symbolic_valid,
            'actual_correct': actual_correct
        })
        
        logger.debug(
            f"Updated weights: neural={self.neural_weight:.3f}, "
            f"symbolic={self.symbolic_weight:.3f}"
        )
    
    def should_continue_refinement(
        self,
        current_step: int,
        max_steps: int,
        confidence: float,
        valid: bool,
        improvement_rate: float
    ) -> bool:
        """
        Decide whether to continue refinement or stop.
        
        Args:
            current_step: Current refinement step
            max_steps: Maximum allowed steps
            confidence: Current neural confidence
            valid: Current symbolic validity
            improvement_rate: Rate of improvement in recent steps
            
        Returns:
            Whether to continue refining
        """
        # Don't exceed maximum steps
        if current_step >= max_steps:
            return False
        
        # If solution is valid and confident, stop
        if valid and confidence > 0.9:
            return False
        
        # If improvement has stalled, stop
        if improvement_rate < 0.01 and current_step > 2:
            logger.debug("Stopping: improvement stalled")
            return False
        
        # If solution is invalid and confidence is very low, might be stuck
        if not valid and confidence < 0.3 and current_step > max_steps // 2:
            logger.debug("Stopping: low confidence and invalid")
            return False
        
        # Otherwise, continue refining
        return True
    
    def get_statistics(self) -> Dict[str, float]:
        """Get performance statistics"""
        total = self.performance_stats['total_decisions']
        if total == 0:
            return {}
        
        return {
            'total_decisions': total,
            'neural_correct_rate': self.performance_stats['neural_correct'] / total,
            'symbolic_correct_rate': self.performance_stats['symbolic_correct'] / total,
            'both_correct_rate': self.performance_stats['both_correct'] / total,
            'current_neural_weight': self.neural_weight,
            'current_symbolic_weight': self.symbolic_weight,
            'avg_neural_weight': sum(d['neural_weight'] for d in self.decision_history) / len(self.decision_history) if self.decision_history else 0.5,
            'avg_symbolic_weight': sum(d['symbolic_weight'] for d in self.decision_history) / len(self.decision_history) if self.decision_history else 0.5
        }
    
    def reset_statistics(self):
        """Reset performance statistics"""
        self.decision_history = []
        self.performance_stats = {
            'neural_correct': 0,
            'symbolic_correct': 0,
            'both_correct': 0,
            'both_wrong': 0,
            'total_decisions': 0
        }
        logger.info("Reset metacognitive controller statistics")
    
    def get_current_strategy(self) -> ReasoningStrategy:
        """Get the current reasoning strategy"""
        return self.strategy
    
    def set_strategy(self, strategy: ReasoningStrategy):
        """Manually set the reasoning strategy"""
        self.strategy = strategy
        logger.info(f"Strategy set to: {strategy.value}")