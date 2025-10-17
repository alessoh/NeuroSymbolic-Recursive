"""
Symbolic Component Module

Implements the symbolic verifier that checks solutions against logical constraints.
"""

import torch
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """
    Represents a logical constraint rule.
    
    Attributes:
        name: Unique identifier for the rule
        description: Human-readable description
        check_fn: Function that checks if solution satisfies the rule
        violation_encoding: Index in violation vector
    """
    name: str
    description: str
    check_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], bool]
    violation_encoding: int


class SymbolicVerifier:
    """
    Symbolic reasoning component that verifies solutions against logical rules.
    
    This component maintains a knowledge base of constraints and provides
    explicit verification with detailed feedback about violations.
    
    Args:
        rules: List of Rule objects to enforce
        max_violations: Maximum number of violation types to track
    """
    
    def __init__(self, rules: Optional[List[Rule]] = None, max_violations: int = 10):
        self.rules = rules if rules is not None else self._create_default_rules()
        self.max_violations = max_violations
        self.violation_history = []
        
        logger.info(f"Initialized SymbolicVerifier with {len(self.rules)} rules")
        for rule in self.rules:
            logger.debug(f"  Rule: {rule.name} - {rule.description}")
    
    def _create_default_rules(self) -> List[Rule]:
        """Create default constraint rules for demonstration"""
        rules = [
            Rule(
                name="sum_constraint",
                description="Sum of values must equal target",
                check_fn=self._check_sum_constraint,
                violation_encoding=0
            ),
            Rule(
                name="range_constraint",
                description="All values must be in valid range [0, 1]",
                check_fn=self._check_range_constraint,
                violation_encoding=1
            ),
            Rule(
                name="sparsity_constraint",
                description="Solution should not be too dense",
                check_fn=self._check_sparsity_constraint,
                violation_encoding=2
            ),
            Rule(
                name="smoothness_constraint",
                description="Adjacent values should not differ too much",
                check_fn=self._check_smoothness_constraint,
                violation_encoding=3
            )
        ]
        return rules
    
    def _check_sum_constraint(
        self,
        solution: torch.Tensor,
        puzzle: torch.Tensor,
        target: Optional[torch.Tensor]
    ) -> bool:
        """Check if sum of solution equals target"""
        if target is None or len(target) == 0:
            return True
        
        expected_sum = target[0].item()
        actual_sum = solution.sum().item()
        tolerance = 0.1
        
        return abs(actual_sum - expected_sum) <= tolerance
    
    def _check_range_constraint(
        self,
        solution: torch.Tensor,
        puzzle: torch.Tensor,
        target: Optional[torch.Tensor]
    ) -> bool:
        """Check if all values are in valid range"""
        return (solution >= 0).all().item() and (solution <= 1).all().item()
    
    def _check_sparsity_constraint(
        self,
        solution: torch.Tensor,
        puzzle: torch.Tensor,
        target: Optional[torch.Tensor]
    ) -> bool:
        """Check if solution is not too dense"""
        # More than 80% of values should be below 0.5
        low_values = (solution < 0.5).sum().item()
        total_values = solution.numel()
        return (low_values / total_values) > 0.5
    
    def _check_smoothness_constraint(
        self,
        solution: torch.Tensor,
        puzzle: torch.Tensor,
        target: Optional[torch.Tensor]
    ) -> bool:
        """Check if adjacent values are smooth"""
        if solution.numel() < 2:
            return True
        
        # Check differences between adjacent elements
        diffs = torch.abs(solution[1:] - solution[:-1])
        max_diff = 0.5
        
        return (diffs <= max_diff).all().item()
    
    def verify(
        self,
        solution: torch.Tensor,
        puzzle: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ) -> Tuple[bool, List[str]]:
        """
        Verify if a solution satisfies all constraints.
        
        Args:
            solution: Proposed solution [batch_size, output_dim]
            puzzle: Original puzzle [batch_size, input_dim]
            target: Target constraints [batch_size, constraint_dim]
            
        Returns:
            valid: Whether all constraints are satisfied
            violations: List of violated constraint descriptions
        """
        # Handle batch processing
        batch_size = solution.size(0)
        all_valid = True
        all_violations = []
        
        for i in range(batch_size):
            item_valid, item_violations = self._verify_single(
                solution[i],
                puzzle[i],
                target[i] if target is not None else None
            )
            if not item_valid:
                all_valid = False
                all_violations.extend(item_violations)
        
        # Store violation history for analysis
        self.violation_history.append({
            'valid': all_valid,
            'violations': all_violations,
            'num_violations': len(all_violations)
        })
        
        return all_valid, all_violations
    
    def _verify_single(
        self,
        solution: torch.Tensor,
        puzzle: torch.Tensor,
        target: Optional[torch.Tensor]
    ) -> Tuple[bool, List[str]]:
        """Verify a single solution instance"""
        violations = []
        
        for rule in self.rules:
            try:
                if not rule.check_fn(solution, puzzle, target):
                    violations.append(f"{rule.name}: {rule.description}")
            except Exception as e:
                logger.warning(f"Error checking rule {rule.name}: {e}")
                violations.append(f"{rule.name}: Error during verification")
        
        return len(violations) == 0, violations
    
    def encode_violations(self, violations: List[str]) -> torch.Tensor:
        """
        Encode violations into a tensor for neural network feedback.
        
        Args:
            violations: List of violation descriptions
            
        Returns:
            encoded: Tensor encoding of violations [max_violations]
        """
        encoded = torch.zeros(self.max_violations)
        
        # Map violation descriptions to encoding indices
        for violation in violations:
            for rule in self.rules:
                if rule.name in violation:
                    idx = rule.violation_encoding
                    if idx < self.max_violations:
                        encoded[idx] = 1.0
                    break
        
        return encoded
    
    def explain_violations(self, violations: List[str]) -> str:
        """
        Generate human-readable explanation of violations.
        
        Args:
            violations: List of violation descriptions
            
        Returns:
            explanation: Formatted explanation text
        """
        if not violations:
            return "All constraints satisfied."
        
        explanation = f"Found {len(violations)} constraint violations:\n"
        for i, violation in enumerate(violations, 1):
            explanation += f"  {i}. {violation}\n"
        
        return explanation
    
    def get_statistics(self) -> Dict[str, float]:
        """Get verification statistics from history"""
        if not self.violation_history:
            return {}
        
        total_checks = len(self.violation_history)
        valid_checks = sum(1 for h in self.violation_history if h['valid'])
        total_violations = sum(h['num_violations'] for h in self.violation_history)
        
        return {
            'total_verifications': total_checks,
            'valid_rate': valid_checks / total_checks if total_checks > 0 else 0.0,
            'avg_violations': total_violations / total_checks if total_checks > 0 else 0.0
        }
    
    def add_rule(self, rule: Rule):
        """Add a new rule to the verifier"""
        self.rules.append(rule)
        logger.info(f"Added rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove a rule by name"""
        self.rules = [r for r in self.rules if r.name != rule_name]
        logger.info(f"Removed rule: {rule_name}")