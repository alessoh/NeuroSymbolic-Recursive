"""
Tests for Symbolic Component
"""

import pytest
import torch

from src.symbolic_component import SymbolicVerifier, Rule


class TestSymbolicVerifier:
    """Test cases for SymbolicVerifier"""
    
    @pytest.fixture
    def verifier(self):
        """Create a test verifier"""
        return SymbolicVerifier()
    
    def test_initialization(self, verifier):
        """Test verifier initialization"""
        assert len(verifier.rules) > 0
        assert verifier.max_violations == 10
    
    def test_verify_valid_solution(self, verifier):
        """Test verification of valid solution"""
        # Create solution that satisfies constraints
        solution = torch.ones(1, 9) * 0.5
        puzzle = torch.randn(1, 16)
        target = torch.tensor([[4.5]])  # Sum equals 4.5
        
        valid, violations = verifier.verify(solution, puzzle, target)
        
        assert valid
        assert len(violations) == 0
    
    def test_verify_invalid_range(self, verifier):
        """Test detection of range violations"""
        # Create solution with out-of-range values
        solution = torch.ones(1, 9) * 1.5  # Exceeds max of 1.0
        puzzle = torch.randn(1, 16)
        target = torch.tensor([[4.5]])
        
        valid, violations = verifier.verify(solution, puzzle, target)
        
        assert not valid
        assert any("range_constraint" in v for v in violations)
    
    def test_verify_invalid_sum(self, verifier):
        """Test detection of sum constraint violations"""
        # Create solution with wrong sum
        solution = torch.ones(1, 9) * 0.1  # Sum is 0.9, not 4.5
        puzzle = torch.randn(1, 16)
        target = torch.tensor([[4.5]])
        
        valid, violations = verifier.verify(solution, puzzle, target)
        
        assert not valid
        assert any("sum_constraint" in v for v in violations)
    
    def test_encode_violations(self, verifier):
        """Test violation encoding"""
        violations = ["sum_constraint: test", "range_constraint: test"]
        
        encoded = verifier.encode_violations(violations)
        
        assert encoded.shape == (10,)
        assert encoded[0] == 1.0  # sum_constraint
        assert encoded[1] == 1.0  # range_constraint
    
    def test_explain_violations(self, verifier):
        """Test violation explanation"""
        violations = ["sum_constraint: sum mismatch", "range_constraint: out of range"]
        
        explanation = verifier.explain_violations(violations)
        
        assert "2 constraint violations" in explanation
        assert "sum_constraint" in explanation
        assert "range_constraint" in explanation
    
    def test_add_remove_rules(self, verifier):
        """Test adding and removing rules"""
        initial_count = len(verifier.rules)
        
        # Add a rule
        new_rule = Rule(
            name="test_rule",
            description="Test rule",
            check_fn=lambda s, p, t: True,
            violation_encoding=5
        )
        verifier.add_rule(new_rule)
        assert len(verifier.rules) == initial_count + 1
        
        # Remove the rule
        verifier.remove_rule("test_rule")
        assert len(verifier.rules) == initial_count
    
    def test_statistics(self, verifier):
        """Test statistics collection"""
        solution = torch.ones(1, 9) * 0.5
        puzzle = torch.randn(1, 16)
        target = torch.tensor([[4.5]])
        
        # Perform several verifications
        for _ in range(5):
            verifier.verify(solution, puzzle, target)
        
        stats = verifier.get_statistics()
        assert stats['total_verifications'] == 5
        assert 'valid_rate' in stats
        assert 'avg_violations' in stats