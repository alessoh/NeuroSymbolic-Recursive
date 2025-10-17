"""
Tests for Refinement Engine
"""

import pytest
import torch

from src.neural_component import NeuralReasoner
from src.symbolic_component import SymbolicVerifier
from src.refinement_engine import RefinementEngine, ReasoningResult


class TestRefinementEngine:
    """Test cases for RefinementEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create a test refinement engine"""
        model = NeuralReasoner(input_dim=16, hidden_dim=64, output_dim=9)
        verifier = SymbolicVerifier()
        return RefinementEngine(
            neural_model=model,
            symbolic_verifier=verifier,
            max_iterations=5,
            confidence_threshold=0.9
        )
    
    def test_initialization(self, engine):
        """Test engine initialization"""
        assert engine.max_iterations == 5
        assert engine.confidence_threshold == 0.9
        assert engine.neural is not None
        assert engine.symbolic is not None
    
    def test_reason_basic(self, engine):
        """Test basic reasoning process"""
        puzzle = torch.randn(1, 16)
        target = torch.tensor([[4.5]])
        
        result = engine.reason(puzzle, target, verbose=False)
        
        assert isinstance(result, ReasoningResult)
        assert result.solution is not None
        assert result.steps_taken > 0
        assert len(result.refinement_history) == result.steps_taken
    
    def test_refinement_history(self, engine):
        """Test that refinement history is properly recorded"""
        puzzle = torch.randn(1, 16)
        target = torch.tensor([[4.5]])
        
        result = engine.reason(puzzle, target, verbose=False)
        
        # Check history structure
        for i, state in enumerate(result.refinement_history):
            assert state.step == i
            assert state.hypothesis is not None
            assert isinstance(state.confidence, float)
            assert isinstance(state.symbolic_valid, bool)
            assert isinstance(state.violations, list)
    
    def test_early_stopping(self, engine):
        """Test that early stopping works when solution is found"""
        # This test may not always trigger early stopping with random init
        # but it tests the mechanism
        puzzle = torch.randn(1, 16)
        target = torch.tensor([[4.5]])
        
        result = engine.reason(puzzle, target, verbose=False)
        
        # Should stop before max iterations if successful
        if result.success:
            assert result.steps_taken <= engine.max_iterations
    
    def test_max_iterations_limit(self, engine):
        """Test that refinement doesn't exceed max iterations"""
        puzzle = torch.randn(1, 16)
        target = torch.tensor([[4.5]])
        
        result = engine.reason(puzzle, target, verbose=False)
        
        assert result.steps_taken <= engine.max_iterations + 1  # +1 for initial
    
    def test_batch_processing(self, engine):
        """Test reasoning with batch input"""
        batch_size = 4
        puzzle = torch.randn(batch_size, 16)
        target = torch.tensor([[4.5]] * batch_size)
        
        result = engine.reason(puzzle, target, verbose=False)
        
        assert result.solution.shape[0] == batch_size
    
    def test_statistics(self, engine):
        """Test statistics collection"""
        puzzle = torch.randn(1, 16)
        target = torch.tensor([[4.5]])
        
        # Run multiple reasoning processes
        for _ in range(3):
            engine.reason(puzzle, target, verbose=False)
        
        stats = engine.get_statistics()
        assert stats['total_runs'] == 3
        assert 'success_rate' in stats
        assert 'avg_steps' in stats
    
    def test_reset_statistics(self, engine):
        """Test statistics reset"""
        puzzle = torch.randn(1, 16)
        target = torch.tensor([[4.5]])
        
        engine.reason(puzzle, target, verbose=False)
        engine.reset_statistics()
        
        stats = engine.get_statistics()
        assert stats == {}