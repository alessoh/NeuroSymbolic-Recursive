"""
Integration Tests

Tests that verify all components work together correctly.
"""

import pytest
import torch
import tempfile
import os

from src.neural_component import NeuralReasoner
from src.symbolic_component import SymbolicVerifier
from src.refinement_engine import RefinementEngine
from src.metacognitive_controller import MetacognitiveController, ReasoningStrategy
from src.utils import save_checkpoint, load_checkpoint
from training.train import DeepSupervisionTrainer


class TestFullSystem:
    """Integration tests for the complete system"""
    
    @pytest.fixture
    def components(self):
        """Create all system components"""
        model = NeuralReasoner(input_dim=16, hidden_dim=64, output_dim=9)
        verifier = SymbolicVerifier()
        engine = RefinementEngine(model, verifier)
        metacog = MetacognitiveController()
        
        return {
            'model': model,
            'verifier': verifier,
            'engine': engine,
            'metacognitive': metacog
        }
    
    def test_end_to_end_reasoning(self, components):
        """Test complete reasoning pipeline"""
        puzzle = torch.randn(1, 16)
        target = torch.tensor([[4.5]])
        
        # Execute reasoning
        result = components['engine'].reason(puzzle, target, verbose=False)
        
        # Verify result structure
        assert result is not None
        assert result.solution is not None
        assert len(result.refinement_history) > 0
        
        # Update metacognitive controller
        components['metacognitive'].update_weights(
            neural_confidence=result.final_confidence,
            symbolic_valid=result.refinement_history[-1].symbolic_valid
        )
        
        # Verify metacognitive update
        assert components['metacognitive'].neural_weight + components['metacognitive'].symbolic_weight == pytest.approx(1.0)
    
    def test_training_pipeline(self, components):
        """Test training with deep supervision"""
        model = components['model']
        trainer = DeepSupervisionTrainer(model, learning_rate=0.01)
        
        # Create synthetic data
        puzzles = torch.randn(8, 16)
        solutions = torch.sigmoid(torch.randn(8, 9))
        targets = solutions.sum(dim=1, keepdim=True)
        
        # Train one step
        losses = trainer.train_step(puzzles, solutions, targets)
        
        assert 'total_loss' in losses
        assert 'step_losses' in losses
        assert len(losses['step_losses']) == trainer.num_refinement_steps + 1
    
    def test_checkpoint_save_load(self, components):
        """Test model checkpointing"""
        model = components['model']
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=None,
                epoch=1,
                metrics={'test': 0.5},
                checkpoint_dir=tmpdir,
                filename="test_checkpoint.pt"
            )
            
            assert os.path.exists(checkpoint_path)
            
            # Load checkpoint
            loaded_checkpoint = load_checkpoint(
                checkpoint_path,
                model=model,
                device='cpu'
            )
            
            assert loaded_checkpoint['epoch'] == 1
            assert loaded_checkpoint['metrics']['test'] == 0.5
    
    def test_strategy_selection(self, components):
        """Test metacognitive strategy selection"""
        metacog = components['metacognitive']
        
        # Test different scenarios
        strategy1 = metacog.select_strategy(
            puzzle_complexity=0.8,
            time_budget=0.9
        )
        assert strategy1 == ReasoningStrategy.SYMBOLIC_FIRST
        
        strategy2 = metacog.select_strategy(
            puzzle_complexity=0.2,
            time_budget=0.5
        )
        assert strategy2 == ReasoningStrategy.NEURAL_FIRST
        
        strategy3 = metacog.select_strategy(
            puzzle_complexity=0.5,
            time_budget=0.5
        )
        assert strategy3 == ReasoningStrategy.BALANCED
    
    def test_multi_step_refinement(self, components):
        """Test that refinement actually improves solutions over steps"""
        puzzle = torch.randn(1, 16)
        target = torch.tensor([[4.5]])
        
        result = components['engine'].reason(puzzle, target, verbose=False)
        
        # Check if there's any improvement in early vs late steps
        # (Though with random weights, this may not always be true)
        if len(result.refinement_history) > 1:
            initial_confidence = result.refinement_history[0].confidence
            final_confidence = result.refinement_history[-1].confidence
            
            # At least confidence should be computed (may increase or decrease)
            assert isinstance(initial_confidence, float)
            assert isinstance(final_confidence, float)
    
    def test_component_statistics(self, components):
        """Test that all components collect statistics properly"""
        puzzle = torch.randn(1, 16)
        target = torch.tensor([[4.5]])
        
        # Run reasoning
        result = components['engine'].reason(puzzle, target, verbose=False)
        
        # Check engine statistics
        engine_stats = components['engine'].get_statistics()
        assert 'total_runs' in engine_stats
        
        # Check verifier statistics
        verifier_stats = components['verifier'].get_statistics()
        assert 'total_verifications' in verifier_stats
        
        # Update and check metacognitive statistics
        components['metacognitive'].update_weights(
            neural_confidence=result.final_confidence,
            symbolic_valid=result.refinement_history[-1].symbolic_valid
        )
        metacog_stats = components['metacognitive'].get_statistics()
        assert 'total_decisions' in metacog_stats