"""
Tests for Neural Component
"""

import pytest
import torch

from src.neural_component import NeuralReasoner


class TestNeuralReasoner:
    """Test cases for NeuralReasoner"""
    
    @pytest.fixture
    def model(self):
        """Create a test model"""
        return NeuralReasoner(input_dim=16, hidden_dim=64, output_dim=9)
    
    def test_initialization(self, model):
        """Test model initialization"""
        assert model.input_dim == 16
        assert model.hidden_dim == 64
        assert model.output_dim == 9
        assert model.count_parameters() > 0
    
    def test_forward_pass_initial(self, model):
        """Test forward pass without feedback"""
        batch_size = 4
        puzzle = torch.randn(batch_size, 16)
        
        solution, confidence = model(puzzle)
        
        assert solution.shape == (batch_size, 9)
        assert confidence.shape == (batch_size, 1)
        assert (confidence >= 0).all() and (confidence <= 1).all()
    
    def test_forward_pass_with_feedback(self, model):
        """Test forward pass with refinement feedback"""
        batch_size = 4
        puzzle = torch.randn(batch_size, 16)
        previous_hypothesis = torch.randn(batch_size, 9)
        violations = torch.randn(batch_size, 10)
        
        solution, confidence = model(puzzle, previous_hypothesis, violations)
        
        assert solution.shape == (batch_size, 9)
        assert confidence.shape == (batch_size, 1)
    
    def test_gradient_flow(self, model):
        """Test that gradients flow properly"""
        puzzle = torch.randn(1, 16)
        target = torch.randn(1, 9)
        
        solution, _ = model(puzzle)
        loss = torch.nn.functional.mse_loss(solution, target)
        loss.backward()
        
        # Check that some parameters have gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                      for p in model.parameters())
        assert has_grad
    
    def test_device_handling(self, model):
        """Test device handling"""
        device = model.get_device()
        assert isinstance(device, torch.device)
    
    def test_parameter_count(self, model):
        """Test parameter counting"""
        param_count = model.count_parameters()
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count == manual_count