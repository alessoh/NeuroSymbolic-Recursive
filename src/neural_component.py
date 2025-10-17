"""
Neural Component Module

Implements the neural reasoner that learns patterns and proposes solutions.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class NeuralReasoner(nn.Module):
    """
    Neural network component that learns patterns and proposes solutions.
    
    This component can process puzzles and generate solution hypotheses,
    incorporating feedback from previous refinement cycles to improve proposals.
    
    Args:
        input_dim: Dimension of input puzzle representation
        hidden_dim: Dimension of hidden layers
        output_dim: Dimension of solution space
        dropout: Dropout probability for regularization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 9,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input embedding layer
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Feedback embedding for refinement cycles
        # Takes previous hypothesis + violation encoding
        self.feedback_embed = nn.Sequential(
            nn.Linear(output_dim + 10, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout)
        )
        
        # Core reasoning network (applied recursively)
        self.reasoning_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Output heads
        self.solution_head = nn.Linear(hidden_dim, output_dim)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized NeuralReasoner: input_dim={input_dim}, "
                   f"hidden_dim={hidden_dim}, output_dim={output_dim}")
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        puzzle: torch.Tensor,
        previous_hypothesis: Optional[torch.Tensor] = None,
        violations: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the neural reasoner.
        
        Args:
            puzzle: Input puzzle state [batch_size, input_dim]
            previous_hypothesis: Previous solution attempt [batch_size, output_dim]
            violations: Encoded constraint violations [batch_size, 10]
            
        Returns:
            solution: Proposed solution [batch_size, output_dim]
            confidence: Confidence score [batch_size, 1]
        """
        batch_size = puzzle.size(0)
        
        # Embed the puzzle
        puzzle_features = self.input_embed(puzzle)
        
        # Process feedback if this is a refinement step
        if previous_hypothesis is not None:
            # Ensure violations tensor exists
            if violations is None:
                violations = torch.zeros(batch_size, 10, device=puzzle.device)
            
            # Combine previous hypothesis with violation feedback
            feedback_input = torch.cat([previous_hypothesis, violations], dim=-1)
            feedback_features = self.feedback_embed(feedback_input)
        else:
            # Initial hypothesis - no feedback
            feedback_features = torch.zeros(batch_size, self.hidden_dim // 2, device=puzzle.device)
        
        # Combine puzzle and feedback features
        combined_features = torch.cat([puzzle_features, feedback_features], dim=-1)
        
        # Apply reasoning network
        reasoning_output = self.reasoning_net(combined_features)
        
        # Generate solution proposal and confidence
        solution = self.solution_head(reasoning_output)
        confidence = self.confidence_head(reasoning_output)
        
        return solution, confidence
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_device(self) -> torch.device:
        """Get the device this model is on"""
        return next(self.parameters()).device