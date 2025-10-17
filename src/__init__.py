"""
NeuroSymbolic-Recursive Reasoning System

A research implementation combining neural-symbolic integration with recursive refinement.
"""

from src.neural_component import NeuralReasoner
from src.symbolic_component import SymbolicVerifier, Rule
from src.refinement_engine import RefinementEngine, RefinementState, ReasoningResult
from src.metacognitive_controller import MetacognitiveController, ReasoningStrategy
from src.utils import setup_logging, save_checkpoint, load_checkpoint

__version__ = "0.1.0"
__all__ = [
    "NeuralReasoner",
    "SymbolicVerifier",
    "Rule",
    "RefinementEngine",
    "RefinementState",
    "ReasoningResult",
    "MetacognitiveController",
    "ReasoningStrategy",
    "setup_logging",
    "save_checkpoint",
    "load_checkpoint",
]