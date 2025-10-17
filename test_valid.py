import torch
from src.neural_component import NeuralReasoner
from src.symbolic_component import SymbolicVerifier
from src.refinement_engine import RefinementEngine
from src.utils import setup_logging

setup_logging(log_level='INFO')

print('='*70)
print('Testing Model Trained on VALID Data')
print('='*70)
print()

# Create model
model = NeuralReasoner(16, 64, 9)

# Load the NEW valid model
checkpoint = torch.load('models/checkpoints/valid_model.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded valid_model from epoch {checkpoint['epoch']}")
print(f"Final training loss: {checkpoint['metrics']['train_loss']:.4f}")
print()

# Test on multiple puzzles
verifier = SymbolicVerifier()
engine = RefinementEngine(model, verifier, max_iterations=5)

print('Testing on 10 puzzles:')
print('-'*70)

successes = 0
total_steps = 0

for i in range(10):
    puzzle = torch.randn(1, 16)
    target = torch.tensor([[4.5]])
    
    result = engine.reason(puzzle, target, verbose=False)
    
    sum_val = result.solution.sum().item()
    status = 'SUCCESS' if result.success else 'FAIL'
    
    print(f'Puzzle {i+1:2d}: {status:7s} Steps={result.steps_taken}, Confidence={result.final_confidence:.3f}, Sum={sum_val:.3f}')
    
    if result.success:
        successes += 1
    total_steps += result.steps_taken

print('-'*70)
print(f'Success Rate: {successes}/10 ({successes*10}%)')
print(f'Average Steps: {total_steps/10:.1f}')