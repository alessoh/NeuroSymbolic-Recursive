import torch
from src.neural_component import NeuralReasoner
from src.symbolic_component import SymbolicVerifier
from src.refinement_engine import RefinementEngine

# Load model
model = NeuralReasoner(16, 64, 9)
checkpoint = torch.load('models/checkpoints/valid_model.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

verifier = SymbolicVerifier()
engine = RefinementEngine(model, verifier, max_iterations=5)

# Test ONE puzzle in detail
print('='*70)
print('DETAILED ANALYSIS OF ONE PUZZLE')
print('='*70)

puzzle = torch.randn(1, 16)
target = torch.tensor([[4.5]])

result = engine.reason(puzzle, target, verbose=False)

print()
print('Final Solution:')
solution = result.solution.flatten().detach().numpy()
print(f'  Values: {solution}')
print(f'  Sum: {solution.sum():.4f} (target: 4.5)')
print(f'  Min: {solution.min():.4f}')
print(f'  Max: {solution.max():.4f}')
print(f'  Range [0,1]: {(solution >= 0).all() and (solution <= 1).all()}')
print()

print('Checking each constraint:')
print('-'*70)

# Check sum
sum_ok = abs(solution.sum() - 4.5) < 0.1
print(f'Sum constraint: {sum_ok} (sum={solution.sum():.4f}, target=4.5)')

# Check range
range_ok = (solution >= 0).all() and (solution <= 1).all()
print(f'Range constraint: {range_ok} (all in [0,1])')

# Check sparsity
low_vals = (solution < 0.5).sum()
sparsity_ok = low_vals > len(solution) * 0.5
print(f'Sparsity constraint: {sparsity_ok} ({low_vals}/{len(solution)} values < 0.5)')

# Check smoothness
diffs = abs(solution[1:] - solution[:-1])
max_diff = diffs.max()
smooth_ok = max_diff < 0.5
print(f'Smoothness constraint: {smooth_ok} (max diff={max_diff:.4f})')

print()
print('Violations from last step:')
for v in result.refinement_history[-1].violations:
    print(f'  - {v}')
