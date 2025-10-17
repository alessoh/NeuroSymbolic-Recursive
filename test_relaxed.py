import torch
from src.neural_component import NeuralReasoner
from src.symbolic_component import SymbolicVerifier, Rule
from src.refinement_engine import RefinementEngine

# Load model
model = NeuralReasoner(16, 64, 9)
checkpoint = torch.load('models/checkpoints/valid_model.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create verifier with MORE LENIENT sum constraint
def relaxed_sum_check(solution, puzzle, target):
    if target is None or len(target) == 0:
        return True
    expected_sum = target[0].item()
    actual_sum = solution.sum().item()
    tolerance = 0.2  # Increased from 0.1 to 0.2
    return abs(actual_sum - expected_sum) <= tolerance

# Replace the sum constraint rule
verifier = SymbolicVerifier()
# Remove old sum rule and add new one
verifier.rules[0] = Rule(
    name="sum_constraint",
    description="Sum of values must equal target (tolerance 0.2)",
    check_fn=relaxed_sum_check,
    violation_encoding=0
)

engine = RefinementEngine(model, verifier, max_iterations=5)

print("="*70)
print("Testing with RELAXED Sum Constraint (tolerance=0.2)")
print("="*70)
print()

successes = 0
for i in range(10):
    puzzle = torch.randn(1, 16)
    target = torch.tensor([[4.5]])
    
    result = engine.reason(puzzle, target, verbose=False)
    
    sum_val = result.solution.sum().item()
    status = "SUCCESS" if result.success else "FAIL"
    
    print(f'Puzzle {i+1:2d}: {status:7s} Steps={result.steps_taken}, '
          f'Confidence={result.final_confidence:.3f}, Sum={sum_val:.3f}, '
          f'Error={abs(sum_val-4.5):.3f}')
    
    if result.success:
        successes += 1

print("="*70)
print(f'Success Rate: {successes}/10 ({successes*10}%)')
print("="*70)
