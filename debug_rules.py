import torch
from src.neural_component import NeuralReasoner
from src.symbolic_component import SymbolicVerifier, Rule
from src.refinement_engine import RefinementEngine

# Load model
model = NeuralReasoner(16, 64, 9)
checkpoint = torch.load('models/checkpoints/valid_model.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Relax sum constraint
def relaxed_sum_check(solution, puzzle, target):
    if target is None or len(target) == 0:
        return True
    expected_sum = target[0].item()
    actual_sum = solution.sum().item()
    tolerance = 0.2
    return abs(actual_sum - expected_sum) <= tolerance

verifier = SymbolicVerifier()
verifier.rules[0] = Rule(
    name="sum_constraint",
    description="Sum of values must equal target (tolerance 0.2)",
    check_fn=relaxed_sum_check,
    violation_encoding=0
)

engine = RefinementEngine(model, verifier, max_iterations=5)

# Test ONE puzzle and show ALL constraint results
puzzle = torch.randn(1, 16)
target = torch.tensor([[4.5]])

result = engine.reason(puzzle, target, verbose=False)
solution = result.solution.flatten()

print("="*70)
print("DETAILED CONSTRAINT ANALYSIS")
print("="*70)
print(f"Solution: {solution.numpy()}")
print(f"Sum: {solution.sum().item():.4f}")
print()

print("Testing each rule:")
print("-"*70)
for i, rule in enumerate(verifier.rules):
    passed = rule.check_fn(solution, puzzle[0], target[0] if target is not None else None)
    status = "PASS" if passed else "FAIL"
    print(f"{i}. {rule.name}: {status}")
    print(f"   {rule.description}")
    if not passed:
        print(f"   *** THIS RULE IS FAILING ***")
    print()

print("Final verification result:")
valid, violations = verifier.verify(solution.unsqueeze(0), puzzle, target)
print(f"Valid: {valid}")
print(f"Violations: {violations}")
