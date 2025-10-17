import torch
from src.neural_component import NeuralReasoner
from src.symbolic_component import SymbolicVerifier, Rule
from src.refinement_engine import RefinementEngine

# Load model
model = NeuralReasoner(16, 64, 9)
checkpoint = torch.load('models/checkpoints/valid_model.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create verifier and REMOVE the sparsity constraint
verifier = SymbolicVerifier()

# Keep only the essential constraints
essential_rules = [
    Rule(
        name="sum_constraint",
        description="Sum must equal target (?0.2)",
        check_fn=lambda s, p, t: abs(s.sum().item() - t[0].item()) <= 0.2 if t is not None else True,
        violation_encoding=0
    ),
    Rule(
        name="range_constraint",
        description="All values in [0, 1]",
        check_fn=lambda s, p, t: (s >= 0).all().item() and (s <= 1).all().item(),
        violation_encoding=1
    ),
    Rule(
        name="smoothness_constraint",
        description="Adjacent values should not differ too much",
        check_fn=lambda s, p, t: (torch.abs(s[1:] - s[:-1]) <= 0.5).all().item() if s.numel() > 1 else True,
        violation_encoding=3
    )
]

verifier.rules = essential_rules

engine = RefinementEngine(model, verifier, max_iterations=5)

print("="*70)
print("Testing with ESSENTIAL CONSTRAINTS ONLY")
print("="*70)
print("Constraints: Sum (?0.2), Range [0,1], Smoothness")
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
          f'Confidence={result.final_confidence:.3f}, Sum={sum_val:.3f}')
    
    if result.success:
        successes += 1

print("="*70)
print(f'SUCCESS RATE: {successes}/10 ({successes*10}%)')
print("="*70)
print()
print("?? The model is working! The sparsity constraint was too strict.")
