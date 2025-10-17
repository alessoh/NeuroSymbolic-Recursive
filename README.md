# NeuroSymbolic-Recursive Reasoning System

A research implementation combining neural-symbolic integration with recursive refinement for advanced reasoning tasks. This project represents an iterative approach to building AI systems that can both learn from data and reason with explicit logical constraints.

## Overview

Modern AI systems excel at pattern recognition but often struggle with logical reasoning and interpretability. Conversely, symbolic AI systems provide transparent reasoning but lack the flexibility to learn from data. This project integrates both approaches through a recursive refinement architecture where neural and symbolic components work together iteratively to solve reasoning problems.

The system architecture consists of four primary components: a neural reasoner that learns patterns and proposes solutions, a symbolic verifier that checks solutions against logical constraints, a refinement engine that coordinates recursive improvement cycles, and a metacognitive controller that makes strategic decisions about the reasoning process. Unlike traditional neural networks that generate outputs in a single forward pass, this system iteratively refines its hypotheses through multiple cycles of proposal, verification, and improvement.

The key innovation is deep supervision across refinement steps. Rather than providing training signal only on final outputs, the system receives feedback at each intermediate refinement stage. This encourages the neural component to learn trajectories of progressive improvement rather than simply memorizing input-output mappings. The symbolic component provides hard constraints that prevent logically invalid solutions while generating specific feedback to guide refinement.

## Key Features

The system implements several important capabilities that distinguish it from conventional approaches. Neural-symbolic integration allows pattern recognition and logical reasoning to enhance each other bidirectionally. The neural component learns to internalize common logical patterns while the symbolic component provides verification and explanation. Recursive refinement with deep supervision enables progressive solution improvement through multiple reasoning cycles, with training signals at each stage encouraging meaningful intermediate steps.

The architecture provides interpretability through explicit reasoning traces that show how solutions evolve through refinement cycles, which constraints were checked at each stage, and why certain proposals were accepted or rejected. The metacognitive controller adapts the reasoning strategy based on problem characteristics and component confidence, learning over time when to rely more heavily on neural intuition versus symbolic verification.

The implementation runs efficiently on standard hardware without requiring GPUs for small to medium scale problems, though GPU acceleration is supported for larger models and datasets. The modular design with clear interfaces between components facilitates testing, modification, and extension of individual subsystems.

## Installation

The system requires Python 3.8 or higher. We recommend using a virtual environment to manage dependencies and avoid conflicts with other projects.

### Prerequisites

Ensure you have Python 3.8 or later installed on your system. You can verify your Python version by running `python --version` or `python3 --version` in your terminal. You will also need pip for package management, which typically comes bundled with Python installations.

### Basic Installation

Clone the repository to your local machine and navigate into the project directory. Create a virtual environment to isolate the project dependencies from your system Python installation. On Linux and macOS, you can create a virtual environment using `python3 -m venv venv` and activate it with `source venv/bin/activate`. On Windows, use `python -m venv venv` to create the environment and `venv\Scripts\activate` to activate it.

Once your virtual environment is activated, install the required packages using pip. The basic installation includes PyTorch for neural network operations, NumPy for numerical computations, and several utility libraries. Install dependencies with `pip install -r requirements.txt`.

### Requirements File Contents

The requirements.txt file should contain the following dependencies:

```
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
pyyaml>=6.0
pytest>=7.0.0
pytest-cov>=4.0.0
```

For CPU-only installations, PyTorch will automatically install the appropriate version. If you have a CUDA-capable GPU and want to leverage hardware acceleration, install PyTorch with CUDA support by visiting pytorch.org and using the installation command appropriate for your system configuration.

### Development Installation

For contributors and developers who want to modify the codebase, install the package in editable mode. This allows changes to the source code to be immediately reflected without reinstalling. Use `pip install -e .` from the project root directory after creating a setup.py file.

### Verification

After installation, verify that everything is working correctly by running the test suite with `pytest tests/` from the project root directory. All tests should pass, confirming that the environment is properly configured and all components are functioning correctly. You can also run the simple example with `python examples/simple_example.py` to see the system in action.

## Quick Start

To quickly understand how the system works, start with the provided example script. This demonstrates the complete reasoning process on a simple puzzle problem, showing how the neural and symbolic components interact through recursive refinement cycles.

Run the basic example with `python examples/simple_example.py`. This will initialize the system components, load a sample puzzle, execute the recursive reasoning process with detailed logging, and display the step-by-step refinement showing solution evolution. The output will show confidence scores, symbolic verification results, and the final solution along with the number of refinement steps required.

For training your own models, use the training script with `python training/train.py --config configs/default_config.yaml`. This will load training data, initialize the model architecture, run multiple epochs with deep supervision, evaluate on validation data, and save checkpoints of the trained model. Training progress is logged to both the console and a log file for later analysis.

To evaluate a trained model on test data, run `python training/evaluate.py --checkpoint models/checkpoints/best_model.pt`. This loads the saved model, runs inference on test puzzles, computes accuracy and reasoning quality metrics, and generates visualizations of the reasoning process.

## Architecture Details

The neural component is implemented as a feedforward network with specialized structure for recursive reasoning. The input embedding layer encodes puzzle states into continuous representations suitable for neural processing. A feedback embedding layer processes previous hypotheses and violation signals to guide refinement. The core reasoning network applies transformations that can be recursively applied to progressively improve solutions. Output heads generate solution proposals and confidence estimates.

The symbolic component maintains a knowledge base of logical rules and constraints specific to the problem domain. The verification engine checks proposed solutions against all applicable constraints and generates detailed feedback about any violations. The explanation generator produces human-readable descriptions of why solutions are valid or invalid. The rule representation uses a flexible format that can encode various constraint types including equality, inequality, uniqueness, and structural constraints.

The refinement engine implements the core recursive reasoning loop that alternates between neural proposal and symbolic verification. It initializes reasoning with a neural hypothesis, then enters a refinement cycle where it verifies the current hypothesis symbolically, encodes any violations as neural feedback, generates an improved hypothesis through the neural network, and repeats until convergence or maximum iterations. The stopping criteria include symbolic validity with high confidence, reaching maximum iterations, or detecting convergence when successive hypotheses are nearly identical.

The metacognitive controller operates at a higher level than the refinement loop, making strategic decisions about the reasoning process. It selects reasoning strategies based on estimated problem complexity and available computational budget. It dynamically adjusts weights between neural and symbolic components based on their recent performance. It learns over time which strategies work best for different problem types. It monitors reasoning quality and can trigger strategy changes mid-process if needed.

## Training Methodology

Training employs deep supervision across all refinement steps rather than only supervising final outputs. For each training sample, the system executes the complete refinement process for a fixed number of steps. Loss is computed at each refinement step by comparing the intermediate hypothesis against the ground truth solution. Later refinement steps receive higher loss weights, reflecting that they should produce solutions closer to the target. Gradients from all supervised steps are aggregated and used to update the neural network parameters.

The symbolic verification results also influence training through auxiliary loss terms. When violations are detected, the system adds penalties that encourage the neural component to avoid similar errors in future iterations. This helps the neural network internalize logical constraints implicitly while maintaining the symbolic component as an explicit safety mechanism.

The metacognitive controller learns through a different mechanism based on performance statistics. It tracks which strategies succeed or fail for different problem types and adjusts its decision-making accordingly. This learning is separate from the gradient-based neural network training but occurs in parallel, gradually improving the system's ability to select appropriate reasoning approaches.

Training hyperparameters can be configured through YAML configuration files that specify model architecture dimensions, learning rates and schedules, batch sizes and gradient accumulation steps, number of refinement steps for deep supervision, loss weights for different supervision levels, and various other training parameters. The default configuration provides reasonable starting values that can be adjusted based on specific problem requirements.

## Usage Examples

The examples directory contains several demonstrations of system capabilities. The simple_example.py script shows basic usage with synthetic puzzles, illustrating the core reasoning loop and refinement process. The advanced_example.py script demonstrates more sophisticated features including custom symbolic rules, hierarchical problem decomposition, and interpretability analysis.

For custom problem domains, you can define your own symbolic rules by subclassing the SymbolicVerifier class and implementing domain-specific constraint checking. The neural architecture can be modified by adjusting the NeuralReasoner class to incorporate domain knowledge or use different network structures. The refinement process can be customized by configuring stopping criteria, maximum iterations, and confidence thresholds.

## Testing

The test suite provides comprehensive coverage of system components and their integration. Unit tests verify individual components in isolation, checking neural network forward and backward passes, symbolic verification logic, refinement engine step execution, and metacognitive controller decisions. Integration tests ensure components work correctly together, validating end-to-end reasoning processes, training with deep supervision, and evaluation metrics computation.

Run all tests with `pytest tests/` or specific test files with `pytest tests/test_neural.py`. Generate a coverage report with `pytest --cov=src --cov-report=html tests/` to identify untested code paths. The coverage report is generated in the htmlcov directory and can be viewed in a web browser.

## Project Structure

The repository is organized as follows. The src directory contains the core implementation with separate modules for each architectural component. The training directory includes scripts for model training, evaluation, and visualization. The tests directory provides comprehensive test coverage for all components. The examples directory contains demonstration scripts and sample puzzle data. The models directory stores trained model checkpoints. The configs directory holds YAML configuration files. The docs directory contains additional documentation and technical notes.

Each source module focuses on a specific responsibility. The neural_component.py module implements the neural reasoner network. The symbolic_component.py module provides symbolic verification and rule management. The refinement_engine.py module coordinates the recursive reasoning loop. The metacognitive_controller.py module handles strategic decision-making. The utils.py module contains shared utilities for data loading, logging, and visualization.

## Performance Considerations

The system is designed to run efficiently on modest hardware while maintaining research flexibility. For small problems with input dimensions under 100 and output dimensions under 20, a modern CPU is sufficient. Medium problems benefit from GPU acceleration, particularly during training with large batch sizes. Large-scale problems with complex symbolic rule sets may require distributed training or specialized hardware.

Memory usage scales primarily with batch size and refinement depth. Each refinement step stores intermediate activations for gradient computation during training. The symbolic verifier maintains rule databases that grow with problem complexity. Checkpointing and gradient accumulation can reduce memory requirements at the cost of training speed.

Inference time depends on the number of refinement cycles required. Simple problems often converge in 2-3 cycles while complex problems may use the maximum allowed iterations. The adaptive computation mechanism learns to allocate more cycles to difficult problems, balancing accuracy and efficiency.

## Extending the System

The modular architecture facilitates several natural extensions. Hierarchical reasoning can be added by stacking multiple neural-symbolic pairs at different abstraction levels, with lower levels handling local constraints and higher levels addressing global structure. Multi-agent capabilities can be incorporated by having different agents specialize in different reasoning types, coordinated through the metacognitive controller.

Enhanced symbolic reasoning can include probabilistic logic for uncertainty handling, temporal reasoning for dynamic problems, and causal reasoning for understanding action consequences. Advanced neural architectures such as graph neural networks, attention mechanisms, or meta-learning can improve pattern recognition while maintaining integration with symbolic components.

The refinement engine can be extended with more sophisticated stopping criteria, adaptive iteration budgets based on problem complexity, or hierarchical refinement where different aspects of the solution are refined at different rates. The metacognitive controller can incorporate reinforcement learning for strategy selection or Bayesian optimization for hyperparameter tuning.

## References

This implementation draws on several foundational works in neural-symbolic integration and recursive reasoning. The neural-symbolic approach is informed by Zhu et al. (2023) "Neural-Symbolic AI: A Survey" available at https://arxiv.org/pdf/2502.12904, which provides comprehensive analysis of integration frameworks. The hierarchical reasoning concepts build on Wang et al. (2025) "Hierarchical Reasoning Model" at https://arxiv.org/abs/2506.21734, demonstrating multi-level abstraction in reasoning systems.

The recursive refinement methodology is inspired by Jolicoeur-Martineau (2025) "Less is More: Recursive Reasoning with Tiny Networks" at https://arxiv.org/abs/2510.04871, which shows that iterative application of small networks can outperform larger single-pass models. The deep supervision training approach relates to work by Wei et al. (2022) "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" presented at NeurIPS 2022, emphasizing the value of intermediate reasoning steps.

Additional relevant research includes Snell et al. (2024) "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters" at https://arxiv.org/abs/2408.03314, which demonstrates the effectiveness of investing computation during inference. The metacognitive control concepts draw on broader cognitive science research on self-monitoring and strategy selection in human reasoning.

For neural-symbolic systems in practice, see the NEXUS architecture at https://github.com/alessoh/Neural-Symbolic-Superintelligence and related implementations at https://github.com/alessoh/HRMlaptop and https://github.com/alessoh/TRMlaptop. These projects demonstrate various approaches to combining learned and logical reasoning in working systems.

## Contributing

Contributions are welcome from researchers and developers interested in advancing reasoning systems. Before contributing, please review the existing codebase and test suite to understand the architecture and coding conventions. Open an issue to discuss significant changes or new features before implementing them. This ensures alignment with project goals and prevents duplicated effort.

When submitting pull requests, include tests for new functionality to maintain code coverage, add docstrings and comments explaining complex logic, update documentation to reflect changes, and ensure all existing tests pass with your modifications. Follow the existing code style and structure to maintain consistency across the project.

Areas particularly valuable for contribution include implementing new symbolic reasoning capabilities for different constraint types, developing visualization tools for reasoning traces and component interactions, extending the neural architecture with attention or graph networks, adding benchmark problems and evaluation metrics, and improving training efficiency and convergence.

## License

This project is released under the MIT License, allowing free use, modification, and distribution with minimal restrictions. See the LICENSE file in the repository root for complete license text. When using this code in research publications, please cite the relevant foundational papers listed in the References section along with this implementation.

## Acknowledgments

This implementation synthesizes ideas from multiple research communities including neural-symbolic AI, recursive reasoning, and metacognitive systems. We acknowledge the foundational work of researchers whose papers informed this architecture. We thank the open source community for tools like PyTorch that make this research possible. We appreciate early users who provided feedback on the system design and implementation.

## Contact and Support

For questions, bug reports, or feature requests, please open an issue in the GitHub repository. For broader discussions about neural-symbolic reasoning or collaboration opportunities, contact the maintainers through the repository. For academic inquiries regarding research applications, please reference the relevant papers in the References section.

Documentation is maintained in the docs directory with additional technical details about component interactions, training procedures, and extension guidelines. The examples directory contains working code demonstrating various system capabilities. The test suite serves as additional documentation showing expected behavior and usage patterns.

## Citation

If you use this implementation in your research, please cite both this repository and the relevant foundational papers:

```bibtex
@software{neurosymbolic_recursive_2025,
  title={NeuroSymbolic-Recursive Reasoning System},
  author={Research Implementation},
  year={2025},
  url={https://github.com/yourusername/neurosymbolic-recursive}
}

@article{zhu2023neural_symbolic,
  title={Neural-Symbolic AI: A Survey},
  author={Zhu, et al.},
  journal={arXiv preprint arXiv:2502.12904},
  year={2023}
}

@article{jolicoeur2025less,
  title={Less is More: Recursive Reasoning with Tiny Networks},
  author={Jolicoeur-Martineau, Alexia},
  journal={arXiv preprint arXiv:2510.04871},
  year={2025}
}

@article{wang2025hierarchical,
  title={Hierarchical Reasoning Model},
  author={Wang, Guan and others},
  journal={arXiv preprint arXiv:2506.21734},
  year={2025}
}
```
