"""
Training Script

Implements deep supervision training for the neural-symbolic reasoning system.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import yaml
import argparse
from pathlib import Path

from src.neural_component import NeuralReasoner
from src.symbolic_component import SymbolicVerifier
from src.refinement_engine import RefinementEngine
from src.metacognitive_controller import MetacognitiveController
from src.utils import setup_logging, save_checkpoint, save_metrics, get_device, set_seed

logger = logging.getLogger(__name__)


class DeepSupervisionTrainer:
    """
    Training system with deep supervision across refinement steps.
    
    Args:
        model: Neural reasoner to train
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        num_refinement_steps: Number of refinement steps to supervise
    """
    
    def __init__(
        self,
        model: NeuralReasoner,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        num_refinement_steps: int = 3
    ):
        self.model = model
        self.num_refinement_steps = num_refinement_steps
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training statistics
        self.training_history = []
        
        logger.info(
            f"Initialized DeepSupervisionTrainer: lr={learning_rate}, "
            f"weight_decay={weight_decay}, refinement_steps={num_refinement_steps}"
        )
    
    def train_step(
        self,
        puzzles: torch.Tensor,
        solutions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Single training step with deep supervision.
        
        Args:
            puzzles: Batch of puzzles [batch_size, input_dim]
            solutions: Ground truth solutions [batch_size, output_dim]
            targets: Constraint targets [batch_size, constraint_dim]
            
        Returns:
            Dictionary of losses at each step
        """
        self.model.train()
        device = self.model.get_device()
        
        # Move data to device
        puzzles = puzzles.to(device)
        solutions = solutions.to(device)
        targets = targets.to(device)
        
        total_loss = 0.0
        step_losses = []
        
        # Step 0: Initial hypothesis
        hypothesis, confidence = self.model(puzzles)
        
        # Loss at step 0
        loss_0 = self.criterion(hypothesis, solutions)
        total_loss += loss_0
        step_losses.append(loss_0.item())
        
        # Refinement steps with progressive weighting
        previous_hypothesis = hypothesis
        
        for step in range(1, self.num_refinement_steps + 1):
            # Generate simulated violations (in real use, would come from symbolic verifier)
            # For training, we use the error signal as a proxy for violations
            with torch.no_grad():
                error = torch.abs(previous_hypothesis - solutions)
                violations = torch.cat([
                    error,
                    torch.zeros(puzzles.size(0), 10 - error.size(1), device=device)
                ], dim=1) if error.size(1) < 10 else error[:, :10]
            
            # Refined hypothesis
            hypothesis, confidence = self.model(
                puzzles,
                previous_hypothesis.detach(),
                violations
            )
            
            # Loss at this step (weighted more heavily for later steps)
            # Later steps should be closer to ground truth
            weight = 1.0 + (step / self.num_refinement_steps)
            loss_step = weight * self.criterion(hypothesis, solutions)
            total_loss += loss_step
            step_losses.append(loss_step.item())
            
            previous_hypothesis = hypothesis
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'step_losses': step_losses,
            'avg_step_loss': sum(step_losses) / len(step_losses)
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (puzzles, solutions, targets) in enumerate(progress_bar):
            losses = self.train_step(puzzles, solutions, targets)
            
            total_loss += losses['total_loss']
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{losses['total_loss']:.4f}",
                'avg': f"{total_loss/num_batches:.4f}"
            })
        
        avg_loss = total_loss / num_batches
        
        return {
            'epoch': epoch,
            'train_loss': avg_loss,
            'num_batches': num_batches
        }


def create_synthetic_data(
    num_samples: int,
    input_dim: int,
    output_dim: int,
    train_split: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """
    Create synthetic puzzle data for training and validation.
    
    Args:
        num_samples: Total number of samples
        input_dim: Dimension of puzzle input
        output_dim: Dimension of solution
        train_split: Fraction of data for training
        
    Returns:
        train_loader, val_loader
    """
    # Generate synthetic puzzles and solutions
    puzzles = torch.randn(num_samples, input_dim)
    
    # Solutions are related to puzzles through a learnable function
    # Here we use a simple transformation
    solutions = torch.sigmoid(torch.randn(num_samples, output_dim))
    
    # Targets are derived from solutions (e.g., sum constraint)
    targets = solutions.sum(dim=1, keepdim=True)
    
    # Split into train and validation
    split_idx = int(num_samples * train_split)
    
    train_dataset = TensorDataset(
        puzzles[:split_idx],
        solutions[:split_idx],
        targets[:split_idx]
    )
    
    val_dataset = TensorDataset(
        puzzles[split_idx:],
        solutions[split_idx:],
        targets[split_idx:]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader


def train_model(config_path: str):
    """
    Main training function.
    
    Args:
        config_path: Path to configuration YAML file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    setup_logging(log_level=config.get('log_level', 'INFO'))
    set_seed(config.get('seed', 42))
    device = get_device(prefer_gpu=config.get('use_gpu', True))
    
    logger.info("="*60)
    logger.info("Starting Training")
    logger.info("="*60)
    logger.info(f"Configuration: {config}")
    
    # Create model
    model = NeuralReasoner(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        dropout=config['model'].get('dropout', 0.1)
    ).to(device)
    
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    # Create data loaders
    train_loader, val_loader = create_synthetic_data(
        num_samples=config['data']['num_samples'],
        input_dim=config['model']['input_dim'],
        output_dim=config['model']['output_dim'],
        train_split=config['data'].get('train_split', 0.8)
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create trainer
    trainer = DeepSupervisionTrainer(
        model=model,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0001),
        num_refinement_steps=config['training'].get('num_refinement_steps', 3)
    )
    
    # Training loop
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        # Train epoch
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        # Validation
        val_loss = validate(model, val_loader, trainer.criterion)
        
        logger.info(
            f"Epoch {epoch}/{config['training']['num_epochs']}: "
            f"train_loss={train_metrics['train_loss']:.4f}, "
            f"val_loss={val_loss:.4f}"
        )
        
        # Save metrics
        epoch_metrics = {
            **train_metrics,
            'val_loss': val_loss
        }
        training_history.append(epoch_metrics)
        
        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=trainer.optimizer,
                epoch=epoch,
                metrics=epoch_metrics,
                filename="best_model.pt"
            )
            logger.info(f"Saved best model: val_loss={val_loss:.4f}")
        
        # Save regular checkpoint every N epochs
        if epoch % config['training'].get('checkpoint_freq', 5) == 0:
            save_checkpoint(
                model=model,
                optimizer=trainer.optimizer,
                epoch=epoch,
                metrics=epoch_metrics
            )
    
    # Save final metrics
    save_metrics(training_history, "training_history.json")
    
    logger.info("="*60)
    logger.info("Training Complete")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("="*60)


def validate(
    model: NeuralReasoner,
    val_loader: DataLoader,
    criterion: nn.Module
) -> float:
    """
    Validate model on validation set.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for puzzles, solutions, targets in val_loader:
            device = model.get_device()
            puzzles = puzzles.to(device)
            solutions = solutions.to(device)
            
            # Forward pass
            hypothesis, _ = model(puzzles)
            loss = criterion(hypothesis, solutions)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeuroSymbolic-Recursive model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    train_model(args.config)