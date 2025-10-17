"""
Utilities Module

Shared utility functions for logging, checkpointing, and data handling.
"""

import torch
import logging
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file name
        log_dir: Directory for log files
        
    Returns:
        Configured logger
    """
    # Create log directory if it doesn't exist
    if log_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)
    else:
        log_path = None
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Set up handlers
    handlers = [logging.StreamHandler()]
    if log_path:
        handlers.append(logging.FileHandler(log_path))
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={log_level}, file={log_path}")
    
    return logger


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: str = "models/checkpoints",
    filename: Optional[str] = None
) -> str:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        metrics: Dictionary of metrics to save
        checkpoint_dir: Directory for checkpoints
        filename: Optional checkpoint filename
        
    Returns:
        Path to saved checkpoint
    """
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_epoch{epoch}_{timestamp}.pt"
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load model onto
        
    Returns:
        Dictionary containing checkpoint metadata
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"  Metrics: {checkpoint.get('metrics', {})}")
    
    return checkpoint


def save_metrics(
    metrics: Dict[str, Any],
    filename: str,
    output_dir: str = "results"
):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        filename: Output filename
        output_dir: Output directory
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Saved metrics: {filepath}")


def load_metrics(filename: str, input_dir: str = "results") -> Dict[str, Any]:
    """
    Load metrics from JSON file.
    
    Args:
        filename: Input filename
        input_dir: Input directory
        
    Returns:
        Dictionary of loaded metrics
    """
    filepath = os.path.join(input_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metrics file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loaded metrics: {filepath}")
    
    return metrics


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Set random seed: {seed}")


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get appropriate device for computation.
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
        
    Returns:
        torch.device
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device}")
    
    return device