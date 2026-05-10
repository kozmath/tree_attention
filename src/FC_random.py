import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import numpy as np
import random
import math




def function_composition(seq_len, batch_size=32, n=10, x=None, k=2, seed=None, device='mps'):
    """
    Fully vectorized function composition for batch generation.
    Creates token sequences for vocabulary-based transformer training.

    Args:
        seq_len: Sequence length
        batch_size: Number of sequences to generate
        n: Range for random indices (0 to n-1)
        x: Starting value (if None, randomly generated for each batch item)
        k: Number of composition steps
        seed: Random seed for reproducibility
        device: Device to place tensors on

    Returns:
        batch_inputs: [batch_size, seq_len] - token sequences
        batch_targets: [batch_size] - target values
    """
    if seq_len < k*n + 1:
        raise ValueError(f"seq_len must be at least {k*n + 1} for k={k} and n={n}")
    # if d_model < 3:
    #     raise ValueError("d_model must be at least 3 to accommodate input format")

    # Set default device to MPS if available, fallback to CPU
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    # Initialize batch tensors - 2D for token sequences [batch_size, seq_len]
    batch_inputs = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    # Generate starting values for each batch item
    if x is None:
        x_values = torch.randint(0, n, (batch_size,), device=device)
    else:
        x_values = torch.full((batch_size,), x, device=device)

    # Generate all function mappings for all batch items at once
    # Shape: [batch_size, k, n] - for each batch item, k functions each mapping n values
    all_functions = torch.randint(0, n, (batch_size, k, n), device=device)

    # Compute function composition results for all batch items
    batch_targets = x_values.clone()
    batch_indices = torch.arange(batch_size, device=device)
    for step in range(k):
        # For each batch item, apply the step-th function
        indices = batch_targets.long()
        batch_targets = all_functions[batch_indices, step, indices]

    # Fill the input sequences with function values only (no positional encoding needed)
    total_positions = k * n
    positions_to_fill = min(total_positions, seq_len)

    if positions_to_fill > 0:
        # Create indices for all positions at once
        all_step_indices = torch.arange(positions_to_fill, device=device) // n  # Which step (0, 1, ..., k-1)
        all_pos_indices = torch.arange(positions_to_fill, device=device) % n   # Which position within step (0, 1, ..., n-1)

        # Fill with function values - vectorized (now just token IDs)
        batch_inputs[:, :positions_to_fill] = all_functions[
            torch.arange(batch_size).unsqueeze(1),
            all_step_indices.unsqueeze(0),
            all_pos_indices.unsqueeze(0)
        ]

    # Set the starting value in the last position
    if seq_len > 0:
        batch_inputs[:, -1] = x_values



    # Encoding index in inputs

    # batch_inputs = batch_inputs + 10*n*torch.arange(seq_len, device=device).unsqueeze(0)

    #batch_inputs: [batch_size, seq_len]
    #batch_targets: [batch_size]

    return batch_inputs, batch_targets



