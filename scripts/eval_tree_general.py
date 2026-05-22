import os
import sys
sys.path.insert(0, '..')
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
from src.tree_general import MultiLayerTransformer
from src.FC_random import function_composition
from src.trained_model import load_model_weights


# Load configuration
with open("../configs/config.yaml", 'r') as f:
    config = yaml.safe_load(f)
    model_config = config['model']
    training_config = config['training']
    func_comp = config['func_composition']
    val_config = config['validation']

    # Unpack model configuration
    d_model = model_config.get('d_model', 16)
    n_heads = model_config.get('n_heads', 4)
    layers = model_config.get('layers', 3)
    d_ff = model_config.get('d_ff', d_model * 4)

    # Unpack training configuration
    device = training_config.get('device', 'cpu')
    dropout = training_config.get('dropout', 0.1)

    # Validation config
    val_batch = val_config.get('batch_size', 16)

    # Unpack function composition settings
    n = func_comp.get('n', 25)
    k = func_comp.get('k', 2)


vocab_in1 = n
vocab_in2 = k*n + 1
seq_len = k*n + 1

# Model architecture
model_poly = [{'t': 3, 'children_list': [[1], [2]]}] * layers


def evaluate_model(model, num_eval_batches=50, eval_batch_size=None, seed_offset=20000):
    """
    Evaluate the tree attention model on function composition task.
    
    Args:
        model: The transformer model to evaluate
        num_eval_batches: Number of evaluation batches to run
        eval_batch_size: Batch size for evaluation (defaults to val_batch)
        seed_offset: Offset for random seed to ensure different data from training
    
    Returns:
        dict: Dictionary containing evaluation metrics and predictions
    """
    if eval_batch_size is None:
        eval_batch_size = val_batch
    
    model.eval()
    
    total_loss = 0.0
    total_accuracy = 0.0
    all_preds = []
    all_targets = []
    batch_losses = []
    batch_accuracies = []
    
    eval_idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(eval_batch_size, -1).unsqueeze(-1)
    
    print(f"Evaluating on {num_eval_batches} batches with batch size {eval_batch_size}...")
    
    with torch.no_grad():
        for batch_idx in range(num_eval_batches):
            # Generate evaluation data
            eval_inputs, eval_targets = function_composition(
                seq_len=seq_len,
                batch_size=eval_batch_size,
                n=n,
                k=k,
                device=device,
                seed=seed_offset + batch_idx
            )
            
            # Prepare input: concatenate function values and indices
            eval_X = torch.cat([eval_inputs.unsqueeze(-1), eval_idx], dim=-1)
            
            # Forward pass
            eval_pred = model(eval_X)
            
            # Compute loss and accuracy on the final output
            loss = F.cross_entropy(eval_pred[:, -1, :], eval_targets)
            preds = eval_pred[:, -1, :].argmax(dim=-1)
            accuracy = (preds == eval_targets).float().mean()
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            batch_losses.append(loss.item())
            batch_accuracies.append(accuracy.item())
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(eval_targets.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{num_eval_batches}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")
    
    # Compute average metrics
    avg_loss = total_loss / num_eval_batches
    avg_accuracy = total_accuracy / num_eval_batches
    
    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Compute standard deviations
    loss_std = np.std(batch_losses)
    accuracy_std = np.std(batch_accuracies)
    
    results = {
        'avg_loss': avg_loss,
        'avg_accuracy': avg_accuracy,
        'loss_std': loss_std,
        'accuracy_std': accuracy_std,
        'predictions': all_preds,
        'targets': all_targets,
        'num_batches': num_eval_batches,
        'batch_losses': np.array(batch_losses),
        'batch_accuracies': np.array(batch_accuracies),
    }
    
    return results


def print_eval_summary(results):
    """
    Print a summary of evaluation results.
    
    Args:
        results: Dictionary returned from evaluate_model()
    """
    print("\n" + "="*60)
    print("TREE ATTENTION EVALUATION RESULTS")
    print("="*60)
    print(f"Average Loss: {results['avg_loss']:.4f} ± {results['loss_std']:.4f}")
    print(f"Average Accuracy: {results['avg_accuracy']:.4f} ± {results['accuracy_std']:.4f}")
    print(f"Total Batches Evaluated: {results['num_batches']}")
    print(f"Total Samples Evaluated: {len(results['targets'])}")
    print(f"Correct Predictions: {(results['predictions'] == results['targets']).sum()} / {len(results['targets'])}")
    print("="*60)


def save_eval_results(results, output_dir="../results"):
    """
    Save evaluation results to disk.
    
    Args:
        results: Dictionary returned from evaluate_model()
        output_dir: Directory to save results to
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, "eval_results.npz")
    np.savez(
        results_path,
        avg_loss=results['avg_loss'],
        avg_accuracy=results['avg_accuracy'],
        loss_std=results['loss_std'],
        accuracy_std=results['accuracy_std'],
        predictions=results['predictions'],
        targets=results['targets'],
        batch_losses=results['batch_losses'],
        batch_accuracies=results['batch_accuracies']
    )
    print(f"\nEvaluation results saved to {results_path}")


if __name__ == "__main__":
    # Generate model path based on configuration
    model_path = f"../models/transformer_weights_k{k}n{n}L{layers}d{d_model}.pt"
    
    # Check if model weights exist
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        print("Available model files:")
        models_dir = "../models"
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                print(f"  - {f}")
        else:
            print(f"  {models_dir} directory does not exist")
        print("\nPlease train the model first using train_tree_general.py")
        sys.exit(1)
    
    # Load the model
    print(f"Loading model from {model_path}...")
    print(f"Configuration: n={n}, k={k}, layers={layers}, d_model={d_model}, n_heads={n_heads}")
    model = load_model_weights(model_path, device, model_poly, vocab_in1, vocab_in2)
    
    # Evaluate the model
    print(f"\nEvaluating on device: {device}")
    results = evaluate_model(model, num_eval_batches=50, eval_batch_size=val_batch)
    
    # Print evaluation results
    print_eval_summary(results)
    
    # Save evaluation results
    save_eval_results(results)
