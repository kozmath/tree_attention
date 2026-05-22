import torch
from src.tree_general import MultiLayerTransformer


def save_model_weights(model, filepath):
    """
    Save transformer model weights and configuration to a file.
    
    Args:
        model: The transformer model to save
        filepath: Path where to save the weights
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'd_model': model.d_model,
            'n_heads': model.n_heads,
            'layers': model.layers,
            'd_ff': model.d_ff,
            'dropout': model.dropout,
        }
    }
    torch.save(checkpoint, filepath)
    print(f"Model weights saved to {filepath}")


def load_model_weights(filepath, device, model_poly, vocab_in1, vocab_in2):
    """
    Load transformer model weights from a file.
    
    Args:
        filepath: Path to the saved weights
        device: Device to load the model on
        model_poly: Model architecture specification
        vocab_in1: Input vocabulary size 1
        vocab_in2: Input vocabulary size 2
    
    Returns:
        model: Loaded transformer model
    """
    checkpoint = torch.load(filepath, map_location=device)
    config = checkpoint['model_config']
    
    # Recreate model with saved configuration
    model = MultiLayerTransformer(
        model_poly=model_poly,
        vocab_in1=vocab_in1,
        vocab_in2=vocab_in2,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        L=config['layers'],
        d_ff=config['d_ff'],
        device=device,
        dropout=config['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model weights loaded from {filepath}")
    return model


# Example usage:
# After training, save the model:
# save_model_weights(model, "../models/transformer_weights.pt")
#
# To load the model later:
# loaded_model = load_model_weights("../models/transformer_weights.pt", device)