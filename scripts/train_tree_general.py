import os
import sys
sys.path.insert(0, '..')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import numpy as np
import random
import math
from src.tree_general import MultiLayerTransformer
from src.FC_random import function_composition
from src.trained_model import save_model_weights, load_model_weights



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
    d_ff = model_config.get('d_ff', d_model * 4)  # Default to 4x d_model if not specified

    # Unpack training configuration
    epochs = training_config.get('num_epochs', 1000)
    batch_size = training_config.get('batch_size', 64)  # Increased for better efficiency
    lr = training_config.get('learning_rate', 0.001)
    dropout = training_config.get('dropout', 0.1)
    device = training_config.get('device', 'cpu')

    #Validation config
    val_batch = val_config.get('batch_size', 16)




    # Unpack function composition settings
    n = func_comp.get('n', 25)
    k = func_comp.get('k', 2)


vocab_in1 = n
vocab_in2 = k*n + 1

seq_len = k*n+1


# model_poly = [{'t': 3, 'children_list': [[1], [2]]}, {'t': 3, 'children_list': [[1], [2]]}]
model_poly = [{'t': 3, 'children_list': [[1], [2]]}] * layers


model = MultiLayerTransformer(
    model_poly=model_poly,
    vocab_in1=vocab_in1,
    vocab_in2=vocab_in2,
    d_model=d_model,
    n_heads=n_heads,
    L=layers,
    d_ff=d_ff,
    device=device,
    dropout=dropout
    )




optimizer = optim.AdamW(model.parameters(), lr=lr)

Loss_train = []
Loss_test = []
Acc_test = []

# Index part of the input tuples
batch_idx_template = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(-1)
test_idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(16, -1).unsqueeze(-1)

# Pre-allocate for concatenation efficiency
X_shape = (batch_size, seq_len, 2)
test_X_shape = (val_batch, seq_len, 2)


for i in range(epochs):

    # Set model to training mode (enables dropout, etc.)
    model.train()
    optimizer.zero_grad()

    # Generate training data with current epoch as seed for variety
    batch_inputs, batch_targets = function_composition(
        seq_len=seq_len,
        batch_size=batch_size,
        n=n,
        k=k,
        device=device,
        seed=i  # Use epoch as seed for reproducible variety
        )



    # input tuples
    batch_idx = batch_idx_template.expand(batch_size, -1, -1)
    X = torch.cat([batch_inputs.unsqueeze(-1), batch_idx], dim=-1)

    pred = model(X)

    outputs = pred[:, -1, :]
    target_output = batch_targets

    # More efficient loss computation
    loss = F.cross_entropy(pred[:, -1, :], target_output)

    Loss_train.append(loss.item())

    loss.backward()
    optimizer.step()

    # Adaptive test frequency - less frequent early on, more frequent later
    # test_freq = max(25, 200 - i // 10)
    test_freq = 100
    if i % test_freq == 0:
        model.eval()

        test_inputs, test_targets = function_composition(
            seq_len=seq_len,
            batch_size=val_batch,
            n=n,
            k=k,
            device=device,
            seed=i + 10000  # Different seed for test data
            )
        with torch.no_grad():
            test_X = torch.cat([test_inputs.unsqueeze(-1), test_idx], dim=-1)
            test_pred = model(test_X)
            # More efficient test computations
            test_loss = F.cross_entropy(test_pred[:, -1, :], test_targets)
            accuracy = (test_pred[:, -1, :].argmax(dim=-1) == test_targets).float().mean()

            Loss_test.append(test_loss.item())
            Acc_test.append(accuracy.item())

        print(f"Epoch {i}, Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

        # Periodic memory cleanup for MPS
        if i % 500 == 0:
            torch.cuda.empty_cache()


#save the model to configs
save_model_weights(model, f"../models/transformer_weights_k{k}n{n}L{layers}d{d_model}.pt")

# string_suffix = "tree" if tree_used else "simple"
# np.savez(f"../data/{n}train{k}_sum_embed_{string_suffix}_L{layers}.npz",
#     Loss_train=np.array(Loss_train),
#     Loss_test=np.array(Loss_test),
#     Acc_test=np.array(Acc_test)
#     )





