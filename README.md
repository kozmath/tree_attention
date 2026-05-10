# Poly-Attention: A General Scheme for Higher-Order Self-Attention

This repository contains the official implementation of **Poly-Attention**, a general framework for constructing higher-order self-attention mechanisms published at **ICLR**.

Standard self-attention computes a weighted combination of value vectors using a single pair of query and key matrices. Poly-attention replaces this with a *tree-structured* polynomial composition of multiple query and value matrices, recovering standard attention as a special case while enabling strictly more expressive attention patterns.

## Repository Structure

```
tree_att_general/
├── configs/
│   └── config.yaml            # Hyperparameters for model and training
├── scripts/
│   ├── train_tree_general.py  # Training script using the general poly-attention transformer
│   └── train_FC.py            # Training script using the simpler 3-query chain attention
├── src/
│   ├── tree_general.py        # Core transformer with general tree-structured attention
│   ├── tree_simple.py         # Simplified transformer with fixed 3-query chain attention
│   └── FC_random.py           # Synthetic function composition task (data generation)
└── README.md
```

## How It Works

### General Poly-Attention (`src/tree_general.py`)

The attention mechanism is defined by a **tree**, where:
- Each node `i` has its own query matrix **Q**_i.
- Each edge (i → j) has an associated value matrix **V**_j.
- The attention output at the root is computed recursively: at every internal node, unnormalized attention scores across children are multiplied together (polynomial composition), and the final softmax normalization is applied at the root.

The tree structure is passed as a `model_poly` dictionary:

```python
model_poly = {
    't': 3,                        # total number of query matrices
    'children_list': [[1], [2]]    # children_list[i] = list of children of node i
}
```

This example defines a depth-2 chain: node 0 → node 1 → node 2, corresponding to the 3-query attention from the paper. Setting `model_poly=None` recovers standard self-attention.

Each transformer layer independently receives its own `model_poly`, so you can mix standard and poly-attention layers in a single model:

```python
model_poly = [
    {'t': 3, 'children_list': [[1], [2]]},  # layer 1: 3-query chain
    {'t': 3, 'children_list': [[1], [2]]},  # layer 2: 3-query chain
]
```

### Simple 3-Query Chain Attention (`src/tree_simple.py`)

A fixed, non-recursive implementation of the specific depth-2 chain case from the paper (3 queries Q1, Q2, Q3 with value matrices V2, V3). This is faster to run and easier to inspect. Used by `scripts/train_FC.py`.

### Synthetic Task: Function Composition (`src/FC_random.py`)

A synthetic algorithmic task designed to probe higher-order reasoning. Given `k` random functions f₁, …, f_k: [n] → [n] presented as lookup tables (kn tokens), and a starting value x ∈ [n] (the final token), the model must predict f_k(f_{k-1}(…f₁(x)…)).

## Configuration

All hyperparameters are set in `configs/config.yaml`:

```yaml
model:
  d_model: 48       # embedding dimension
  layers: 2         # number of transformer layers
  n_heads: 4        # number of attention heads

training:
  batch_size: 256
  device: mps       # use 'cuda', 'mps', or 'cpu'
  learning_rate: 0.001
  num_epochs: 50000
  dropout: 0.05

validation:
  batch_size: 16

func_composition:
  n: 5              # function domain size [n] = {0, ..., n-1}
  k: 3              # number of composed functions
```

## Installation

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, PyYAML, NumPy.

```bash
git clone https://github.com/sayaksc/tree_attention.git
cd tree_attention
pip install torch pyyaml numpy
```

## Running Experiments

All scripts must be run from the `scripts/` directory so that relative imports and config paths resolve correctly.

### General Poly-Attention (tree-structured, configurable)

```bash
cd scripts
python train_tree_general.py
```

This trains a `MultiLayerTransformer` from `src/tree_general.py` on the function composition task. The tree topology per layer is defined inline in the script via the `model_poly` list. Edit it to experiment with different tree shapes or depths.

### Simple 3-Query Chain Attention

```bash
cd scripts
python train_FC.py
```

This trains the simpler `MultiLayerTransformer` from `src/tree_simple.py`, which hard-codes the 3-query chain attention described in the paper. Useful as a baseline or sanity check before running the full general version.

### Changing the Device

Edit `configs/config.yaml` and set `device` to one of `cpu`, `cuda`, or `mps` (Apple Silicon). The data generation code will automatically fall back to CPU if the specified accelerator is unavailable.

### Adjusting Task Difficulty

In `configs/config.yaml`, increase `k` to compose more functions (harder) or `n` to enlarge the function domain (larger vocabulary). The sequence length is automatically set to `k*n + 1`.

## Citation

If you use this code, please cite the corresponding ICLR paper.
