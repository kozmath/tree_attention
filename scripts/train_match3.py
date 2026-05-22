import os
import sys
sys.path.insert(0, "..")

import random
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.tree_general import MultiLayerTransformer


# ============================================================
# Strassen-paper Match3 data generation, adapted for tree attention
# ============================================================

def match3_labels_single(x, M=37, allow_reuse=True):
    """
    x: LongTensor [n]
    returns y: LongTensor [n]

    y_i = 1 iff exists j,k such that:
        x_i + x_j + x_k == 0 mod M

    The Strassen paper's Match3 definition allows j,k in [n].
    It does not impose distinctness.
    """
    n = x.shape[0]
    y = torch.zeros(n, dtype=torch.long)

    if allow_reuse:
        pair_sums = (x.unsqueeze(0) + x.unsqueeze(1)) % M  # [n, n]

        for i in range(n):
            needed = (-x[i]) % M
            y[i] = (pair_sums == needed).any().long()

    else:
        for i in range(n):
            found = False
            for j in range(n):
                for k in range(n):
                    if i == j or i == k or j == k:
                        continue
                    if (x[i] + x[j] + x[k]) % M == 0:
                        found = True
                        break
                if found:
                    break
            y[i] = int(found)

    return y


def bin_id_from_labels(y):
    """
    Four bins by percentage of ones:
        0: [0, 25)%
        1: [25, 50)%
        2: [50, 75)%
        3: [75, 100]%
    """
    p = y.float().mean().item()

    if p < 0.25:
        return 0
    elif p < 0.50:
        return 1
    elif p < 0.75:
        return 2
    else:
        return 3


def random_match3_sequence_with_skew(
    n,
    M=37,
    target_min_percent=0.0,
    max_tries=10_000,
    allow_reuse=True,
):
    """
    Generates one sequence whose percentage of positive Match3 labels
    is at least target_min_percent.

    This approximates the paper's Algorithm 3 line:
        ensuring the percentage of tokens that satisfied Match3 condition
        is at least skewness

    Here skewness is interpreted as a percentage in [1, 40].
    """
    target_min_rate = target_min_percent / 100.0

    for _ in range(max_tries):
        x = torch.randint(0, M, (n,), dtype=torch.long)
        y = match3_labels_single(x, M=M, allow_reuse=allow_reuse)

        if y.float().mean().item() >= target_min_rate:
            return x, y

    # Fallback: return the last sample if rejection fails.
    return x, y


def pad_example(x, y, N_max, pad_token, ignore_index=-100):
    """
    Pads x and y to N_max.
    x is padded with pad_token.
    y is padded with ignore_index.
    """
    n = x.shape[0]

    x_pad = torch.full((N_max,), pad_token, dtype=torch.long)
    y_pad = torch.full((N_max,), ignore_index, dtype=torch.long)
    mask = torch.zeros(N_max, dtype=torch.bool)

    x_pad[:n] = x
    y_pad[:n] = y
    mask[:n] = True

    return x_pad, y_pad, mask


def permute_padded_example(x, y, mask, pad_token, ignore_index=-100):
    """
    Applies the same random permutation to the valid part of X and Y.
    Padding stays at the end.
    """
    valid_n = mask.long().sum().item()

    x_valid = x[:valid_n]
    y_valid = y[:valid_n]

    perm = torch.randperm(valid_n)

    x_perm = x_valid[perm]
    y_perm = y_valid[perm]

    x_out = torch.full_like(x, pad_token)
    y_out = torch.full_like(y, ignore_index)
    mask_out = torch.zeros_like(mask)

    x_out[:valid_n] = x_perm
    y_out[:valid_n] = y_perm
    mask_out[:valid_n] = True

    return x_out, y_out, mask_out

def generate_strassen_match3_dataset_safe(
    D=50_000,
    N_min=30,
    N_max=35,
    M=37,
    allow_reuse=True,
    seed=0,
    ignore_index=-100,
    candidate_multiplier=20,
    min_candidates_per_bin=1,
):
    """
    Safe finite-pass Match3 dataset generator.

    No unbounded while loops.

    Procedure:
      1. Generate a finite candidate pool.
      2. Assign each candidate to one of the four label-density bins.
      3. Fill each bin to D/4 by sampling with replacement from candidates in that bin.
      4. If a bin is empty, raise a clear error instead of looping forever.

    Returns:
        X:    LongTensor [D, N_max]
        Y:    LongTensor [D, N_max]
        MASK: BoolTensor [D, N_max]
    """
    assert D % 4 == 0, "D must be divisible by 4."

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    pad_token = M
    target_per_bin = D // 4

    # Large but finite candidate pool.
    # For D=50k and multiplier=20, this samples 1M examples.
    # You can reduce this if generation is slow.
    num_candidates = D * candidate_multiplier

    bins = [[] for _ in range(4)]

    for t in range(num_candidates):
        if t%10000 == 0:
            print(f"Generated {t}/{num_candidates} candidates...")
        n = random.randint(N_min, N_max)

        # Experimental appendix version: x_i in {0, ..., M-1}
        x = torch.randint(0, M, (n,), dtype=torch.long)

        # Formal no-zero variant:
        # x = torch.randint(1, M, (n,), dtype=torch.long)

        y = match3_labels_single(
            x=x,
            M=M,
            allow_reuse=allow_reuse,
        )

        b = bin_id_from_labels(y)

        x_pad, y_pad, mask = pad_example(
            x=x,
            y=y,
            N_max=N_max,
            pad_token=pad_token,
            ignore_index=ignore_index,
        )

        bins[b].append((x_pad, y_pad, mask))

    bin_counts = [len(b) for b in bins]
    print("Candidate bin counts:", bin_counts)

    empty_bins = [i for i, c in enumerate(bin_counts) if c < min_candidates_per_bin]

    if len(empty_bins) > 0:
        raise RuntimeError(
            f"Empty or underfilled Match3 bins: {empty_bins}. "
            f"Candidate counts were {bin_counts}. "
            f"This means the chosen N/M distribution almost never produces those label densities. "
            f"Try larger M, smaller N, allow_reuse=False, or disable four-bin balancing."
        )

    full_examples = []

    for b in range(4):
        source_bin = bins[b]

        # Sample with replacement to exactly D/4 examples.
        for _ in range(target_per_bin):
            x, y, mask = random.choice(source_bin)

            # Apply permutation augmentation.
            xp, yp, mp = permute_padded_example(
                x=x,
                y=y,
                mask=mask,
                pad_token=pad_token,
                ignore_index=ignore_index,
            )

            full_examples.append((xp, yp, mp))

    random.shuffle(full_examples)

    X = torch.stack([e[0] for e in full_examples], dim=0)
    Y = torch.stack([e[1] for e in full_examples], dim=0)
    MASK = torch.stack([e[2] for e in full_examples], dim=0)

    return X, Y, MASK
# ============================================================
# Training
# ============================================================

def make_position_channel(batch_size, seq_len, device):
    return torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)


def masked_token_accuracy(logits, targets, ignore_index=-100):
    """
    logits: [B, N, 2]
    targets: [B, N]
    """
    pred = logits.argmax(dim=-1)
    mask = targets != ignore_index

    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    return (pred[mask] == targets[mask]).float().mean()


def evaluate(model, loader, device, seq_len, ignore_index=-100):
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch_inputs, batch_targets, batch_mask in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            B = batch_inputs.shape[0]
            batch_idx = make_position_channel(B, seq_len, device)

            X = torch.cat([batch_inputs.unsqueeze(-1), batch_idx], dim=-1)

            pred = model(X)

            # Your model likely returns [B, N, vocab_in1].
            # Match3 only needs binary labels, so use first two logits.
            logits = pred[:, :, :2]

            loss = F.cross_entropy(
                logits.reshape(-1, 2),
                batch_targets.reshape(-1),
                ignore_index=ignore_index,
            )

            acc = masked_token_accuracy(
                logits=logits,
                targets=batch_targets,
                ignore_index=ignore_index,
            )

            total_loss += loss.item()
            total_acc += acc.item()
            total_batches += 1

    return total_loss / total_batches, total_acc / total_batches


def main():
    with open("../configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_config = config.get("model", {})
    training_config = config.get("training", {})
    match3_config = config.get("match3", {})

    # ------------------------------------------------------------
    # Strassen paper Match3 settings
    # ------------------------------------------------------------

    D = match3_config.get("dataset_size", 5000)
    N_min = match3_config.get("N_min", 30)
    N_max = match3_config.get("N_max", 35)
    M = match3_config.get("M", 2048)
    allow_reuse = match3_config.get("allow_reuse", True)
    seed = match3_config.get("seed", 0)

    ignore_index = -100
    pad_token = M

    # Paper settings for Match3.
    d_model = model_config.get("d_model", 128)
    n_heads = model_config.get("n_heads", 2)
    layers = model_config.get("layers", 1)
    d_ff = model_config.get("d_ff", 4 * d_model)

    epochs = training_config.get("num_epochs", 500)
    batch_size = training_config.get("batch_size", 2500)
    lr = training_config.get("learning_rate", 1e-3)
    dropout = training_config.get("dropout", 0.4)
    device = training_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    seq_len = N_max

    # Values are 0,...,36 plus PAD token 37.
    # So input-value vocab size is M + 1.
    vocab_in1 = M + 1

    # Position IDs are 0,...,N_max-1.
    vocab_in2 = N_max

    print("Generating Match3 dataset...")
    X_all, Y_all, MASK_all = generate_strassen_match3_dataset_safe(
        D=D,
        N_min=N_min,
        N_max=N_max,
        M=M,
        allow_reuse=allow_reuse,
        seed=seed,
        ignore_index=ignore_index,
        candidate_multiplier=20,
    )

    valid_targets = Y_all[Y_all != ignore_index]
    print("Dataset X:", X_all.shape)
    print("Dataset Y:", Y_all.shape)
    print("Positive rate:", valid_targets.float().mean().item())
    print("Label counts:", torch.unique(valid_targets, return_counts=True))

    # 90/10 train/validation split.
    n_train = int(0.9 * D)

    X_train = X_all[:n_train]
    Y_train = Y_all[:n_train]
    MASK_train = MASK_all[:n_train]

    X_val = X_all[n_train:]
    Y_val = Y_all[n_train:]
    MASK_val = MASK_all[n_train:]

    train_dataset = TensorDataset(X_train, Y_train, MASK_train)
    val_dataset = TensorDataset(X_val, Y_val, MASK_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # ------------------------------------------------------------
    # Tree-attention model
    # ------------------------------------------------------------
    # For 3-token interaction, use a 3-child tree polynomial.
    #
    # This is analogous to using a third-order interaction for Match3,
    # but through your tree-attention interface.
    # ------------------------------------------------------------

    layers = 1
    model_poly = [
        {"t": 4, "children_list": [[1, 2, 3]]}
    ]

    model = MultiLayerTransformer(
        model_poly=model_poly,
        vocab_in1=vocab_in1,
        vocab_in2=vocab_in2,
        d_model=d_model,
        n_heads=n_heads,
        L=layers,
        d_ff=d_ff,
        device=device,
        dropout=dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    Loss_train = []
    Loss_val = []
    Acc_val = []

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    print("Starting training...")
    for epoch in range(epochs):
        model.train()

        epoch_loss = 0.0
        num_batches = 0

        for batch_inputs, batch_targets, batch_mask in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            B = batch_inputs.shape[0]

            batch_idx = make_position_channel(B, seq_len, device)

            # X shape: [B, N_max, 2]
            # X[:, :, 0] = token value in {0,...,36, PAD=37}
            # X[:, :, 1] = position ID in {0,...,34}
            X = torch.cat([batch_inputs.unsqueeze(-1), batch_idx], dim=-1)

            optimizer.zero_grad()

            pred = model(X)

            # Binary logits for Match3.
            logits = pred[:, :, :2]

            loss = F.cross_entropy(
                logits.reshape(-1, 2),
                batch_targets.reshape(-1),
                ignore_index=ignore_index,
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        Loss_train.append(avg_train_loss)

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            seq_len=seq_len,
            ignore_index=ignore_index,
        )

        Loss_val.append(val_loss)
        Acc_val.append(val_acc)

        print(
            f"Epoch {epoch:04d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Token Acc: {val_acc:.4f}"
        )

        if device == "cuda" and epoch % 50 == 0:
            torch.cuda.empty_cache()

    os.makedirs("../data", exist_ok=True)

    np.savez(
        f"../data/tree_attention_match3_strassen_setting_M{M}_N{N_min}_{N_max}.npz",
        Loss_train=np.array(Loss_train),
        Loss_val=np.array(Loss_val),
        Acc_val=np.array(Acc_val),
    )


if __name__ == "__main__":
    main()