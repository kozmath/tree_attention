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




class MultiLayerTransformer(nn.Module):
    def __init__(self, vocab_in1, vocab_in2, d_model, n_heads, L, d_ff, vocab_out=None, device='cpu', dropout=0, bias=False):
        super().__init__()

        #Transformer parameters

        if vocab_out is None:
            vocab_out = vocab_in1

        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.device = device

        # One embedding function for the value of f_i(j), another embedding function for (i*n + j)
        self.embed1 = nn.Embedding(vocab_in1, self.d_model)
        self.embed2 = nn.Embedding(vocab_in2, self.d_model)

        self.LM = nn.Linear(self.d_model, vocab_out, bias=bias)

        self.multi_layer_transformer = nn.ModuleList()



        #Create the multi layer transformer
        for i in range(L):
            layer = Transformer(d_model, n_heads, d_ff, dropout)

            self.multi_layer_transformer.append(layer)

        self.to(self.device)

    def pos_encoding(self, X):


        n = X.shape[-2]

        pe = torch.zeros(n, self.d_model, dtype=torch.float32, device=self.device)

        position = torch.arange(n, dtype=torch.float32, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32, device=self.device) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)

        if self.d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        return pe


    def forward(self, X):
        #X given as a tensor of dim (batch, seq_len, 2)

        X0 = X[:, :, 0]
        y = X[:, :, 1]

        X_embed = self.embed1(X0) # (B, n, d)
        X_embed = X_embed + self.embed2(y) # (B, n, d)
        pe = self.pos_encoding(X_embed)


        X_pos = X_embed + pe.unsqueeze(0).expand(X.shape[0], -1, -1) # (B, n, d)



        for layer in self.multi_layer_transformer:
            X_pos = layer(X_pos)


        return self.LM(X_pos)


class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0, bias=False):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_h = d_model // n_heads

        self.SA_dropout = nn.Dropout(dropout)
        self.MLP_dropout = nn.Dropout(dropout)

        self.W_Q1 = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.W_Q2 = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.W_Q3 = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.W_V2 = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.W_V3 = nn.Linear(self.d_model, self.d_model, bias=bias)

        self.W_h = nn.Linear(self.d_model, d_model, bias=bias) if n_heads > 1 else None

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)



        self.MLP = nn.Sequential(
            nn.Linear(self.d_model, d_ff),
            nn.ReLU(),
            self.MLP_dropout,
            nn.Linear(d_ff, self.d_model)
        )

    def TreeAttention_Simple(self, Q1, Q2, Q3, V2, V3, mask=None):
        # Handle  [batch, heads, seq_len, d_h]

        # Compute attention scores
        scores12 = torch.matmul(Q1, Q2.transpose(-2, -1)) / (2*math.sqrt(self.d_h))
        scores23 = torch.matmul(Q2, Q3.transpose(-2, -1)) / (2*math.sqrt(self.d_h))

        scores12 = scores12.clamp(max=20, min=-20)

        scores23 = scores23.clamp(max=20, min=-20)


        if mask is not None:
            scores12 = scores12.masked_fill(mask==0, -1e9)
            scores23 = scores23.masked_fill(mask==0, -1e9)



        eQ12 = torch.exp(scores12)
        eQ23 = torch.exp(scores23)




        # Computing denominator of softmax
        sum_eQ23 = eQ23.sum(dim=-1, keepdim=True)
        R = torch.max(torch.matmul(eQ12, sum_eQ23), torch.tensor(1e-9))  # Prevent division by zero

        eQ12 = self.SA_dropout(eQ12)
        eQ23 = self.SA_dropout(eQ23)

        # # Optimized computation without for loop

        eQ23V3 = eQ23 @ V3 #[batch, heads, seq_len, d_h]
        eQ23V3 = eQ23V3.transpose(-2, -1).unsqueeze(-1)  #[batch, heads, d_h, seq_len, 1]
        V2 = V2.transpose(-2, -1).unsqueeze(-1)  #[batch, heads, d_h, seq_len, 1]
        # V2eQ23V3 =  eQ23V3  #[batch, heads, d_h, seq_len, 1]
        # V2eQ23V3 = V2eQ23V3.squeeze(-1).transpose(-2, -1)  #[batch, heads, seq_len, d_h]
        eQ12V2eQ23V3 = (eQ12.unsqueeze(2)*V2) @ eQ23V3  #[batch, heads, seq_len, d_h]
        P23 = eQ12V2eQ23V3.squeeze(-1).transpose(-2, -1)  #[batch, heads, seq_len, d_h]


        result = P23 / R

        return result





    def forward(self, X):
        # X given as (B, n, d_model)

        Q1 = self.W_Q1(X) # (B, n, d_model)
        Q2 = self.W_Q2(X)
        Q3 = self.W_Q3(X)
        V2 = self.W_V2(X)
        V3 = self.W_V3(X)

        batch_size = X.shape[0]
        seq_len = X.shape[1]

        Q1 = Q1.view(batch_size, seq_len, self.n_heads, self.d_h).permute(0, 2, 1, 3) # (B, n_heads, n, d_h)
        Q2 = Q2.view(batch_size, seq_len, self.n_heads, self.d_h).permute(0, 2, 1, 3)
        Q3 = Q3.view(batch_size, seq_len, self.n_heads, self.d_h).permute(0, 2, 1, 3)
        V2 = V2.view(batch_size, seq_len, self.n_heads, self.d_h).permute(0, 2, 1, 3)
        V3 = V3.view(batch_size, seq_len, self.n_heads, self.d_h).permute(0, 2, 1, 3)

        SA_output = self.TreeAttention_Simple(Q1, Q2, Q3, V2, V3) # (B, n_heads, n, d_h)
        SA_output = SA_output.permute(0, 2, 1, 3) #(B, n, n_heads, d_h)

        if self.W_h is not None:

            SA_output = self.W_h(SA_output.reshape(batch_size, seq_len, self.d_model)) # (B, n, d_model)
        else:
            SA_output = SA_output.view(batch_size, seq_len, self.d_model)  # (B, n, d_model)


        SA_output = self.SA_dropout(SA_output)
        SA_output = self.norm1(X + SA_output)

        MLP_output = self.MLP(SA_output)

        return self.norm2(MLP_output + SA_output)










