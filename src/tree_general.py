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
from collections import deque



class Tree:
    def __init__(self, index=None):
        self.index = index
        self.children = []
    
    def create_tree(self, children_list):
        """
        Recursively builds the tree structure from a children list.
        
        Args:
            children_list: List where children_list[i] contains the indices of children of node i.
                          If a node has no children, children_list[i] should be None or empty.
        """
        # Guard against None index and out-of-bounds access
        if self.index is None or self.index >= len(children_list):
            return
        
        # Get children indices for this node
        node_children = children_list[self.index]
        
        # If node has no children, return (base case for recursion)
        if node_children is None or len(node_children) == 0:
            return
        
        # Recursively create child nodes
        for child_index in node_children:
            child_node = Tree(child_index)
            self.children.append(child_node)
            # Recursively build subtree for this child
            child_node.create_tree(children_list)


class MultiLayerTransformer(nn.Module):
    def __init__(self, model_poly, vocab_in1, vocab_in2, d_model, n_heads, L, d_ff=None, vocab_out=None, device='cpu', dropout=0, bias=False):
        super().__init__()

        #Transformer parameters

        if vocab_out is None:
            vocab_out = vocab_in1

        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4*self.d_model
        self.n_heads = n_heads
        self.device = device

        # One embedding function for the value of f_i(j), another embedding function for (i*n + j)
        self.embed1 = nn.Embedding(vocab_in1, self.d_model)
        self.embed2 = nn.Embedding(vocab_in2, self.d_model)

        self.LM = nn.Linear(self.d_model, vocab_out, bias=bias)

        self.multi_layer_transformer = nn.ModuleList()



        #Create the multi layer transformer
        for i in range(L):
            layer = Transformer(model_poly[i], d_model, n_heads, self.d_ff, dropout)
            # layer = Transformer(model_poly, d_model, n_heads, d_ff, dropout)
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
    def __init__(self, model_poly, d_model, n_heads, d_ff, dropout=0, bias=False):
        '''
        model_poly is a dictionary containing keys 't' (int) and 
        'children_list' (list of lists, where children_list[i] contains the indices of children of node i)
        '''
        
        super().__init__()


        self.model_poly = model_poly
        if self.model_poly is not None:
            self.tree = Tree(0)
            self.tree.create_tree(model_poly['children_list'])
            self.t = model_poly['t']



        self.d_model = d_model
        self.n_heads = n_heads
        self.d_h = d_model // n_heads

        self.SA_dropout = nn.Dropout(dropout)
        self.MLP_dropout = nn.Dropout(dropout)

        if self.model_poly is not None:
            self.WQi = nn.ModuleList([
                nn.Linear(self.d_model, self.d_model, bias=bias)
                for _ in range(self.t)
            ])

            self.WVi = nn.ModuleList([
                nn.Linear(self.d_model, self.d_model, bias=bias)
                for _ in range(self.t - 1)
            ])
        else:
            self.WQ = nn.Linear(self.d_model, self.d_model, bias=bias)
            self.WK = nn.Linear(self.d_model, self.d_model, bias=bias)
            self.WV = nn.Linear(self.d_model, self.d_model, bias=bias)


        self.W_h = nn.Linear(self.d_model, d_model, bias=bias) if n_heads > 1 else None

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)



        self.MLP = nn.Sequential(
            nn.Linear(self.d_model, d_ff),
            nn.ReLU(),
            self.MLP_dropout,
            nn.Linear(d_ff, self.d_model)
        )


    def SelfAttention(self, Q, K, V, mask=None):
        # Q, K, V given as (B, n_heads, n, d_h)

        QK = Q @ K.transpose(-2, -1) / math.sqrt(self.d_h)
        if mask is not None:
            QK = QK.masked_fill(mask==0, -1e9)
        A = torch.softmax(QK, dtype=torch.float32, dim=-1)
        A = self.SA_dropout(A)

        return A @ V


    
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


        #Dropout needs to be corrected later.
        eQ12 = self.SA_dropout(eQ12)
        eQ23 = self.SA_dropout(eQ23)

        # # Optimized computation without for loop

        eQ23V3 = eQ23 @ V3 #[batch, heads, seq_len, d_h]
        eQ23V3 = eQ23V3.transpose(-2, -1).unsqueeze(-1)  #[batch, heads, d_h, seq_len, 1]
        V2 = V2.transpose(-2, -1).unsqueeze(-1)  #[batch, heads, d_h, seq_len, 1]
        # V2eQ23V3 =  eQ23V3  #[batch, heads, d_h, seq_len, 1]
        # V2eQ23V3 = V2eQ23V3.squeeze(-1).transpose(-2, -1)  #[batch, heads, seq_len, d_h]
        eQ12V2eQ23V3 = (eQ12.unsqueeze(2)*V2).squeeze(2) @ eQ23V3  #[batch, heads, seq_len, d_h]
        P23 = eQ12V2eQ23V3.squeeze(-1).transpose(-2, -1)  #[batch, heads, seq_len, d_h]


        result = P23 / R

        return result





    def TreeAttention_General(self, root, Qi, Vi, mask=None):

        '''
        Inputs:         tree structure,
                        Qi: list of Q matrices for each level of the tree (length t),
                            dimension: [t, batch, heads, seq_len, d_h]
                        Vi: list of V matrices for each level of the tree (length t-1)
                            dimension: [t-1, batch, heads, seq_len, d_h]
                        mask: optional mask for attention
        Outputs:        Tuple of (P, R) where P is the numerator and R is the denominator
                        P has dimension [batch, heads, d_h, seq_len, 1]
                        R has dimension [batch, heads, seq_len, 1]
        '''



        branch_P = [] #Collection of [batch, heads, d_h, seq_len, 1] tensors from each branch
        branch_R = [] #Collection of [batch, heads, seq_len, 1] tensors from each branch

        i = root.index


        for child in root.children:
            
            j = child.index


            if len(child.children) == 0:
                
                QiQj = Qi[i] @ Qi[j].transpose(-2, -1)/(math.sqrt(self.d_h)) # [batch, heads, seq_len, seq_len]
                QiQj = QiQj.clamp(max=20, min=-20)


                scores_ij = torch.exp(QiQj) # [batch, heads, seq_len, seq_len]
                scores_ij = self.SA_dropout(scores_ij)

                R_i = scores_ij.sum(dim=-1, keepdim=True) # [batch, heads, seq_len, 1]

                P_i = scores_ij@ Vi[j-1] # [batch, heads, seq_len, d_h]
                P_i = P_i.transpose(-2, -1).unsqueeze(-1) # [batch, heads, d_h, seq_len, 1]

                branch_R.append(R_i)
                branch_P.append(P_i)


            else:
                P_j, R_j = self.TreeAttention_General(child, Qi, Vi, mask) # [batch, heads, d_h, seq_len, 1], [batch, heads, seq_len, 1]
                V_j = Vi[j-1].transpose(-2, -1).unsqueeze(-1) # [batch, heads, d_h, seq_len, 1]

                QiQj = Qi[i] @ Qi[j].transpose(-2, -1)/(math.sqrt(self.d_h)) # [batch, heads, seq_len, seq_len]
                QiQj = QiQj.clamp(max=20, min=-20)

                scores_ij = torch.exp(QiQj) # [batch, heads, seq_len, seq_len]
                scores_ij = self.SA_dropout(scores_ij)

                R_i = scores_ij @ R_j # [batch, heads, seq_len, 1]

                QiQjVj = (scores_ij.unsqueeze(2)*V_j) # [batch, heads, d_h, seq_len, seq_len]
                
                P_i = QiQjVj @ P_j # [batch, heads, d_h, seq_len, 1]

                branch_R.append(R_i)
                branch_P.append(P_i)

        
        # Combine branches at this node
        P_final = torch.ones_like(branch_P[0]) # [batch, heads, d_h, seq_len, 1]
        R_final = torch.ones_like(branch_R[0]) # [batch, heads, seq_len, 1]
        for P_i, R_i in zip(branch_P, branch_R):
            P_final = P_final * P_i
            R_final = R_final * R_i
        
        return P_final, R_final # [batch, heads, d_h, seq_len, 1], [batch, heads, seq_len, 1]








    def forward_SA(self, X):
        
        # X given as (B, n, d_model)

        Q = self.WQ(X) # (B, n, d_model)
        K = self.WK(X)
        V = self.WV(X)

        batch_size = X.shape[0]
        seq_len = X.shape[1]

        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_h).permute(0, 2, 1, 3) # (B, n_heads, n, d_h)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_h).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_h).permute(0, 2, 1, 3)

        SA_output = self.SelfAttention(Q, K, V) # (B, n_heads, n, d_h)
        SA_output = SA_output.permute(0, 2, 1, 3) #(B, n, n_heads, d_h)

        if self.W_h is not None:

            SA_output = self.W_h(SA_output.reshape(batch_size, seq_len, self.d_model)) # (B, n, d_model)
        else:
            SA_output = SA_output.view(batch_size, seq_len, self.d_model)  # (B, n, d_model)


        SA_output = self.SA_dropout(SA_output)
        SA_output = self.norm1(X + SA_output)

        MLP_output = self.MLP(SA_output)

        return self.norm2(MLP_output + SA_output)



    def forward_TA(self, X):

        # X given as (B, n, d_model)
        batch_size = X.shape[0]
        seq_len = X.shape[1]


        Qi = []
        Vi = []
        for Wq in self.WQi:
            Qi_1 = Wq(X) # (B, n, d_model)
            Qi_1 = Qi_1.view(batch_size, seq_len, self.n_heads, self.d_h).permute(0, 2, 1, 3) # (B, n_heads, n, d_h)
            Qi.append(Qi_1)
        for Wv in self.WVi:
            Vi_1 = Wv(X) # (B, n, d_model)
            Vi_1 = Vi_1.view(batch_size, seq_len, self.n_heads, self.d_h).permute(0, 2, 1, 3) # (B, n_heads, n, d_h)
            Vi.append(Vi_1)

        

        P, R = self.TreeAttention_General(self.tree, Qi, Vi) # [B, n_heads, d_h, seq_len, 1], [B, n_heads, seq_len, 1]
        P = P.squeeze(-1).transpose(-2, -1) # [B, n_heads, seq_len, d_h]

        SA_output = P / R # [B, n_heads, seq_len, d_h]
        SA_output = SA_output.permute(0, 2, 1, 3) #(B, n, n_heads, d_h)

        if self.W_h is not None:

            SA_output = self.W_h(SA_output.reshape(batch_size, seq_len, self.d_model)) # (B, n, d_model)
        else:
            SA_output = SA_output.view(batch_size, seq_len, self.d_model)  # (B, n, d_model)


        SA_output = self.SA_dropout(SA_output)
        SA_output = self.norm1(X + SA_output)

        MLP_output = self.MLP(SA_output)

        return self.norm2(MLP_output + SA_output)







    def forward(self, X):
        if self.model_poly is None:
            return self.forward_SA(X)

        else:
            return self.forward_TA(X)
        









