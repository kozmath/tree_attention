import torch
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

weights = torch.load("transformer_weights_k2n50L1d32.pt", map_location="cpu")
#print(weights["model_state_dict"].keys())

A = weights["model_state_dict"]["multi_layer_transformer.0.WQi.0.weight"]

B = weights["model_state_dict"]["multi_layer_transformer.0.WQi.1.weight"]

C = weights["model_state_dict"]["multi_layer_transformer.0.WQi.2.weight"]



X = A.numpy()

Y = B.numpy()

Z = C.numpy()

W = (X.transpose()).dot(Y)

U = (X.transpose()).dot(Z)

V = (Y.transpose()).dot(Z)

mat = plt.matshow(X)

#plt.matshow(Y)

#plt.matshow(Z)

print(linalg.norm(W, 'nuc'))

print(linalg.norm(U, 'nuc'))

print(linalg.norm(V, 'nuc'))



# Agregar la barra de color
#plt.colorbar(mat)

#plt.show()

