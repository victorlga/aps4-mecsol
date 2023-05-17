from node import Node
from element import Element
import numpy as np
from math import *

node_1 = Node(1, 0, 0, 0, 0, True, False)
node_2 = Node(2, 0, 0.4, 0, 0, True, True)
node_3 = Node(3, 0.3, 0.4, 150, -100, False, False)
nodes = [node_1, node_2, node_3]

E = 2.10e11
A = 2e-4

element_1 = Element(node_1, node_2, A, E)
element_2 = Element(node_2, node_3, A, E)
element_3 = Element(node_1, node_3, A, E)

K1 = element_1.stiffness_matrix
K2 = element_2.stiffness_matrix
K3 = element_3.stiffness_matrix

ndof = 2 * len(nodes)
K = np.zeros((ndof, ndof))

# 2n + 1
K[np.ix_(np.array([0, 1,  2, 3]), np.array([0, 1, 2, 3]))] += K1
K[np.ix_(np.array([2, 3,  4, 5]), np.array([2, 3, 4, 5]))] += K2
K[np.ix_(np.array([0, 1,  4, 5]), np.array([0, 1, 4, 5]))] += K3

# Restrictions
# 0 é o índice do GDL

# No 1 tem restrição em X
K[0,:] = 0
K[:,0] = 0
K[0, 0] = 1

# Nó 2 tem em restrição em X
K[2, :] = 0
K[:, 2] = 0
K[2, 2] = 1

# Nó 2 tem restrição em Y
K[3, :] = 0
K[:, 3] = 0
K[3, 3] = 1

# Matriz de forças
F = np.zeros([1, ndof])

F[0, 4] = 150
F[0, 5] = -100

U = np.linalg.solve(K, F.T)

print(U)


