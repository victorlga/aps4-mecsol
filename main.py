from node import Node
from element import Element
import numpy as np
from math import *
from funcoesTermosol import *
import xlrd

entradaNome = 'entrada.xlsx'

[nn,N,nm,Inc,nc,F,nr,R] = importa(entradaNome)

print('nn = ', nn)
print('N = ', N)
print('nm = ', nm)
print('Inc = ', Inc)
print('nc = ', nc)
print('F = ', F)
print('nr = ', nr)
print('R = ', R)
#plota(N, Inc)
nodes = []
for i in range(nn):
    node = Node(i, N[0, i], N[1, i])
    node.dof_x_index = 2*i
    node.dof_y_index = 2*i + 1
    node.load_x = F[2*i, 0]
    node.load_y = F[2*i + 1, 0]
    node.restriction_x = R[2*i, 0]
    node.restriction_y = R[2*i + 1, 0]
    nodes.append(node)

for node in nodes:
    print(f'Node {node.node_number}: ({node.node_x}, {node.node_y}), Fx = {node.load_x}, Fy = {node.load_y}, Rx = {node.restriction_x}, Ry = {node.restriction_y}')

elements = []
for i in range(nm):
    node_1 = nodes[int(Inc[i, 0]) - 1]
    node_2 = nodes[int(Inc[i, 1]) - 1]
    A = Inc[i, 3]
    E = Inc[i, 2]
    element = Element(i, node_1, node_2, A, E)
    elements.append(element)

for element in elements:
    print(f'Element {element.element_number}: ({element.node_1.node_number}, {element.node_2.node_number}), A = {element.area}, E = {element.youngs_module}, L = {element.length}, angle = {degrees(element.angle)}')


ndof = 2 * nn
K = np.zeros((ndof, ndof))

for element in elements:
    K[np.ix_([element.node_1.dof_x_index, element.node_1.dof_y_index, element.node_2.dof_x_index, element.node_2.dof_y_index], [element.node_1.dof_x_index, element.node_1.dof_y_index, element.node_2.dof_x_index, element.node_2.dof_y_index])] += element.stiffness_matrix

# Restrictions
# 0 é o índice do GDL

# No 1 tem restrição em X
for node in nodes:
    if node.restriction_x == 1:
        K[node.dof_x_index, :] = 0
        K[:, node.dof_x_index] = 0
        K[node.dof_x_index, node.dof_x_index] = 1
    if node.restriction_y == 1:
        K[node.dof_y_index, :] = 0
        K[:, node.dof_y_index] = 0
        K[node.dof_y_index, node.dof_y_index] = 1

F = np.reshape(F, (1, -1))

U = np.linalg.solve(K, F.T)

print(U)

