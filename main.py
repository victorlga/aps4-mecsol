from node import Node
from element import Element
import numpy as np
from math import *
from funcoesTermosol import *

entradaNome = 'entrada.xlsx'

[nn,N,nm,Inc,nc,F,nr,R] = importa(entradaNome)

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

'''
for node in nodes:
    print(f'Node {node.node_number}: ({node.node_x}, {node.node_y}), Fx = {node.load_x}, Fy = {node.load_y}, Rx = {node.restriction_x}, Ry = {node.restriction_y}')
'''

elements = []
for i in range(nm):
    node_1 = nodes[int(Inc[i, 0]) - 1]
    node_2 = nodes[int(Inc[i, 1]) - 1]
    A = Inc[i, 3]
    E = Inc[i, 2]
    element = Element(i, node_1, node_2, A, E)
    elements.append(element)

'''
for element in elements:
    print(f'Element {element.element_number}: ({element.node_1.node_number}, {element.node_2.node_number}), A = {element.area}, E = {element.youngs_module}, L = {element.length}, angle = {degrees(element.angle)}')
'''

ndof = 2 * nn
K = np.zeros((ndof, ndof))

for element in elements:
    K[np.ix_([element.node_1.dof_x_index, element.node_1.dof_y_index, element.node_2.dof_x_index, element.node_2.dof_y_index], [element.node_1.dof_x_index, element.node_1.dof_y_index, element.node_2.dof_x_index, element.node_2.dof_y_index])] += element.stiffness_matrix

K_with_restriction = K.copy()

for node in nodes:
    if node.restriction_x == 1:
        K_with_restriction[node.dof_x_index, :] = 0
        K_with_restriction[:, node.dof_x_index] = 0
        K_with_restriction[node.dof_x_index, node.dof_x_index] = 1
    if node.restriction_y == 1:
        K_with_restriction[node.dof_y_index, :] = 0
        K_with_restriction[:, node.dof_y_index] = 0
        K_with_restriction[node.dof_y_index, node.dof_y_index] = 1

F = np.reshape(F, (1, -1))

x0 = np.zeros(ndof)
eps = 1e-6
max_iter = 10000
U = gauss_seidel(K_with_restriction, F.T, x0, eps, max_iter)
U = np.reshape(U, (ndof, 1))

for node in nodes:
    node.displacement_x = U[node.dof_x_index, 0]
    node.displacement_y = U[node.dof_y_index, 0]
    #print(f'Node {node.node_number}: ({node.displacement_x}, {node.displacement_y})')

# Reações
print("Reações de apoio [N]:")
reactions = np.zeros((ndof, 1))
Ft = K @ U

# Reações em cada nó
for node in nodes:  
    if node.restriction_x == 1:
        reactions[node.dof_x_index, 0] = Ft[node.dof_x_index, 0]
    if node.restriction_y == 1:
        reactions[node.dof_y_index, 0] = Ft[node.dof_y_index, 0]

# remove zeros desnecessários
reactions = reactions[reactions != 0].reshape(-1, 1)
print(reactions)
print('\n')

print("Deslocamentos [m]:")
print(U)
print('\n')


# Deformações
print("Deformações []:")
deformations = np.zeros((nm, 1))
for element in elements:
    #print(f'Element {element.element_number}: {element.get_deformation()}')
    deformations[element.element_number, 0] = element.get_deformation()

print(deformations)
print('\n')


# Forças internas
print("Forças internas [N]:")
internal_forces = np.zeros((nm, 1))
for element in elements:
    #print(f'Element {element.element_number}: {element.get_internal_force()}')
    internal_forces[element.element_number, 0] = element.get_internal_force()

print(internal_forces)
print('\n')


# Tensões
print("Tensões internas [Pa]:")
stresses = np.zeros((nm, 1))
for element in elements:
    #print(f'Element {element.element_number}: {element.get_internal_stress()}')
    stresses[element.element_number, 0] = element.get_internal_stress()

print(stresses)


