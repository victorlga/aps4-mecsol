import numpy as np
from node import Node


class Element:
    element_number: int
    node_1: Node
    node_2: Node
    area: float
    youngs_module: float
    length: float
    angle: float
    stiffness_matrix: np.ndarray

    def __init__(self, element_number: int, node_1: Node, node_2: Node, area: float, youngs_module: float) -> None:
        self.element_number = element_number
        self.node_1 = node_1
        self.node_2 = node_2
        self.area = area
        self.youngs_module = youngs_module
        self.length = self.get_length()
        self.angle = self.get_angle()
        self.stiffness_matrix = self.get_stiffness_matrix()

    def get_length(self) -> float:
        return np.sqrt((self.node_2.node_x - self.node_1.node_x)**2 + (self.node_2.node_y - self.node_1.node_y)**2)

    def get_angle(self) -> float:
        return np.arctan2(self.node_2.node_y - self.node_1.node_y, self.node_2.node_x - self.node_1.node_x)

    # Matriz de rigidez porém em inglês
    def get_stiffness_matrix(self) -> np.ndarray:
        c = np.cos(self.angle)
        s = np.sin(self.angle)
        k = np.array([
            [c**2, c*s, -c**2, -c*s],
            [c*s, s**2, -c*s, -s**2],
            [-c**2, -c*s, c**2, c*s],
            [-c*s, -s**2, c*s, s**2]
        ])
        k *= self.youngs_module * self.area / self.length
        return k

    def get_deformation(self) -> float:
        c = np.cos(self.angle)
        s = np.sin(self.angle)
        m = np.array([-c, -s, c, s])
        u = np.array([self.node_1.displacement_x, self.node_1.displacement_y, self.node_2.displacement_x, self.node_2.displacement_y])
        return np.dot(m, u)/self.length
