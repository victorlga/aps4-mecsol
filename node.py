import numpy as np

class Node:
    node_number: int
    node_x: float
    node_y: float
    load_x: float
    load_y: float
    restriction_x: int
    restriction_y: int
    dof_x_index: int
    dof_y_index: int

    def __init__(self, node_number: int, node_x, node_y) -> None:
        self.node_number = node_number
        self.node_x = node_x
        self.node_y = node_y


    