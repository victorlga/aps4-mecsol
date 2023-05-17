import numpy as np

class Node:
    node_number: int
    node_x: float
    node_y: float
    load_x: float
    load_y: float
    restriction_x: bool
    restriction_y: bool
    dof_x_index: int
    dof_y_index: int

    def __init__(self, node_number: int, node_x, node_y, load_x, load_y, restriction_x: bool, restriction_y: bool) -> None:
        self.node_number = node_number
        self.node_x = node_x
        self.node_y = node_y
        self.load_x = load_x
        self.load_y = load_y
        self.restriction_x = restriction_x
        self.restriction_y = restriction_y

    