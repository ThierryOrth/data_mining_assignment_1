from data_loader import ATTRIBUTES
import graphviz
import string
import os

class Node:
    """Constructs a node with a set of internal properties useful 
        for classification trees: the feature index in the data matrix,
        the split threshold, the left and right child nodes and whether 
        it is a leaf or not."""
    def __init__(self):
        self.feature_index = None # COLUMN IN THE MATRIX FOR NON-TERMINAL NODES, MAJORITY VOTE PREDICTION FOR TERMINAL NODES 
        self.split_threshold = None
        self.left_child = None
        self.right_child = None
        self.is_leaf = False

class Tree:
    """Constructs a tree by taking an initial node
        as root."""
    def __init__(self, root:Node):
        if isinstance(root,Node):
            self.root = root
        else:
            raise ValueError("")