from data_loader import ATTRIBUTES

class Node:
    def __init__(self):
        self.feature_value = None
        self.split_threshold = None
        self.left_child = None
        self.right_child = None
        self.is_leaf = False

    def set_name(self, name):
        self.name = name

class Tree:
    def __init__(self, root):
        if isinstance(root,Node):
            self.root = root
        else:
            raise ValueError("")
        
    def node_repr(self, current, node_index=1):
        if current:
            print(f"feature : {ATTRIBUTES[current.feature_value]}  \
                                \n\t feature value: {current.feature_value} \
                                \n\t\t left child node: {ATTRIBUTES[current.left_child.feature_value] if current.left_child else None} \
                                \n\t\t right child node: {ATTRIBUTES[current.right_child.feature_value] if current.right_child else None} \
                                \n\t\t\t threshold : {current.split_threshold} \
                                \n\t\t\t\t leaf node? {current.is_leaf}")
                                
            self.node_repr(current.left_child, node_index+1)
            self.node_repr(current.right_child, node_index+2)


