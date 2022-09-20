class Node:
    def __init__(self, node_index):
        self.node_index = node_index
        self.feature_value = None
        self.split_threshold = None
        self.left_child = None
        self.right_child = None
        self.is_leaf = False

class Tree:
    def __init__(self, root):
        if isinstance(root,Node):
            self.root = root
        else:
            raise ValueError("")
        
    def node_repr(self, current):
        if current:
            print(f"index of node : {current.node_index} \n\t feature : {current.feature_value} \
                                \n\t\t child node indices : {current.left_child.node_index if current.left_child else None, current.right_child.node_index if current.right_child else None} \
                                \n\t\t\t threshold : {current.split_threshold} \
                                \n\t\t\t\t leaf node? {current.is_leaf}")
                                # 
                                
            self.node_repr(current.left_child)
            self.node_repr(current.right_child)


