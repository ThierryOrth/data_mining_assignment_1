class Node:
    def __init__(self):
        self.name = None
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
        
    def node_repr(self, current):
        if current:
            print(f"node name : {current.name} \n\t feature : {current.feature_value} \
                                \n\t\t child node indices : {current.left_child.name if current.left_child else None, current.right_child.name if current.right_child else None} \
                                \n\t\t\t threshold : {current.split_threshold} \
                                \n\t\t\t\t leaf node? {current.is_leaf}")
                                # 
                                
            self.node_repr(current.left_child)
            self.node_repr(current.right_child)


