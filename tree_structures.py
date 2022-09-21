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

    # FUNCTION BELOW DOESN'T WORK AT THE MOMENT
        
    # def visualize_tree(self): 
    #     node_names = list(string.ascii_uppercase)
    #     current = self.root
    #     i = 0
        
    #     def extend_nodes(current, previous=None):
    #         nonlocal i
    #         i+=1

    #         if not current:
    #             tree_graph.node(node_names[i], "leaf")
    #             return
            
    #         tree_graph.node(node_names[i], ATTRIBUTES[current.feature_value])

    #         if previous:
    #             tree_graph.edges([node_names[i-1]+node_names[i]])

    #         extend_nodes(current.left_child, current)
    #         extend_nodes(current.right_child, current)
        
    #     tree_graph = graphviz.Digraph("Classification Tree")
    #     extend_nodes(current)

    #     tree_graph.render(directory=os.getcwd(), view=True)  
    #     print(tree_graph.source)
        

    def visualize_tree_2(self, current):
        if current:
            print(f"feature : {ATTRIBUTES[current.feature_value]}  \
                                \n\t feature value: {current.feature_value} \
                                \n\t\t left child node: {ATTRIBUTES[current.left_child.feature_value] if current.left_child else None} \
                                \n\t\t right child node: {ATTRIBUTES[current.right_child.feature_value] if current.right_child else None} \
                                \n\t\t\t threshold : {current.split_threshold} \
                                \n\t\t\t\t leaf node? {current.is_leaf}")
                                
            self.visualize_tree_2(current.left_child)
            self.visualize_tree_2(current.right_child)


    