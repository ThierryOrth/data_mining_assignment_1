import numpy as np
from data_loader import ATTRIBUTES, CREDIT_DATA
from tree_structures import Node, Tree
import collections


def impurity(y:np.array):
    """Computes impurity using the Gini index as impurity function."""
    total = len(y)
    p_zeros = len(y[y[:] == 0])/total
    p_ones = len(y[y[:]==1])/total
    return p_zeros*p_ones

def bestsplit(x:np.array, y:np.array)->float:
    """Finds the best split from a range of candidate splits
        using the Gini index as impurity function."""

    lowest_imp = 1 
    x_sorted = np.sort(np.unique(x))
    candidate_splits = (x_sorted[0:-1]+x_sorted[1:len(x)])/2 
    best_split = None # IF THIS VALUE IS RETURNED, THEN WE KNOW THAT THERE EXISTS NO POSSIBLE SPLIT

    for split in candidate_splits:
        left_child = y[x <= split]
        right_child = y[split < x]

        i_left = impurity(left_child)
        i_right = impurity(right_child)
        pi_left = len(left_child)/len(x)
        pi_right = len(right_child)/len(x)

        imp = (pi_left * i_left) + (pi_right * i_right)

        if imp < lowest_imp:
            lowest_imp = imp
            best_split = split

    return best_split

def tree_grow(x:np.array, y:np.array, nmin:int, minleaf:int, nfeat:int)->Tree:
    """Starts with an initial node, extends it and returns a tree consisting of
        the initial node as root."""

    root_node = Node()
    extend_node(root_node, x, y, nmin, minleaf, nfeat)
    return Tree(root_node)

def extend_node(node:Node, x:np.array, y:np.array, nmin:int, minleaf:int, nfeat:int):
    """Given a current node, checks whether it can be extended. If not, the node becomes
        a leaf node. If so, then we split the node into two child nodes, which are fed into
        the same function by recursion."""

    _, num_of_obs = x.shape
    feature_index = np.random.choice(np.arange(0,num_of_obs,1))
    feature_values = x[:, feature_index]
    best_split = bestsplit(feature_values, y)
    majority_class = collections.Counter(y).most_common(1)[0][0]

    # MAKE A LEAF NODE IF THERE EXISTS NO SPLIT, THE NUMBERS OF 
    # OBSERVATIONS IS TOO LOW OR IMPURITY EQUALS ZERO
    if not best_split or num_of_obs<nmin or impurity(y)==0: 
        node.is_leaf = True
        node.feature_value = majority_class
        return

    left = feature_values[feature_values <= best_split]
    right = feature_values[best_split < feature_values]

    left_labels = y[feature_values <= best_split]
    right_labels = y[best_split < feature_values]

    print(f"\n left : {left} \n right : {right} \n left_labels : {left_labels} \
                 \n right_labels : {right_labels} \n feature_index : {feature_index} \n split : {best_split} \n")

    # MAKE A LEAF NODE IF THE NUMBER OF OBSERVATIONS 
    # FOR EITHER CHILD NODE IS TOO LOW

    if len(left)<minleaf or len(right)<minleaf:
        node.is_leaf = True
        node.feature_value = majority_class
        return

    node.feature_value = feature_index
    node.split_threshold = best_split

    node.left_child = Node()
    node.right_child = Node()
    extend_node(node.left_child, x[feature_values<=best_split], left_labels, nmin, minleaf, nfeat)
    extend_node(node.right_child, x[best_split<feature_values], right_labels, nmin, minleaf, nfeat)
        
def tree_pred(x:np.array, tr:Tree) -> float:
    """"Traverses the tree by comparing feature value with split threshold until it finds a leaf. If that leaf is reached, 
        we predict the majority vote."""
    current = tr.root 

    while not current.is_leaf: 
        feature_value = int(current.feature_value)
        if x[feature_value] <= current.split_threshold:
            current = current.left_child
            print(f"LEFT CHILD : {x[feature_value], current.split_threshold}")
        else:
            current = current.right_child
            print(f"RIGHT CHILD : {x[feature_value], current.split_threshold}")
    
    return current.feature_value

def tree_grow_b(x:np.array, y:np.array, nmin:int, minleaf:int, nfeat:int, m:int)->list:
    trees = []
    for i in range(m):
        pass
    return trees

def tree_pred_b(x:np.array, trs):
    pass

if __name__ == "__main__":
    #bestsplit(CREDIT_DATA[:,3],CREDIT_DATA[:,5])

    X = CREDIT_DATA[:,:5] 
    Y = CREDIT_DATA[:,5]
    tree = tree_grow(x=X, y=Y, nmin = 1, minleaf = 1, nfeat = 1)
    pred = tree_pred(x=X[3],tr=tree)

    print(f"X : {X[3]}\n \
            PRED : {pred} \n")

    tree.visualize_tree_2(tree.root)

    
   
  