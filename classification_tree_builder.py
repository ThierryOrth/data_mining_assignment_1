import numpy as np
from data_loader import CREDIT_DATA
from tree_structures import Node, Tree
import collections

ATTRIBUTES = {0:"age", 1:"married", 2:"house", 3:"income", 4:"gender", 5:"class"}

def impurity(y):
    total = len(y)
    p_zeros = len(y[y[:] == 0])/total
    p_ones = len(y[y[:]==1])/total
    return p_zeros*p_ones

def bestsplit(x,y):
    lowest_imp = 1 
    x_sorted = np.sort(np.unique(x))
    candidate_splits = (x_sorted[0:-1]+x_sorted[1:len(x)])/2 # WHAT TO DO IF THERE ARE NO CANDIDATE SPLITS?

    #print(f"X : {x}, Y : {y}, SORTED X : {x_sorted}, CANDIDATE SPLITS : {candidate_splits}")

    best_split = candidate_splits[0]

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

def tree_grow(x, y, nmin, minleaf, nfeat):
    node_index = 1
    root_node = Node(node_index=1)
    extend_node(root_node, x, y, nmin, minleaf, nfeat)
    return Tree(root_node)

def extend_node(node, x, y, nmin, minleaf, nfeat):
    _, num_of_obs = x.shape
        
    if nmin <= num_of_obs and 0<impurity(y):
        feature_index = np.random.choice(np.arange(0,num_of_obs,1))
        feature_values = x[:, feature_index]
        best_split = bestsplit(feature_values, y)

        left = feature_values[feature_values <= best_split]
        right = feature_values[best_split < feature_values]

        left_labels = y[feature_values <= best_split]
        right_labels = y[best_split < feature_values]

        node.feature_value = feature_index
        node.split_threshold = best_split

        node.left_child = Node(node_index=0)
        node.right_child = Node(node_index=0)

        print(f"\n left : {left} \n right : {right} \n left_labels : {left_labels} \n right_labels : {right_labels} \n feature_index : {feature_index} \n split : {best_split} \n")

        if minleaf<=len(left) and minleaf<=len(right):
            extend_node(node.left_child, x[feature_values<=best_split], left_labels, nmin, minleaf, nfeat)
            extend_node(node.right_child, x[best_split<feature_values], right_labels, nmin, minleaf, nfeat)

def tree_pred(x, tr):
    current = tr.root # CONCEPT

    while not current.is_leaf:
        print(x[current.feature_value, current.split_threshold])
        print(x[current.feature_value], current.split_threshold)
        if x[current.feature_value] <= current.split_threshold:
            current = current.left_child
        else:
            current = current.right_child
    
    return current.feature_value

def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
    trees = []
    for i in range(m):
        pass
    return trees

def tree_pred_b(x, trs):
    pass



if __name__ == "__main__":
    #bestsplit(CREDIT_DATA[:,3],CREDIT_DATA[:,5])

    X = CREDIT_DATA[:,:5] 
    Y = CREDIT_DATA[:,5]
    tree = tree_grow(x=X, y=Y, nmin = 1, minleaf = 1, nfeat = 1)
    tree.node_repr(current=tree.root) #TODO: ADD INDICES
    
    tree_pred(x=X[3],tr=tree)
   
  