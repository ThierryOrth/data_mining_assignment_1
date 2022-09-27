import numpy as np
from data_loader import ATTRIBUTES, CREDIT_DATA
from tree_structure import Node, Tree
import collections

def gini_index(y):
    """Computes the Gini index as impurity function."""
    left = y[y[:]==1].size #Amount of observations in the left child
    right = y[y[:]==0].size #Amount of observations in the right child
    total = y.size #Amount of observations in the parent node (where the split is adjusted)
    ##Calculate the Gini index
    gini = (left/total)*(right/total)
    return gini



def bestsplit(x,y):
    """Finds the best split from a range of candidate splits
            using the Gini index as impurity function."""
    highest_red = 0
    x_sorted = np.sort(np.unique(x))
    candidate_splitpoints = (x_sorted[0:(x_sorted.size-1)] + x_sorted[1:x_sorted.size]) / 2
    best_split = None  # IF THIS VALUE IS RETURNED, THEN WE KNOW THAT THERE EXISTS NO POSSIBLE SPLIT

    ## for every split option calculate the gini index per subgroup.
    for split in candidate_splitpoints:
        left_child = y[x <= split]
        right_child = y[split < x]

        imp_left = gini_index(left_child)
        imp_right = gini_index(right_child)
        pi_left = len(left_child) / len(x)
        pi_right = len(right_child) / len(x)
        imp_parent = gini_index(y)
        imp = imp_parent - ((pi_left * imp_left) + (pi_right * imp_right))

        if imp > highest_red:
            highest_red = imp
            best_split = split
    #return the best split option with the corresponding impurity reduction
    return best_split, highest_red

def tree_grow(x,y,nmin,minleaf,nfeat):
    """Starts with an initial node, extends it and returns a tree consisting of
            the initial node as root."""

    root_node = Node()
    extend_node(root_node, x, y, nmin, minleaf, nfeat)
    return Tree(root_node)

def extend_node(node:Node, x,y,nmin,minleaf,nfeat):
    """Given a current node, checks whether it can be extended. If not, the node becomes
            a leaf node. If so, then we split the node into two child nodes, which are fed into
            the same function by recursion."""
    splits = []
    num_of_obs, num_of_classes = x.shape

    for c in range(num_of_classes):
        split = bestsplit(x[:,c],y)
        splits.append(split)
    majority_class = collections.Counter(y).most_common(1)[0][0]
    print(majority_class)
    #check for all the best splits of all features which one
    # of these has the best impurity reduction
    best_red = np.max([b[1]for b in splits])
    #get the index for the splitpoint
    feature_index = [tup[1] for tup in splits].index(best_red)
    feature_values = x[:, feature_index]
    #return the best split point for the best split of all features
    best_split = splits[feature_index][0]

    # MAKE A LEAF NODE IF THERE EXISTS NO SPLIT, THE NUMBERS OF
    # OBSERVATIONS IS TOO LOW OR IMPURITY EQUALS ZERO
    if gini_index(y) == 0 or not best_split or num_of_obs < nmin:
        node.is_leaf = True
        node.feature_value = majority_class
        return

    left = feature_values[feature_values <= best_split]
    right = feature_values[best_split < feature_values]

    left_labels = y[feature_values <= best_split]
    right_labels = y[best_split < feature_values]

    #print(f"\n left : {left} \n right : {right} \n left_labels : {left_labels} \
     #               \n right_labels : {right_labels} \n feature_index : {feature_index} \n split : {best_split} \n")

    # MAKE A LEAF NODE IF THE NUMBER OF OBSERVATIONS
    # FOR EITHER CHILD NODE IS TOO LOW
    if len(left) < minleaf or len(right) < minleaf:
        node.is_leaf = True
        node.feature_value = majority_class
        return

    node.feature_value = feature_index
    node.split_threshold = best_split

    node.left_child = Node()
    node.right_child = Node()
    extend_node(node.left_child, x[feature_values <= best_split], left_labels, nmin, minleaf, nfeat)
    extend_node(node.right_child, x[best_split < feature_values], right_labels, nmin, minleaf, nfeat)


def tree_pred(x: np.array, tr: Tree) -> float:
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

if __name__ == "__main__":
    #bestsplit(CREDIT_DATA[:,3],CREDIT_DATA[:,5])

    X = CREDIT_DATA[:,:4]
    Y = CREDIT_DATA[:,5]
    tree = tree_grow(x=X, y=Y, nmin = 2, minleaf = 1, nfeat = 4)
    pred = tree_pred(x=X[3],tr=tree)

    """print(f"X : {X[3]}\n \
            PRED : {pred} \n")"""

    tree.visualize_tree_2(tree.root)
