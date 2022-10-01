import collections
import numpy as np
import sys
from data_loader import credit_data, pima_data
from tree_structures import Node, Tree

def gini_index(y:np.array) -> float:
    """Computes the Gini index as impurity function."""
    left = y[y[:]==1].size #Amount of observations in the left child
    right = y[y[:]==0].size #Amount of observations in the right child
    total = y.size #Amount of observations in the parent node (where the split is adjusted)
    ##Calculate the Gini index
    gini = (left/total)*(right/total)
    return gini

def bestsplit(x:np.array, y:np.array):
    """Finds the best split from a range of candidate splits
            using the Gini index as impurity function."""
    lowest_imp = 0
    x_sorted = np.sort(np.unique(x))
    candidate_splitpoints = (x_sorted[0:(x_sorted.size-1)] + x_sorted[1:x_sorted.size]) / 2
    best_split = None  # is no split exists, this value is used to create a leaf node
   
    # compute impurity on each splitpoint
    for split in candidate_splitpoints:
        left_child = y[x <= split]
        right_child = y[split < x]

        imp_parent = gini_index(y)
        pi_left = len(left_child) / len(x)
        pi_right = len(right_child) / len(x)
        imp_left = gini_index(left_child)
        imp_right = gini_index(right_child)
        
        imp = imp_parent - ((pi_left * imp_left) + (pi_right * imp_right))

        # update impurity and splitpoint if lower impurity is found
        if lowest_imp < imp:
            lowest_imp = imp
            best_split = split

    return best_split, lowest_imp

def tree_grow(x:np.array,y:np.array,nmin:int,minleaf:int,nfeat:int) -> Tree:
    """Starts with an initial node, extends it and returns a tree consisting of
            the initial node as root."""
    root_node = Node()
    extend_node(root_node, x, y, nmin, minleaf, nfeat)
    return Tree(root_node)

def extend_node(node:Node, x:np.array, y:np.array, nmin:int, minleaf:int, nfeat:int):
    """Given a current node, checks whether it can be extended. If not, the node becomes
            a leaf node. If so, then we split the node into two child nodes, which are fed into
            the same function by recursion."""
    n_of_obs, n_of_feat = x.shape
    majority_class = collections.Counter(y).most_common(1)[0][0]

    if nfeat<n_of_feat:
        indices = np.random.choice(np.arange(0,nfeat,1), nfeat, replace=False)
        x = x[:, indices]
        n_of_feat = nfeat
    
    # SELECT SPLIT ON FEATURE THAT YIELDS MINIMAL IMPURITY
    splits = [bestsplit(x[:,i],y) for i in range(n_of_feat)]
    feature_index = np.argmin([split[1] for split in splits])
    best_split, _ = splits[feature_index]
    feature_values = x[:, feature_index]


    # THIS DOES NOT PICK OUT A UNIQUE VALUE, AS THERE MIGHT BE DUPLICATE VALUES
    # best_reduction = np.max([split[1] for split in splits])
    # feature_index = [tup[1] for tup in splits].index(best_reduction)
    # feature_values = x[:, feature_index]
    #return the best split point for the best split of all features
    # best_split = splits[feature_index][0]

    # create a leaf node if gini impurity is zero, if there exists no split or 
    # if the number of observations in the node is too low
    if gini_index(y) == 0 or not best_split or n_of_obs < nmin:
        node.is_leaf = True
        node.feature_value = majority_class
        return

    left = feature_values[feature_values <= best_split]
    right = feature_values[best_split < feature_values]

    left_labels = y[feature_values <= best_split]
    right_labels = y[best_split < feature_values]


    # create a leaf node if the number of observations for leaf nodes is too low
    if len(left) < minleaf or len(right) < minleaf:
        node.is_leaf = True
        node.feature_value = majority_class
        return

    node.feature_value = feature_index
    node.split_threshold = best_split
    node.left_child, node.right_child = Node(), Node()
    
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
        else:
            current = current.right_child

    return current.feature_value

def tree_grow_b(x:np.array, y:np.array, nmin:int, minleaf:int, nfeat:int, m:int) -> list:
    """Grows a tree using bagging: retrieve a sample with replacement. This function can
        be used for random forests by the hyperparameter nfeat."""
    n_of_obs, _ = x.shape
    trees = []

    for i in range(m):
        root_node = Node()
        indices = np.random.choice(np.arange(0,n_of_obs,1), n_of_obs, replace=True)
        sampled_x, sampled_y = x[indices], y[indices]

        extend_node(root_node, sampled_x, sampled_y, nmin, minleaf, nfeat)
        trees.append(Tree(root_node))

    return trees

def tree_pred_b(x: np.array, trs: list) -> float:
    """Computes majority prediction given a feature array 
            and a collection of classification trees."""
    predictions = []
    
    for tr in trs:
        predictions.append(tree_pred(x, tr))
    
    majority_class = collections.Counter(predictions).most_common(1)[0][0]
    return majority_class

if __name__ == "__main__":
    X_credit = credit_data[:, :4]
    Y_credit = credit_data[:, 5]

    # n_of_obs, n_of_feat = PIMA_DATA.shape

    # X_pima = pima_data[:, :n_of_feat-1]
    # Y_pima = pima_data[:, n_of_feat-1]

    # tree = tree_grow(x=X_P, y=Y_P, nmin=20, minleaf=5, nfeat=n_of_feat)

    
