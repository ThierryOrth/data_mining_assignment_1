import collections
import numpy as np
from data_loader import credit_data, pima_data
from tree_structures import Node, Tree

def impurity(class_obs:np.array) -> float:
    """Computes the Gini index of an array of 
       class observations."""
    total = class_obs.size 
    pos_class = class_obs[class_obs[:]==1].size 
    neg_class = class_obs[class_obs[:]==0].size
    return (pos_class/total)*(neg_class/total)

def bestsplit(feature_obs:np.array, class_obs:np.array) -> tuple:
    """Given the Gini index as impurity function, find the 
       split that minimizes impurity."""
    best_imp = 1
    best_split = None 
    sorted_feat = np.sort(np.unique(feature_obs))
    candidate_splitpoints = (sorted_feat[0:(sorted_feat.size-1)] + sorted_feat[1:sorted_feat.size]) / 2
   
    # COMPUTE GINI IMPURITY FOR EACH SPLIT
    for split in candidate_splitpoints:
        left_child = class_obs[feature_obs <= split]
        right_child = class_obs[split < feature_obs]

        pi_left = len(left_child) / len(feature_obs)
        pi_right = len(right_child) / len(feature_obs)

        imp = impurity(class_obs) - ((pi_left * impurity(left_child)) + (pi_right * impurity(right_child)))

        # SELECT A SPLIT IF IT REDUCES IMPURITY
        if imp < best_imp:
            best_imp = imp
            best_split = split

    return best_split, best_imp

def tree_grow(feature_obs:np.array, class_obs:np.array, nmin:int, minleaf:int, nfeat:int) -> Tree:
    """Starts with an initial node, extends it and returns a tree consisting of
            the initial node as root."""
    root_node = Node()
    extend_node(root_node, feature_obs, class_obs, nmin, minleaf, nfeat)
    return Tree(root_node)

def extend_node(node:Node, feature_obs:np.array, class_obs:np.array, nmin:int, minleaf:int, nfeat:int):
    """Given a current node, checks whether it can be extended. If not, the node becomes
            a leaf node. If so, then we split the node into two child nodes, which are fed into
            the same function by recursion."""
    n_of_obs, n_of_feat = feature_obs.shape
    majority_class = collections.Counter(class_obs).most_common(1)[0][0]

    ### SOMETHING GOES WRONG IN RANDOM FOREST PART ###
    # SELECT A SUBSET OF FEATURES 
    if nfeat<n_of_feat:
        #print(feature_obs.shape)
        indices = np.random.choice(np.arange(0,nfeat,1), nfeat, replace=False)
        feature_obs = feature_obs[:,indices]
        #print(feature_obs.shape)
        n_of_feat = nfeat 
    
    # SPLIT ON FEATURE THAT MINIMIZES IMPURITY
    splits = [bestsplit(feature_obs[:,i], class_obs) for i in range(n_of_feat)]

    # MAKE LEAF NODE IF IMPURITY REACHES ZERO, NO SPLIT IS FOUND 
    # OR THE NUMBER OF OBSERVATIONS IS TOO LOW
    if impurity(class_obs)<=0 or n_of_obs<nmin or not splits:
        set_node_to_leaf(node, majority_class)
        return

    feature_index = np.argmin([split[1] for split in splits])
    best_split, _ = splits[feature_index]
    feature_values = feature_obs[:, feature_index]

    l_feat_obs = feature_obs[feature_obs[:, feature_index] <= best_split]
    r_feat_obs = feature_obs[feature_obs[:, feature_index] > best_split]

    # MAKE LEAF NODE IF NUMBER OF OBSERVATIONS IS TOO LOW
    l_size, _ = l_feat_obs.shape
    r_size, _ = r_feat_obs.shape

    if l_size<minleaf or r_size<minleaf:
        set_node_to_leaf(node, majority_class)
        return

    l_class_obs = class_obs[feature_values <= best_split]
    r_class_obs = class_obs[feature_values > best_split]

    node.feature_value = feature_index
    node.split_threshold = best_split
    node.left_child, node.right_child = Node(), Node()
    
    # RECURSE OVER LEFT AND RIGHT NODE CHILD
    extend_node(node.left_child, l_feat_obs, l_class_obs, nmin, minleaf, n_of_feat)
    extend_node(node.right_child, r_feat_obs, r_class_obs, nmin, minleaf, n_of_feat)

def set_node_to_leaf(node, feature_value):
    node.is_leaf = True
    node.feature_value = feature_value

def tree_pred(single_feature_obs: np.array, tr: Tree) -> float:
    """"Traverses the tree by comparing feature value with split threshold until it finds a leaf. If that leaf is reached,
        we predict the majority vote."""
    current = tr.root

    # LOOP UNTIL WE REACH A LEAF NODE
    while not current.is_leaf:
        feature_value = int(current.feature_value)

        if single_feature_obs[feature_value] <= current.split_threshold:
            current = current.left_child
        else:
            current = current.right_child

    # RETURN PREDICTION OF LEAF NODE, WHICH IS THE MAJORITY CLASS
    return current.feature_value

def tree_grow_b(feature_obs:np.array, class_obs:np.array, nmin:int, minleaf:int, nfeat:int, m:int) -> list:
    """Grows a bagged tree by sampling with replacement from the feature observations. 
       The hyperparameter nfeat can be used to train a random forest."""

    n_of_obs, _ = feature_obs.shape
    trees = []

    for i in range(m):
        root_node = Node()
        indices = np.random.choice(np.arange(0,n_of_obs,1), n_of_obs, replace=True)
        feat_sample, class_sample = feature_obs[indices,:], class_obs[indices]

        extend_node(root_node, feat_sample, class_sample, nmin, minleaf, nfeat)
        trees.append(Tree(root_node))

    return trees

def tree_pred_b(feature_obs: np.array, trs: list) -> float:
    """Computes majority prediction given a feature array 
            and a collection of classification trees."""
    predictions = []
    
    for tree in trs:
        predictions.append(tree_pred(feature_obs, tree))
    
    majority_class = collections.Counter(predictions).most_common(1)[0][0]
    return majority_class