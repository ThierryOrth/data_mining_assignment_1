import numpy as np
import pandas as pd
import collections, os, graphviz, time, sys
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

class Node:
    """Constructs a node with a set of internal properties useful 
        for classification trees: the feature index in the data matrix,
        the split threshold, the left and right child nodes and whether 
        it is a leaf or not."""
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.feature = None 
        self.threshold = None
        self.majority_label = None
        self.is_leaf = False

class Tree:
    """Constructs a tree by taking an initial node
        as root."""
    def __init__(self, root:Node):
        if isinstance(root,Node):
            self.root = root
        else:
            raise ValueError("A Tree object requires a Node object as root!")

def tree_grow(feature_obs:np.array, class_obs:np.array, nmin:int, minleaf:int, nfeat:int) -> Tree:
    """Starts with an initial node, extends it and returns a tree consisting of
       the initial node as root."""
    root_node = Node()
    extend_node(root_node, feature_obs, class_obs, nmin, minleaf, nfeat)
    return Tree(root_node)

def tree_pred(feature_obs: np.array, tr: Tree) -> float:
    """"Traverses the tree by comparing feature value with split threshold until it finds a leaf. If that leaf is reached,
        we predict the majority_label vote."""

    # never a negative classification value assigned in prediction
    n_of_obs, _ = feature_obs.shape
    pred = np.zeros(n_of_obs)

    # LOOP UNTIL WE REACH A LEAF NODE

    for obs_index in range(n_of_obs):
        current_node = tr.root
        while not current_node.is_leaf:
            feature_index = current_node.feature

            if feature_obs[obs_index, feature_index] <= current_node.threshold:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child

        # APPEND MAJORITY LABEL OF LEAF NODE
        pred[obs_index] = current_node.majority_label
    return pred

def tree_grow_b(feature_obs:np.array, class_obs:np.array, nmin:int, minleaf:int, nfeat:int, m:int) -> list:
    """Grows a bagged tree by sampling with replacement from the feature observations. 
       The hyperparameter nfeat can be used to train a random forest."""
    n_of_obs, _ = feature_obs.shape
    trees = []

    for i in range(m):
        print(f"===EPOCH {i}===")
        root_node = Node()
        indices = np.random.choice(np.arange(0,n_of_obs,1), n_of_obs, replace=True)
        feat_sample, class_sample = feature_obs[indices,:], class_obs[indices]

        extend_node(root_node, feat_sample, class_sample, nmin, minleaf, nfeat)
        trees.append(Tree(root_node))

    return trees

def tree_pred_b(feature_obs: np.array, trs: list) -> float:
    """Computes majority_label prediction given a feature array 
       and a collection of classification trees."""
    row_index = len(trs)
    col_index, _ = feature_obs.shape 
    predictions = np.zeros((row_index,col_index))
    
    for index, tree in enumerate(trs):
        predictions[index, :] = tree_pred(feature_obs,tree)    

    majority_labels = [collections.Counter(predictions[:,i]).most_common(1)[0][0] for i in range(col_index)]

    return majority_labels

# def extend_node(node:Node, x:np.array, y:np.array, nmin:int, minleaf:int, nfeat:int):
#     """Given a current node, checks whether it can be extended. If not, the node becomes
#                a leaf node. If so, then we split the node into two child nodes, which are fed into
#                the same function by recursion."""
#     feat_to_split = feat_to_imp = dict()
#     n_of_obs, n_of_feat = x.shape
#     indices = np.arange(0, n_of_feat, 1)
#     node.majority_label = collections.Counter(y).most_common(1)[0][0]

#     if n_of_obs < nmin or impurity(y) == 0:
#         node.is_leaf=True
#         return

#     if nfeat<n_of_feat:
#         indices = np.random.choice(np.arange(0,nfeat,1), nfeat, replace=False)
#         n_of_feat = nfeat

#     # MAP FEATURE INDICES TO SPLITS AND IMPURITY SCORES, GET FEATURE WITH MAXIMAL SPLIT
#     for index in indices:
#         split, imp = bestsplit(x[:, index], y, minleaf)  # best split for feature f in n_of_feat

#         if split:
#             feat_to_split[index] = split
#             feat_to_imp[index] = imp

#     if not feat_to_split:
#         node.is_leaf = True
#         return

#     feature_index = min(feat_to_imp, key = feat_to_imp.get)    
#     threshold = feat_to_split[feature_index]
    
#     # assign each row to either the left or right child based on the threshold
#     l_feat_obs = x[x[:, feature_index] <= threshold]
#     r_feat_obs = x[x[:, feature_index]> threshold]
#     l_class_obs = y[x[:, feature_index] <= threshold]
#     r_class_obs = y[x[:, feature_index] > threshold]

#     node.feature = feature_index
#     node.threshold = threshold
#     node.left_child = node.right_child = Node()

#     extend_node(node.left_child, l_feat_obs, l_class_obs, nmin, minleaf, nfeat)
#     extend_node(node.right_child, r_feat_obs, r_class_obs, nmin, minleaf, nfeat)

def extend_node(node:Node, x:np.array, y:np.array, nmin:int, minleaf:int, nfeat:int):
    """Given a current node, checks whether it can be extended. If not, the node becomes
               a leaf node. If so, then we split the node into two child nodes, which are fed into
               the same function by recursion."""
    feat_to_split = dict()
    feat_to_imp = dict()
    num_of_obs, n_of_feat = x.shape
    # get feature indices relative to nfeat parameter

    indices = np.random.choice(np.arange(0,nfeat,1), nfeat, replace=False) if nfeat<n_of_feat else np.arange(0, n_of_feat, 1)
    # set majority label for nodes
    node.majority_label = collections.Counter(y).most_common(1)[0][0]

    # construct mapping from feature values to split and impurity values
    for index in indices:
        split, imp = bestsplit(x[:,index],y,minleaf)
        
        if split:
            feat_to_split[index]=split
            feat_to_imp[index]=imp

    # make leaf node if there are no splits, the number of observations is too low
    # or impurity is zero
    if not feat_to_split or num_of_obs < nmin or impurity(y) == 0:
        node.is_leaf = True
        return

    # get feature value of best split, get best split and 
    feature_index = min(feat_to_imp, key=feat_to_imp.get)
    threshold = feat_to_split[feature_index]
    feature_values = x[:, feature_index]

    node.feature = feature_index
    node.threshold = threshold

    # check for all the best splits of all features which one
    # of these has the best impurity reduction


    # MAKE A LEAF NODE IF THERE EXISTS NO SPLIT, THE NUMBERS OF
    # OBSERVATIONS IS TOO LOW OR IMPURITY EQUALS ZERO

    # assign each row to either the left or right child based on its value
    left = x[x[:, feature_index] <= threshold]
    right = x[x[:, feature_index] > threshold]

    # print(f"x:{x}")
    # print(f"left:{left}")
    # print(f"right:{right}")

    left_labels = y[feature_values <= threshold]
    right_labels = y[feature_values > threshold]

    node.left_child = Node()
    node.right_child = Node()

    extend_node(node.left_child, left, left_labels, nmin, minleaf, nfeat)
    extend_node(node.right_child, right, right_labels, nmin, minleaf, nfeat)


def impurity(class_obs: np.array) -> float:
    """Computes the Gini index of an array of
       class observations."""
    total = class_obs.size
    pos_class = class_obs[class_obs == 1].size
    neg_class = class_obs[class_obs == 0].size
    return (pos_class / total) * (neg_class / total)


def bestsplit(x,y,minleaf):
    """Finds the best split from a range of candidate splits
            using the Gini index as impurity function."""
    best_imp = 1
    x_sorted = np.sort(np.unique(x))
    cand_splits = (x_sorted[0:(x_sorted.size - 1)] + x_sorted[1:x_sorted.size]) / 2
    best_split = None  # IF THIS VALUE IS RETURNED, THEN WE KNOW THAT THERE EXISTS NO POSSIBLE SPLIT

    ## for every split option calculate the gini index.
    for split in cand_splits:
        left_child = y[x <= split]
        right_child = y[x > split]

        # MAKE A LEAF NODE IF THE NUMBER OF OBSERVATIONS
        # FOR EITHER CHILD NODE IS TOO LOW
        if minleaf <= len(left_child) and minleaf <= len(right_child):
            pi_left = len(left_child) / len(x)
            pi_right = len(right_child) / len(x) 
            imp = impurity(y) - ((pi_left * impurity(left_child)) + (pi_right * impurity(right_child)))

            if imp<best_imp:
                best_imp = imp
                best_split = split

    # return the best split option with the corresponding impurity reduction
    return best_split, best_imp

def load_data(dataset):
    dir = os.getcwd()

    if dataset=="credit":
        credit_data = np.genfromtxt(dir + "/credit.txt", delimiter=",", skip_header=True)
        return credit_data
    
    elif dataset=="pima":
        pima_data = np.genfromtxt(dir + "/pima.txt", delimiter=",", skip_header=False)
        return pima_data

    elif dataset=="bug":
        bug_dataframe_train = pd.read_csv("eclipse-metrics-packages-2.0.csv", delimiter=";")
        bug_dataframe_test = pd.read_csv("eclipse-metrics-packages-3.0.csv", delimiter=";")

        column_indices=[2]+[i for i in range(4, 44)]

        # PREPROCESSING STEP: SET BUG VALUES TO CATEGORICALS
        x_train = bug_dataframe_train.iloc[:,column_indices].to_numpy()
        x_train[:,0] = np.where(0<x_train[:,0], 1.0, 0.0)

        x_test = bug_dataframe_train.iloc[:,3].to_numpy()
        x_test = np.where(0<x_test, 1.0, 0.0)

        y_train = bug_dataframe_test.iloc[:,column_indices].to_numpy()
        y_train[:,0] = np.where(0<y_train[:,0], 1.0, 0.0)

        y_test = bug_dataframe_test.iloc[:,3].to_numpy()
        y_test = np.where(0.0<y_test, 1.0, 0.0)

        return bug_dataframe_train.columns, x_train, x_test, y_train, y_test

def print_results(y_true,y_pred, name, runtime, verbose=False):
    scores = precision_recall_fscore_support(y_true,y_pred)
    string = f"MODEL : {name} \n \
                    CONFUSION MATRIX: {confusion_matrix(y_true,y_pred)} \n \
                        PRECISION: {scores[0]} \n \
                            RECALL: {scores[1]} \n \
                                ACCURACY SCORE: {accuracy_score(y_true,y_pred)} \n \
                                    RUNTIME IN SECONDS: {runtime}"
    if verbose:
        print(string)

    else:
        with open(f"results_{name}.txt", "w") as file:
            file.write(string)

def visualize_tree(nodes, edges, columns):
    #TODO
    digraph = graphviz.Digraph(comment="first two splits classification tree")

    for name, node in nodes.items():
        majority_label = node.majority_label
    
        if not node.is_leaf:
            digraph.node(name, f"feature: {columns[node.feature]} \n majority class: {majority_label}")
        else:
            digraph.node(name, f"leaf node \n majority class: {majority_label}")

    for parent, children in edges.items():
        left_child, right_child = children
        threshold = nodes[parent].threshold

        digraph.edge(parent, left_child, label=f"<={threshold}", len="1.00")
        digraph.edge(parent, right_child, label=f">{threshold}", len="1.00")

    digraph.render(directory='doctest-output', view=True)  

if __name__ == "__main__":
    np.random.seed(42)
    column_names, x_train, x_test, y_train, y_test = load_data(dataset="bug")

    start_time = time.time()
    tree = tree_grow(x_train, x_test, nmin=15, minleaf=5, nfeat=41)
    y_pred_n = tree_pred(y_train, tree)
    print_results(y_test, y_pred_n, name="regular_tree", runtime=time.time() - start_time,verbose=True)

    nodes = dict({"A":tree.root, "B":tree.root.left_child, "C":tree.root.right_child, \
            "D":tree.root.right_child.left_child,"E":tree.root.right_child.right_child})
    edges = dict({"A":("B","C"),"C":("D","E")})
    
    visualize_tree(nodes, edges, column_names)

    # A picture of the first two splits of the single tree (the split in the root
    # node, and the split in its left or right child). Consider the classification
    # rule that you get by assigning to the majority class in the three leaf nodes
    # of this heavily simplified tree. Discuss whether this classification rule
    # makes sense, given the meaning of the attributes.

    #visualize_tree(node_names= ,edge_list=, name_to_threshold=,)

    # REQUIREMENTS

    # PARENT NODE, CHILD NODES, CHILD NODES OF ONE CHILD NODE

    # PROPERTIES: MAJORITY CLASS, FEATURE THRESHOLD AND SPLIT

    start_time = time.time()
    tree_bagged = tree_grow_b(x_train, x_test, nmin=15, minleaf=5, nfeat=41, m=100)
    y_pred_b = tree_pred_b(y_train, tree_bagged)
    print_results(y_test,y_pred_b, name="bagged_tree", runtime=time.time() - start_time,verbose=True)

    start_time = time.time()
    random_forest = tree_grow_b(x_train, x_test, nmin=15, minleaf=5, nfeat=6, m=100)
    y_pred_f = tree_pred_b(y_train, random_forest)
    print_results(y_test, y_pred_f, name="random_forest", runtime=time.time() - start_time, verbose=True)
