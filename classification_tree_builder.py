import collections
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import time
import sys
import os


class Node:
    """Constructs a node with a set of internal properties useful 
        for classification trees: the feature index in the data matrix,
        the split threshold, the left and right child nodes and whether 
        it is a leaf or not."""
    def __init__(self):
        self.feature_value = None # COLUMN IN THE MATRIX FOR NON-TERMINAL NODES, majority_label VOTE PREDICTION FOR TERMINAL NODES 
        self.split_threshold = None
        self.left_child = None
        self.right_child = None
        self.is_leaf = False
        self.majority_label = None

class Tree:
    """Constructs a tree by taking an initial node
        as root."""
    def __init__(self, root:Node):
        if isinstance(root,Node):
            self.root = root
        else:
            raise ValueError("")

def tree_grow(feature_obs:np.array, class_obs:np.array, nmin:int, minleaf:int, nfeat:int) -> Tree:
    """Starts with an initial node, extends it and returns a tree consisting of
       the initial node as root."""
    root_node = Node()
    extend_node(root_node, feature_obs, class_obs, nmin, minleaf, nfeat)
    return Tree(root_node)

def tree_pred(feature_obs: np.array, tr: Tree) -> float:
    """"Traverses the tree by comparing feature value with split threshold until it finds a leaf. If that leaf is reached,
        we predict the majority_label vote."""
    pred=[]

    # LOOP UNTIL WE REACH A LEAF NODE
    for feature_row in feature_obs:
        current = tr.root
        while not current.is_leaf:
            feature_value = int(current.feature_value)

            if feature_row[feature_value] <= current.split_threshold:
                current = current.left_child
            else:
                current = current.right_child

        # APPEND MAJORITY LABEL OF LEAF NODE
        pred.append(current.majority_label)

    return pred

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
    """Computes majority_label prediction given a feature array 
       and a collection of classification trees."""
    row_index = len(trs)
    col_index, _ = feature_obs.shape 
    predictions = np.zeros((row_index,col_index))
    print("shape", predictions.shape)
    
    for index, tree in enumerate(trs):
        predictions[index, :] = tree_pred(feature_obs,tree)

    majority_label_labels = [collections.Counter(predictions[:,i]).most_common(1)[0][0] for i in range(col_index)]

    return majority_label_labels


def extend_node(node: Node, feature_obs: np.array, class_obs: np.array, nmin: int, minleaf: int, nfeat: int):
    """Given a current node, checks whether it can be extended. If not, the node becomes
       a leaf node. If so, then we split the node into two child nodes, which are fed into
       the same function by recursion."""
    n_of_obs, n_of_feat = feature_obs.shape
    majority_label = collections.Counter(class_obs).most_common(1)[0][0]
    node.majority_label = majority_label
    splits = []
    
    ### SOMETHING GOES WRONG IN RANDOM FOREST PART ###
    # SELECT A SUBSET OF FEATURES
    if nfeat < n_of_feat:
        indices = np.random.choice(np.arange(0, nfeat, 1), nfeat, replace=False)
        feature_obs = feature_obs[:, indices]
        n_of_feat = nfeat

        # MAP SPLITS TO FEATURES
    for feat_index in range(n_of_feat):
        split, imp = bestsplit(feature_obs[:, feat_index], class_obs,minleaf)
        if split:
            splits.append((split, imp))

    # MAKE LEAF NODE IF IMPURITY REACHES ZERO, NO SPLIT IS FOUND
    # OR THE NUMBER OF OBSERVATIONS IS TOO LOW
    if impurity(class_obs) <= 0 or n_of_obs < nmin or not splits:
        node.is_leaf = True
        return

    # SORT FEATURES BY IMPURITY AND PICK THE FEATURE WITH LOWEST IMPURITY
    # THAT MEETS THE MINLEAF CONSTRAINT
    sorted_splits = sorted(enumerate(splits), key=lambda x: x[1][1])
    for feature_index, (split, _) in sorted_splits:
        feature_values = feature_obs[:, feature_index]
        split_feature = feature_index
        l_feat_obs = feature_obs[feature_obs[:, feature_index] <= split]
        r_feat_obs = feature_obs[feature_obs[:, feature_index] > split]
        
        l_size, _ = l_feat_obs.shape
        r_size, _ = r_feat_obs.shape

        if minleaf <= l_size and minleaf <= r_size:
            break

    l_class_obs = class_obs[feature_values <= split]
    r_class_obs = class_obs[feature_values > split]

    node.feature_value = split_feature
    node.split_threshold = split
    node.left_child, node.right_child = Node(), Node()

    # RECURSE OVER LEFT AND RIGHT NODE CHILD
    extend_node(node.left_child, l_feat_obs, l_class_obs, nmin, minleaf, n_of_feat)
    extend_node(node.right_child, r_feat_obs, r_class_obs, nmin, minleaf, n_of_feat)


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
    highest_red = 0
    x_sorted = np.sort(np.unique(x))
    candidate_splitpoints = (x_sorted[0:(x_sorted.size - 1)] + x_sorted[1:x_sorted.size]) / 2
    best_split = None  # IF THIS VALUE IS RETURNED, THEN WE KNOW THAT THERE EXISTS NO POSSIBLE SPLIT

    ## for every split option calculate the gini index.
    for split in candidate_splitpoints:
        left_child = y[x <= split]
        right_child = y[x > split]
        majority_label_class = collections.Counter(y).most_common(1)[0][0]
        # MAKE A LEAF NODE IF THE NUMBER OF OBSERVATIONS
        # FOR EITHER CHILD NODE IS TOO LOW
        if len(left_child) >= minleaf or len(right_child) >= minleaf:
            imp_left = impurity(left_child)
            imp_right = impurity(right_child)
            pi_left = len(left_child) / len(x)
            pi_right = len(right_child) / len(x)
            imp_parent = impurity(y)
            red = imp_parent - ((pi_left * imp_left) + (pi_right * imp_right))

            if red > highest_red:
                highest_red = red
                best_split = split
    # return the best split option with the corresponding impurity reduction
    return best_split, highest_red

def print_results(conf_matrix, acc, prec_rec, model_name):
    print(f"MODEL: {model_name} \n\n CONFUSION MATRIX: \n\n \
                    {conf_matrix} \n\n ACCURACY: {acc} \n\n \
                    PRECISION AND RECALL: {prec_rec} \n\n")

if __name__ == "__main__":
    dir = os.getcwd()
    credit_data = np.genfromtxt(dir + "/credit.txt", delimiter=",", skip_header=True)
    pima_data = np.genfromtxt(dir + "/pima.txt", delimiter=",", skip_header=False)

    num_of_obs, n_of_feat = pima_data.shape
    X = pima_data[:, :n_of_feat - 1]
    Y = pima_data[:, n_of_feat - 1]
    tree = tree_grow(X, Y, nmin=20, minleaf=5, nfeat=8)
    pred = tree_pred(X, tr=tree)

    print(confusion_matrix(Y, pred))
    sys.exit()
    n_of_obs, _  = y_train.shape

    start_time = time.time()

    n,m = pima_data.shape
    x_train = pima_data[:,:m-1]
    x_test = pima_data[:, m-1]

    tree = tree_grow(x_train, x_test, nmin=20, minleaf=5, nfeat=8)
    tree_bagged = tree_grow_b(x_train, x_test, nmin=20, minleaf=5, nfeat=8, m=100)
    random_forest = tree_grow_b(x_train, x_test, nmin=20, minleaf=5, nfeat=6, m=100) 

    y_pred_n = tree_pred(x_train, tree) 
    print_results(confusion_matrix(x_test,y_pred_n), accuracy_score(x_test,y_pred_n), \
        precision_recall_fscore_support(x_test,y_pred_n), model_name="REGULAR TREE")

    y_pred_b = tree_pred_b(x_train, tree_bagged) 
    print_results(confusion_matrix(x_test,y_pred_b), accuracy_score(x_test,y_pred_b), \
        precision_recall_fscore_support(x_test,y_pred_b), model_name="BAGGED TREE")
    

    y_pred_f = tree_pred_b(x_train, random_forest) 
    print_results(confusion_matrix(x_test,y_pred_f), accuracy_score(x_test,y_pred_f), \
        precision_recall_fscore_support(x_test, y_pred_f), model_name="RANDOM FOREST")

    sys.exit()

    tree = tree_grow(x_train, x_test, nmin=15, minleaf=5, nfeat=41)
    tree_bagged = tree_grow_b(x_train, x_test, nmin=15, minleaf=5, nfeat=41, m=100)
    random_forest = tree_grow_b(x_train, x_test, nmin=15, minleaf=5, nfeat=6, m=100) 

    print(f"FIRST TWO SPLITS REGULAR TREE:\
            {tree.root.feature_value, tree.root.left_child.feature_value,tree.root.right_child.feature_value} \n")

    y_pred_n = tree_pred(y_train, tree) 
    print_results(confusion_matrix(y_test,y_pred_n), accuracy_score(y_test,y_pred_n), \
        precision_recall_fscore_support(y_test,y_pred_n), model_name="REGULAR TREE")

    print(time.time()-start_time)
    start_time = time.time()


    y_pred_b = tree_pred_b(y_train, tree_bagged) 
    print_results(confusion_matrix(y_test,y_pred_b), accuracy_score(y_test,y_pred_b), \
        precision_recall_fscore_support(y_test,y_pred_b), model_name="BAGGED TREE")
    
    print(time.time()-start_time)
    start_time = time.time()

    y_pred_f = tree_pred_b(y_train, random_forest) 
    print_results(confusion_matrix(y_test,y_pred_f), accuracy_score(y_test,y_pred_f), \
        precision_recall_fscore_support(y_test, y_pred_f), model_name="RANDOM FOREST")

    print(time.time()-start_time)