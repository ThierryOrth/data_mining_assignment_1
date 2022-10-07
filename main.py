import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import os
import collections

dir = os.getcwd()
credit_data = np.genfromtxt(dir + "/credit.txt", delimiter=",", skip_header=True)
pima_data = np.genfromtxt(dir + "/pima.txt", delimiter=",", skip_header=False)

class Node:
    """Constructs a node with a set of internal properties useful
        for classification trees: the feature index in the data matrix,
        the split threshold, the left and right child nodes and whether
        it is a leaf or not."""
    def __init__(self, feature_value=None,split_threshold=None, left_child=None,right_child=None,is_leaf=None, majority=None):
        self.feature_value = feature_value # COLUMN IN THE MATRIX FOR NON-TERMINAL NODES, MAJORITY VOTE PREDICTION FOR TERMINAL NODES
        self.split_threshold = split_threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = is_leaf
        self.majority = majority

class Tree:
    """Constructs a tree by taking an initial node
        as root."""
    def __init__(self, root:Node):
        if isinstance(root,Node):
            self.root = root
        else:
            raise ValueError("")


def tree_grow(x:np.array,y:np.array,nmin:int,minleaf:int,nfeat:int) -> Tree:
    """Starts with an initial node, extends it and returns a tree consisting of
                the initial node as root."""
    root_node = Node()
    extend_node(root_node, x, y, nmin, minleaf, nfeat)
    return Tree(root_node)


def tree_pred(x, tr)->float:
    """"Traverses the tree by comparing feature value with split threshold until it finds a leaf. If that leaf is reached,
        we predict the majority vote."""
    pred = []
    for row in x:
        current = tr.root
        while not current.is_leaf:
            feature_value = current.feature_value
            threshold = current.split_threshold
            if row[feature_value] <= threshold:
                current = current.left_child
            else:
                current = current.right_child
        pred.append(current.feature_value)
    return pred


def tree_grow_b(feature_obs: np.array, class_obs: np.array, nmin: int, minleaf: int, nfeat: int, m: int) -> list:
    """Grows a bagged tree by sampling with replacement from the feature observations.
       The hyperparameter nfeat can be used to train a random forest."""

    n_of_obs, _ = feature_obs.shape
    trees = []

    for i in range(m):
        root_node = Node()
        indices = np.random.choice(np.arange(0, n_of_obs, 1), n_of_obs, replace=True)
        feat_sample, class_sample = feature_obs[indices, :], class_obs[indices]

        extend_node(root_node, feat_sample, class_sample, nmin, minleaf, nfeat)
        trees.append(Tree(root_node))

    return trees


def tree_pred_b(feature_obs: np.array, trs: list) -> float:
    """Computes majority prediction given a feature array
       and a collection of classification trees."""
    row_index = len(trs)
    col_index, _ = feature_obs.shape
    predictions = np.zeros((row_index, col_index))
    print("shape", predictions.shape)

    for index, tree in enumerate(trs):
        predictions[index, :] = tree_pred(feature_obs, tree)
        # predictions.append(tree_pred(feature_obs, tree))

    majority_labels = [collections.Counter(predictions[:, i]).most_common(1)[0][0] for i in range(col_index)]
    print("PREDICTIONS ALL", predictions[:20, :])
    print("PREDICTIONS MAJ", majority_labels[:20])
    return majority_labels

def extend_node(node:Node, x:np.array, y:np.array, nmin:int, minleaf:int, nfeat:int):
    """Given a current node, checks whether it can be extended. If not, the node becomes
               a leaf node. If so, then we split the node into two child nodes, which are fed into
               the same function by recursion."""
    splits = []
    num_of_obs, n_of_feat = x.shape

    if nfeat<n_of_feat:
        indices = np.random.choice(np.arange(0,nfeat,1), nfeat, replace=False)
        x = x[:, indices]
        n_of_feat = nfeat

    # Calculate the best split option for every feature in array x
    for f in range(n_of_feat):
        split = bestsplit(x[:, f], y,minleaf)  # best split for feature f in n_of_feat
        splits.append(split)  # add best split to list splits

    # check for all the best splits of all features which one
    # of these has the best impurity reduction
    best_red = np.max([b[1] for b in splits])

    # get the index for the best split of all features
    feature_index = [tup[1] for tup in splits].index(best_red)
    feature_values = x[:, feature_index]
    # return the threshold value
    threshold = splits[feature_index][0]

    node.feature_value = feature_index
    node.split_threshold = threshold

    majority_class = collections.Counter(y).most_common(1)[0][0]
    node.majority = majority_class
    # MAKE A LEAF NODE IF THERE EXISTS NO SPLIT, THE NUMBERS OF
    # OBSERVATIONS IS TOO LOW OR IMPURITY EQUALS ZERO
    if num_of_obs < nmin or gini_index(y) == 0:
        node.is_leaf = True
        node.feature_value = majority_class
        return

    # assign each row to either the left or right child based on its value
    left = x[x[:, feature_index] <= threshold]
    right = x[x[:, feature_index] > threshold]

    left_labels = y[feature_values <= threshold]
    right_labels = y[feature_values > threshold]

    node.left_child = Node()
    node.right_child = Node()

    extend_node(node.left_child, left, left_labels, nmin, minleaf, nfeat)
    extend_node(node.right_child, right, right_labels, nmin, minleaf, nfeat)

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
        majority_class = collections.Counter(y).most_common(1)[0][0]
        # MAKE A LEAF NODE IF THE NUMBER OF OBSERVATIONS
        # FOR EITHER CHILD NODE IS TOO LOW
        if len(left_child) >= minleaf or len(right_child) >= minleaf:
            imp_left = gini_index(left_child)
            imp_right = gini_index(right_child)
            pi_left = len(left_child) / len(x)
            pi_right = len(right_child) / len(x)
            imp_parent = gini_index(y)
            red = imp_parent - ((pi_left * imp_left) + (pi_right * imp_right))

            if red > highest_red:
                highest_red = red
                best_split = split
    # return the best split option with the corresponding impurity reduction
    return best_split, highest_red


def gini_index(arr):
    """Computes the Gini index as impurity function."""
    left = arr[arr[:]==1].size #Amount of observations in the left child
    right = arr[arr[:]==0].size #Amount of observations in the right child
    total = arr.size #Amount of observations in the parent node (where the split is adjusted)
    ##Calculate the Gini index
    gini = (left/total)*(right/total)
    return gini

if __name__ == "__main__":
    
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

    #num_of_obs, n_of_feat = pima_data.shape
    #X = pima_data[:,:n_of_feat-1]
    #Y = pima_data[:,n_of_feat-1]
    #tree = tree_grow(x=X, y=Y, nmin = 20, minleaf = 5, nfeat = 8)
    #pred = tree_pred(x=X,tr=tree)

    #print(confusion_matrix(Y,pred))
    #sys.exit()
    #n_of_obs, _ = y_train.shape

    start_time = time.time()

    n, m = pima_data.shape
    x_train = pima_data[:, :m - 1]
    x_test = pima_data[:, m - 1]

    tree = tree_grow(x_train, x_test, nmin=15, minleaf=5, nfeat=41)
    tree_bagged = tree_grow_b(x_train, x_test, nmin=15, minleaf=5, nfeat=41, m=100)
    random_forest = tree_grow_b(x_train, x_test, nmin=15, minleaf=5, nfeat=6, m=100)

    print(f"FIRST TWO SPLITS REGULAR TREE:\
                {tree.root.feature_value, tree.root.left_child.feature_value, tree.root.right_child.feature_value} \n")

    y_pred_n = tree_pred(y_train, tree)
    print_results(confusion_matrix(y_test, y_pred_n), accuracy_score(y_test, y_pred_n), \
                  precision_recall_fscore_support(y_test, y_pred_n), model_name="REGULAR TREE")

    print(time.time() - start_time)
    start_time = time.time()

    y_pred_b = tree_pred_b(y_train, tree_bagged)
    print_results(confusion_matrix(y_test, y_pred_b), accuracy_score(y_test, y_pred_b), \
                  precision_recall_fscore_support(y_test, y_pred_b), model_name="BAGGED TREE")

    print(time.time() - start_time)
    start_time = time.time()

    y_pred_f = tree_pred_b(y_train, random_forest)
    print_results(confusion_matrix(y_test, y_pred_f), accuracy_score(y_test, y_pred_f), \
                  precision_recall_fscore_support(y_test, y_pred_f), model_name="RANDOM FOREST")

    print(time.time() - start_time)
