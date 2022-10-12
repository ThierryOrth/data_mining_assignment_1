"""Christine Hedde-von Westernhagen (5987932), Kim van Genderen (6497039), Thierry Orth (6176178)"""
import collections, os, time, graphviz
from tkinter import N
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from statsmodels.stats.contingency_tables import mcnemar

class Node:
    """Constructs a node with the following properties: left child node, right child node, 
       feature value, threshold value, the majority label and whether the node is leaf."""
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.feature = None 
        self.threshold = None
        self.is_leaf = False
        self.majority_label = None

class Tree:
    """Constructs a tree by a node input. Tree structure is given recursively by
       the left and child nodes of that node."""
    def __init__(self, root:Node):
        if isinstance(root,Node):
            self.root = root
        else:
            raise ValueError("A tree cannot be constructed without root node!")

def tree_grow(feature_obs:np.array, class_obs:np.array, nmin:int, minleaf:int, nfeat:int) -> Tree:
    """Grows a tree from an initial node by using the extend_node function 
       and returning a tree object from the initial node as root node."""
    root_node = Node()

    extend_node(root_node, feature_obs, class_obs, nmin, minleaf, nfeat)

    return Tree(root_node)

def tree_pred(feature_obs: np.array, tr: Tree) -> float:
    """Predicts the class label for a feature vector by traversing the tree. For each feature vector 
       and at each node, the tree is traversed by comparing the split threshold with the observed 
       feature value. Once a leaf node is reached, the majority label is predicted. Returns a vector of the predicted class labels."""
    pred=[]

    ### loop until we find a leaf node ###
    for feature_row in feature_obs:
        current_node = tr.root
        while not current_node.is_leaf:
            feature_value = int(current_node.feature)

            ### go to the left child node if the feature value is below the threshold ###
            if feature_row[feature_value] <= current_node.threshold:
                current_node = current_node.left_child

            ### otherwise, go to the right child ###
            else:
                current_node = current_node.right_child

        ### set the predicted label for the current feature vector ###
        pred.append(current_node.majority_label)

    return pred

def tree_grow_b(feature_obs:np.array, class_obs:np.array, nmin:int, minleaf:int, nfeat:int, m:int) -> list:
    """Grows a bagged tree by sampling with replacement from the feature observations. If the hyperparameter
       nfeat is below feature vector length, the function trains a random forest. Returns list of trees."""

    n_of_obs, _ = feature_obs.shape
    trees = []

    ### construct m trees ###
    for i in range(m):
        # print(f"===== EPOCH #{i} =====")
        root_node = Node()

        ### draw a sample with replacement as big as the number of observed feature vectors ###
        indices = np.random.choice(np.arange(0,n_of_obs,1), n_of_obs, replace=True)
        feat_sample, class_sample = feature_obs[indices,:], class_obs[indices]

        extend_node(root_node, feat_sample, class_sample, nmin, minleaf, nfeat)
        trees.append(Tree(root_node))

    return trees

def tree_pred_b(feature_obs: np.array, trs: list) -> float:
    """Predicts the class labels of a set of feature vectors using an ensemble of
       classification trees. For each feature vector, the function returns the 
       majority vote given by the tree ensemble."""

    row_index = len(trs)
    col_index, _ = feature_obs.shape 
    predictions = np.zeros((row_index,col_index))
    print("shape", predictions.shape)

    ### get predicted class labels for all feature vectors and a single tree ###
    for index, tree in enumerate(trs):
        predictions[index, :] = tree_pred(feature_obs,tree)

    ### get majority labels for each feature vector ###    
    majority_labels = [collections.Counter(predictions[:,i]).most_common(1)[0][0] for i in range(col_index)]

    return majority_labels

def extend_node(node:Node, x:np.array, y:np.array, nmin:int, minleaf:int, nfeat:int):
    """Given a current node, extends it if it meets the following constraints: there are features to
       split on, the number of observations is sufficiently high, impurity is above zero and the number
       of observations in the leaf node are sufficient. If the node can be extended, we split the node
       into two child nodes which are fed into this function by recursion. If not, it becomes a leaf node."""

    splits = []
    feat_to_imp = dict()
    feat_to_split = dict()
    num_of_obs, n_of_feat = x.shape
    majority_label = collections.Counter(y).most_common(1)[0][0]
    node.majority_label = majority_label

    indices = np.arange(0,n_of_feat,1)

    ### sample random indices without replacement if lower than the number of feature variables ###
    if nfeat<n_of_feat:
        indices = np.random.choice(np.arange(0,nfeat,1), nfeat, replace=False)
        n_of_feat = nfeat

    for index in indices:
        split, imp = bestsplit(x[:,index],y,minleaf)
        
        ### if split is found, map features to split threshold and impurity scores ###
        if split:
            feat_to_split[index]=split
            feat_to_imp[index]=imp

    ### leaf node if no splits are found, number of observations is too low or node is pure ###
    if not feat_to_split or num_of_obs < nmin or impurity(y) == 0:
        node.is_leaf = True
        return

    ### get feature index and split threshold of feature with lowest impurity ### 
    feature_index = max(feat_to_imp, key=feat_to_imp.get)
    threshold = feat_to_split[feature_index]

     ### set feature value and threshold at node ###
    node.feature = feature_index
    node.threshold = threshold

    feature_values = x[:, feature_index]
    
    ### leaf node if no splits are found, number of observations is too low or node is pure ###
    if num_of_obs < nmin or impurity(y) == 0:
        node.is_leaf = True
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


def impurity(class_obs: np.array) -> float:
    """Returns the Gini index of an array of
       class observations."""
    total = class_obs.size
    pos_class = class_obs[class_obs == 1].size
    neg_class = class_obs[class_obs == 0].size
    return (pos_class / total) * (neg_class / total)


def bestsplit(x,y,minleaf):
    """Returns the best split and associated impurity reduction from a range of candidate splits
            using the Gini index as impurity function and using the minleaf constraint."""
    highest_red = 0
    x_sorted = np.sort(np.unique(x))
    candidate_splitpoints = (x_sorted[0:(x_sorted.size - 1)] + x_sorted[1:x_sorted.size]) / 2
    best_split = None  ### used as check if no splits are found ###

    ### compute Gini impurity score for each split option ###
    for split in candidate_splitpoints:

        ### check whether the resulting split meets the minleaf constraint ### 
        left_child = y[x <= split]
        right_child = y[x > split]

        if minleaf <=len(left_child) or  minleaf<=len(right_child):
            imp_left = impurity(left_child)
            imp_right = impurity(right_child)
            pi_left = len(left_child) / len(x)
            pi_right = len(right_child) / len(x)
            imp_parent = impurity(y)
            red = imp_parent - ((pi_left * imp_left) + (pi_right * imp_right))

            ### update impurity scores and split threshold if current impurity is lower ###
            if highest_red<red:
                highest_red = red
                best_split = split

    return best_split, highest_red

def load_data(dataset):
    """Loads dataset and partitions dataset into training and test data."""
    dir = os.getcwd()

    if dataset=="credit":
        credit_data = np.genfromtxt(dir + "/credit.txt", delimiter=",", skip_header=True)
        _, m= credit_data.shape

        x_train = credit_data[:,:m-1]
        y_train = credit_data[:,m-1]

        return x_train, y_train
    
    elif dataset=="pima":
        pima_data = np.genfromtxt(dir + "/pima.txt", delimiter=",", skip_header=False)
        _, m = pima_data.shape

        x_train = pima_data[:,:m-1]
        y_train = pima_data[:,m-1]

        return x_train, y_train

    elif dataset=="bug":
        bug_dataframe_train = pd.read_csv("eclipse-metrics-packages-2.0.csv", delimiter=";")
        bug_dataframe_test = pd.read_csv("eclipse-metrics-packages-3.0.csv", delimiter=";")

        ### get the relevant variable indices ###
        column_indices=[2]+[i for i in range(4, 44)]

        ### preprocessing: enforce categorical values on pre-bugs ###
        x_train = bug_dataframe_train.iloc[:,column_indices].to_numpy()
        x_train[:,0] = np.where(0.0<x_train[:,0], 1.0, 0.0)

        y_train = bug_dataframe_train.iloc[:,3].to_numpy()
        y_train = np.where(0.0<y_train, 1.0, 0.0)

        x_test = bug_dataframe_test.iloc[:,column_indices].to_numpy()
        x_test[:,0] = np.where(0.0<x_test[:,0], 1.0, 0.0)

        y_test = bug_dataframe_test.iloc[:,3].to_numpy()
        y_test = np.where(0.0<y_test, 1.0, 0.0)

        ### retrieve column names for graphical representation ###
        column_list = bug_dataframe_train.columns.to_list()
        column_names = [column_list[i] for i in column_indices]
        
        return column_names, x_train, y_train, x_test, y_test

def print_results(y_true,y_pred, name, runtime=None, save_results=False):
    """Prints the following classification metrics: confusion matrix, precision, recall, accuracy and runtime in seconds."""
    scores = precision_recall_fscore_support(y_true,y_pred)
    string = f"NAME OF MODEL: {name} \n CONFUSION MATRIX: \n {confusion_matrix(y_true,y_pred)} \n PRECISION: {scores[0]} \n RECALL: {scores[1]} \n ACCURACY SCORE: {accuracy_score(y_true,y_pred)} \n RUNTIME IN SECONDS: {runtime}"
    
    print(string)

    if save_results:
        with open(f"results_{name}.txt", "w") as file:
            file.write(string)

def visualize_tree(nodes, edges, node_names):
    """Visualizes tree given a set of nodes, edges and associated node names."""
    digraph = graphviz.Digraph(comment="first two splits classification tree")

    for name, node in nodes.items():
        majority_label = node.majority_label
    
        if not node.is_leaf:
            digraph.node(name, f"feature: {node_names[node.feature]} \n majority class: {majority_label}")
        else:
            digraph.node(name, f"leaf node \n majority class: {majority_label}")

    for parent, children in edges.items():
        left_child, right_child = children
        threshold = nodes[parent].threshold

        digraph.edge(parent, left_child, label=f"<={threshold}", len="1.00")
        digraph.edge(parent, right_child, label=f">{threshold}", len="1.00")

    digraph.render(directory='doctest-output', view=True)  

def stats_test(y_test, y_pred_A, y_pred_B):
    """Assembles confusion matrix needed for McNemar's test, conducts test, and returns results."""
    # identify in/correct predictions of algorithm A vs. B 
    n00 = [i for i in range(len(y_test)) if (y_pred_A[i] != y_test[i]) & (y_pred_B[i] != y_test[i])]
    n01 = [i for i in range(len(y_test)) if (y_pred_A[i] != y_test[i]) & (y_pred_B[i] == y_test[i])]
    n10 = [i for i in range(len(y_test)) if (y_pred_A[i] == y_test[i]) & (y_pred_B[i] != y_test[i])]
    n11 = [i for i in range(len(y_test)) if (y_pred_A[i] == y_test[i]) & (y_pred_B[i] == y_test[i])]

    # confusion matrix of number of in/correct predicitons
    table = [[len(n00), len(n01)],
             [len(n10), len(n11)]]

    # test table for significant differences from expected counts under Chi2 distribution
    return mcnemar(table, exact=False, correction=False)


if __name__ == "__main__":
    np.random.seed(42)
    
    ### TEST EXAMPLE ###
    # x_train, y_train = load_data("pima")
    # _, m = x_train.shape

    # regular_tree = tree_grow(x_train,y_train, nmin=20, minleaf = 5, nfeat=m)
    # y_pred = tree_pred(x_train, regular_tree)
    # print(print_results(y_train, y_pred, ""))

    column_names, x_train, y_train, x_test, y_test = load_data(dataset="bug")

    ### train and evaluate regular classification tree ###
    start_time = time.time()
    regular_tree = tree_grow(x_train, y_train, nmin=15, minleaf=5, nfeat=41)
    y_pred_n = tree_pred(x_test, regular_tree)
    print_results(y_test, y_pred_n, name="regular_tree", runtime=time.time() - start_time, save_results=True)

    nodes = dict({"A":regular_tree.root, "B":regular_tree.root.left_child, "C":regular_tree.root.right_child, \
            "D":regular_tree.root.right_child.left_child,"E":regular_tree.root.right_child.right_child})
    edges = dict({"A":("B","C"),"C":("D","E")})
    
    visualize_tree(nodes, edges, column_names)

    ## train and evaluate bagged tree ###
    start_time = time.time()
    bagged_tree = tree_grow_b(x_train, y_train, nmin=15, minleaf=5, nfeat=41, m=100)
    y_pred_b = tree_pred_b(x_test, bagged_tree)
    print_results(y_test,y_pred_b, name="bagged_tree", runtime=time.time() - start_time, save_results=True)

    ### train and evaluate random forest ###
    start_time = time.time()
    random_forest = tree_grow_b(x_train, y_train, nmin=15, minleaf=5, nfeat=6, m=100)
    y_pred_f = tree_pred_b(x_test, random_forest)
    print_results(y_test, y_pred_f, name="random_forest", runtime=time.time() - start_time, save_results=True)

    ### compare predictive performance statistically ###
    print(f"Single vs. bagging \n {stats_test(y_test=y_test, y_pred_A=y_pred_n, y_pred_B=y_pred_b)}")
    print(f"Bagging vs. random forest \n {stats_test(y_test=y_test, y_pred_A=y_pred_b, y_pred_B=y_pred_f)}")
    print(f"Random forest vs. single \n {stats_test(y_test=y_test, y_pred_A=y_pred_f, y_pred_B=y_pred_n)}")
