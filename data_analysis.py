import numpy as np
from data_loader import x_train, y_train, x_test, y_test
from classification_tree_builder import tree_grow, tree_grow_b, tree_pred, tree_pred_b
import sys

from sklearn.metrics import accuracy_score


def evaluation_metrics(y_true,y_pred,pos=1.0, neg=0.0):
    """Returns evaluation metrics given true
        and predicted values."""
    obs = len(y_true)
    true_pos = false_pos = true_neg = false_neg = 0

    for obs_index in range(obs):
        if y_pred[obs_index] == pos and y_true[obs_index] == pos:
            true_pos+=1
        elif y_pred[obs_index] == neg and y_true[obs_index] == pos:
            false_neg+=1
        elif y_pred[obs_index] == pos and y_true[obs_index] == neg:
            false_pos+=1
        elif y_pred[obs_index] == neg and y_true[obs_index] == neg:
            true_neg+=1
    
    def confusion_matrix(true_pos,false_pos,true_neg,false_neg):
        return np.array([[true_pos,false_neg],[false_pos,true_neg]])

    def accuracy(true_pos,false_pos,true_neg,false_neg):
        return true_pos/(true_pos+false_pos+true_neg+false_neg)

    def precision(true_pos,false_pos):
        return true_pos/(true_pos+false_pos) if 0<true_pos else 0.0

    def recall(true_pos,false_neg):
        return true_pos/(true_pos+false_neg) if 0<true_pos else 0.0

    def f1_score(precision, recall):
        return 2*((precision*recall)/(precision+recall))

    return confusion_matrix(true_pos,false_pos,true_neg,false_neg), accuracy(true_pos,false_pos,true_neg,false_neg), \
                                     precision(true_pos,false_pos), recall(true_pos,false_neg)

def print_results(conf_matrix, acc, prec, rec, model_name):
    print(f"MODEL: {model_name} \n CONFUSION MATRIX: \n {conf_matrix} \n ACCURACY: {acc} \n PRECISION: {prec} \n RECALL: {rec} \n")

if __name__ == "__main__":
    n_of_obs, _  = y_train.shape

    tree = tree_grow(x_train, x_test, nmin=15, minleaf=5, nfeat=41)
    tree_bagged = tree_grow_b(x_train, x_test, nmin=15, minleaf=5, nfeat=41, m=100)
    random_forest = tree_grow_b(x_train, x_test, nmin=15, minleaf=5, nfeat=6, m=100) # set nfeat to 6 for analysis

    print(f"FIRST TWO SPLITS REGULAR TREE:\
            {tree.root.feature_value, tree.root.left_child.feature_value,tree.root.right_child.feature_value} \n")


    y_pred_n = [tree_pred(y_train[i, :], tree) for i in range(n_of_obs)]
    conf_matrix_n, acc_n, prec_n, rec_n = evaluation_metrics(y_test, y_pred_n)
    print_results(conf_matrix_n, acc_n, prec_n, rec_n, model_name="REGULAR CLASSIFICATION TREE")

    y_pred_b = [tree_pred_b(y_train[i, :], tree_bagged) for i in range(n_of_obs)]
    conf_matrix_b, acc_b, prec_b, rec_b = evaluation_metrics(y_test, y_pred_b)
    print_results(conf_matrix_b, acc_b, prec_b, rec_b, model_name="BAGGED CLASSIFICATION TREE")

    y_pred_f = [tree_pred_b(y_train[i, :], random_forest) for i in range(n_of_obs)]
    conf_matrix_f, acc_f, prec_f, rec_f = evaluation_metrics(y_test, y_pred_f)
    print_results(conf_matrix_f, acc_f, prec_f, rec_f, model_name="RANDOM FOREST")
    
    print(accuracy_score(y_test,y_pred_n), accuracy_score(y_test,y_pred_b),accuracy_score(y_test,y_pred_f))

