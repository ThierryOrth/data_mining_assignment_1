from classification_tree_builder import tree_grow, tree_grow_b, tree_pred, tree_pred_b
from data_loader import x_train, y_train, x_test, y_test
import numpy as np

def eval_metrics(y_true,y_pred,pos=1.0, neg=0.0):
    n = len(y_true)
    tp = fp = tn = fn = 0

    for i in range(n):
        if y_true[i] == pos and y_pred[i] == pos:
            tp+=1
        elif y_true[i] == pos and y_pred[i] == neg:
            fp+=1
        elif y_true[i] == neg and y_pred[i] == pos:
            fn+=1
        elif y_true[i] == neg and y_pred[i] == neg:
            tn+=1
    
    def confusion_matrix(tp,fp,tn,fn):
        return np.array([[tp,fn],[fp,tn]])

    def accuracy(tp,fp,tn,fn):
        return tp/(tp+fp+tn+fn)

    def precision(tp,fp):
        return tp/(tp+fp)

    def recall(tp,fn):
        return tp/(tp+fn)

    return confusion_matrix(tp,fp,tn,fn), accuracy(tp,fp,tn,fn), \
                                     precision(tp,fp), recall(tp,fn)

def print_results(conf_matrix, acc, prec, rec, model_name):
    print(f"Model: {model_name} \n \
            \t Confusion matrix: {conf_matrix} \n \
            \t\t Accuracy: {acc} \n \
            \t\t\t Precision: {prec} \n \
            \t\t\t\t Recall {rec} \n")

if __name__ == "__main__":
    tree = tree_grow(x=x_train, y=y_train, nmin=15, minleaf=5, nfeat=41)
    tree_bagged = tree_grow_b(x=x_train, y=y_train, nmin=15, minleaf=5, nfeat=41, m=100)
    random_forest = tree_grow_b(x=x_train, y=y_train, nmin=15, minleaf=5, nfeat=41, m=100) # nfeat:=6 for analysis

    y_pred_n = y_pred_b = y_pred_f = []

    n , _ = x_test.shape

    for i in range(n):
        y_pred_n.append(tree_pred(x_test[i, :], tree))
        y_pred_b.append(tree_pred_b(x_test[i, :], tree_bagged))
        y_pred_f.append(tree_pred_b(x_test[i, :], random_forest))

    conf_matrix_n, acc_n, prec_n, rec_n = eval_metrics(y_test, y_pred_n)
    print_results(conf_matrix_n, acc_n, prec_n, rec_n, model_name="regular classification tree")

    conf_matrix_b, acc_b, prec_b, rec_b = eval_metrics(y_test, y_pred_b)
    print_results(conf_matrix_b, acc_b, prec_b, rec_b, model_name="bagged classfication tree")

    conf_matrix_f, acc_f, prec_f, rec_f = eval_metrics(y_test, y_pred_f)
    print_results(conf_matrix_f, acc_f, prec_f, rec_f, model_name="random forest")