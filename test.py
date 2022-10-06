from tkinter import N
import numpy as np
from data_loader import x_train, y_train, x_test, y_test
from classification_tree_builder import tree_grow, tree_grow_b, tree_pred, tree_pred_b

import sys
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.tree import DecisionTreeClassifier
    
### FOR COMPARISON ### 

decision_tree = DecisionTreeClassifier(min_samples_split=15, min_samples_leaf=5)
decision_tree.fit(x_train,x_test)
y_pred_t = decision_tree.predict(y_train)
print(accuracy_score(y_test,y_pred_t))
print(confusion_matrix(y_test, y_pred_t))
print(precision_recall_fscore_support(y_test,y_pred_t))


# bagging_tree = BaggingClassifier(min_samples_split=15, min_samples_leaf=5, max_samples=100)
# bagging_tree.fit(x_train,x_test)
# y_pred_b = bagging_tree.predict(y_train)
# print(accuracy_score(y_test,y_pred_b))
# print(confusion_matrix(y_test, y_pred_b))
# print(precision_recall_fscore_support(y_test,y_pred_b))
# # bagged_tree = ???

random_forest = RandomForestClassifier(n_estimators=100, min_samples_split=15,min_samples_leaf=5, max_features=6)
random_forest.fit(x_train,x_test)
y_pred_f = random_forest.predict(y_train)
print(accuracy_score(y_test,y_pred_f))
print(confusion_matrix(y_test, y_pred_f))
print(precision_recall_fscore_support(y_test,y_pred_f))

    # tree = tree_grow(x_train, x_test, nmin=15, minleaf=5, nfeat=41)
    # tree_bagged = tree_grow_b(x_train, x_test, nmin=15, minleaf=5, nfeat=41, m=100)