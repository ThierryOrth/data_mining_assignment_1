import numpy as np
import collections
import sys
from data_loader import *
from tree_structures import *

    # FOR EACH NODE, PROVIDE A SPLIT THRESHOLD USING A DICTIONARY MAPPING

    # FOR EACH NODE, PROVIDE A DICTIONARY THAT MAPS ATTRIBUTE VALUES TO PREDICTION 

    # IF LEAF NODE IS REACHED, MAJORITY CLASS IS ASSIGNED BY TAKING THE KEY WHICH HAS THE MAXIMUM VALUE

ATTRIBUTES = {0:"age", 1:"married", 2:"house", 3:"income", 4:"gender", 5:"class"}
ATTRIBUTE_MAP = {0:0, 1:1, 2:2, 3:3, 4:4}

def getimpurity(y):
    total = len(y)
    p_zeros = len(y[y[:] == 0])/total
    p_ones = len(y[y[:]==1])/total
    return p_zeros*p_ones

def bestsplit(x,y):
    lowest_impurity = 1
    best_split = -np.infty
        
    x_sorted = np.sort(np.unique(x))
    candidate_splits = (x_sorted[0:-1]+x_sorted[1:len(x)])/2

    for split in candidate_splits:
        left_child = y[x <= split]
        right_child = y[split < x]

        i_left = getimpurity(left_child)
        i_right = getimpurity(right_child)
        pi_left = len(left_child)/len(x)
        pi_right = len(right_child)/len(x)

        impurity = (pi_left * i_left) + (pi_right * i_right)

        if impurity < lowest_impurity:
            lowest_impurity = impurity
            best_split = split

    return best_split

def tree_grow(x, y, nmin, minleaf, nfeat):
    tree = Tree()

    if 1<len(x.shape):
        _ , m = x.shape
        
    else:
        m = len(x)
        nodes = None

    print(f"x:{x}, \n y:{y}")
        
    if nmin<m:

        if 0<getimpurity(y):
            index = np.random.choice(np.arange(0,m,1))
            feature_values = x[:, index]
            best_split = bestsplit(feature_values, y)

            print(best_split)

            tree.set_threshold(best_split)
            tree.feature_value = index

            left = feature_values[feature_values <= best_split]
            right = feature_values[best_split < feature_values]

            left_labels = y[feature_values <= best_split]
            right_labels = y[best_split < feature_values]

            if len(left)<minleaf or len(right)<minleaf:
                feature_value = 1 if len(right_labels)<len(left_labels) else 0
                tree.feature_value = feature_value           
            else:
                # next nodes should be connected to previous nodes
                tree_grow(x,y,nmin,minleaf,nfeat)
                tree_grow(x,y,nmin,minleaf,nfeat)

        
            print(f"x:{x} \n, y:{y}, \n feature index:{index}, \n split:{best_split}, \n left:{left},\n right:{right}")

        
    return tree




            





    # TODO: CONSTRUCT TREE OBJECT
    # TODO: MAKE LEAF NODES IF CONTAIN ONLY SINGLE OBSERVATION
    # TODO: NMIN, MINLEAF, NFEAT


    # while 0<len(nodes):
    #     
    #     current_node = nodes[index]
    #     node_labels = labels[index]
    #     print("INDEX", index)
    #     del nodes[index]
    #     del labels[index]
    #     m=len(nodes)

    #     if 0<getimpurity(node_labels):
    #         best_split = bestsplit(current_node, node_labels)

    #         print(f"SPLIT: {best_split}")
            
    #         left_child = current_node[current_node <= best_split]
    #         right_child = current_node[best_split < current_node]

    #         left_child_labels = node_labels[current_node <= best_split]
    #         right_child_labels = node_labels[best_split < current_node]



    #         nodes.append(left_child)
    #         nodes.append(right_child)
    #         labels.append(left_child_labels)
    #         labels.append(right_child_labels)

    #         # tree_grow(left)
    #         # tree_grow(right)

    #         print(f"BEST SPLIT: {best_split}, \n\n LEFT CHILD WITH PREDICTION: {left_child, left_child_labels}, \
    #                                     \n\n RIGHT CHILD WITH PREDICTION: {right_child, right_child_labels}, \
    #                                         \n\n UPDATED NODE LIST: {nodes}, \n\n UPDATED LABELS: {labels}")

    return 

def tree_pred(x, tr):
    pass

def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
    trees = []
    for i in range(m):
        pass
    return trees

def tree_pred_b(x, trs):
    pass



if __name__ == "__main__":
    bestsplit(CREDIT_DATA[:,3],CREDIT_DATA[:,5])

    # n, m = CREDIT_DATA.shape
    # data = [elem for elem in CREDIT_DATA[:n,:]]
    # labels = CREDIT_DATA[:,5]
    # print(data)
    # print(labels)
    tree_grow(x=CREDIT_DATA[:,:5], y=CREDIT_DATA[:,5], nmin = 1, minleaf = 1, nfeat = 1)
    # tree_grow(x=credit_data[:,:5],y=credit_data[:,5], nmin = 15, minleaf = 5, nfeat = 41)
  