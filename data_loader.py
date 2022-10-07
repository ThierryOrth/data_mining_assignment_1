import numpy as np
import pandas as pd
import os

dir = os.getcwd()

credit_data = np.genfromtxt(dir+"/CREDIT.txt", delimiter=",", skip_header=True)
pima_data = np.genfromtxt(dir+"/pima.txt", delimiter=",", skip_header=False)

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

