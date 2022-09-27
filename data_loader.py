import numpy as np
import os
import pandas as pd


DIR = os.getcwd()

CREDIT_DATA = np.genfromtxt(DIR+"/CREDIT.txt", delimiter=",", skip_header=True)
PIMA_DATA = np.genfromtxt(DIR+"/pima.txt", delimiter=",", skip_header=False)

BUG_DATAFRAME_TRAIN = pd.read_csv("eclipse-metrics-packages-2.0.csv", delimiter=";")
BUG_DATAFRAME_TEST = pd.read_csv("eclipse-metrics-packages-3.0.csv", delimiter=";")
 
column_indices=[2]+[i for i in range(4, 44)]
x_train = BUG_DATAFRAME_TRAIN.iloc[:,column_indices].to_numpy()
y_train = BUG_DATAFRAME_TRAIN.iloc[:,3].to_numpy()
y_train = np.where(y_train>0.0, 1.0, 0.0)

x_test = BUG_DATAFRAME_TEST.iloc[:,column_indices].to_numpy()
y_test = BUG_DATAFRAME_TEST.iloc[:,3].to_numpy()
y_test = np.where(y_test>0.0, 1.0, 0.0)

