import numpy as np
import os
import pandas as pd


DIR = os.getcwd()

CREDIT_DATA = np.genfromtxt(DIR+"/CREDIT.txt", delimiter=",", skip_header=True)
PIMA_DATA = np.genfromtxt(DIR+"/pima.txt", delimiter=",", skip_header=False)

# BUG_DATA_TRAIN = np.genfromtxt(DIR+"/eclipse-metrics-packages-2.0.csv", delimiter=";", skip_header=True, dtype=None)
# BUG_DATA_TEST = np.genfromtxt(DIR+"/eclipse-metrics-packages-3.0.csv", delimiter=";", skip_header=True, dtype=None)

BUG_DATAFRAME_TRAIN = pd.read_csv("eclipse-metrics-packages-2.0.csv", delimiter=";")
BUG_DATAFRAME_TEST = pd.read_csv("eclipse-metrics-packages-3.0.csv", delimiter=";")
 
column_indices=[2]+[i for i in range(4, 44)]
BUG_DATA_X_TRAIN = BUG_DATAFRAME_TRAIN.iloc[:,column_indices].to_numpy()
BUG_DATA_Y_TRAIN = BUG_DATAFRAME_TRAIN.iloc[:,3].to_numpy()

BUG_DATA_X_TEST = BUG_DATAFRAME_TEST.iloc[:,column_indices].to_numpy()
BUG_DATA_Y_TEST = BUG_DATAFRAME_TEST.iloc[:,3].to_numpy

if __name__ == "__main__":
    array=np.array([1,0,1,1,1,0,0,1,1,0,1])
    
    print(BUG_DATA_X_TRAIN.shape)
