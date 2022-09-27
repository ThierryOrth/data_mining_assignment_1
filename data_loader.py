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

column_indices = [2]
column_indices+=[i for i in range(4, 44)]
BUG_DATA_Y_TRAIN = BUG_DATAFRAME_TRAIN.loc("post")
BUG_DATA_Y_TEST = BUG_DATAFRAME_TEST.loc("post")
# BUG_DATA_X_TRAIN = 
# BUG_DATA_X_TEST =


# for idx, column in enumerate(columns):
#     print(idx, column)



#ataFrame.to_numpy(


44,210

df.drop(df.columns[cols],axis=1,inplace=True)
#df.drop(df.columns[i], axis=1)
#BUG_DATA_TRAIN.drop()

# POST BUGS AS TARGET

# PRE BUGS AS FEATURE



BUG_DATA_TEST = pd.read_csv("eclipse-metrics-packages-2.0.csv", delimiter=";")

#BOTH CONSIST OF TUPLES FOR EACH FEATURE

if __name__ == "__main__":
    array=np.array([1,0,1,1,1,0,0,1,1,0,1])
    
    print(BUG_DATA_TRAIN.shape)

