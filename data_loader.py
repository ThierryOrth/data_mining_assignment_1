import numpy as np
import os

DIR = os.getcwd()
FILENAME = "CREDIT.TXT"
CREDIT_DATA = np.genfromtxt(DIR+"/"+FILENAME, delimiter=",", skip_header=True)
ATTRIBUTES = {0:"age", 1:"married", 2:"house", 3:"income", 4:"gender", 5:"class"}

if __name__ == "__main__":
    array=np.array([1,0,1,1,1,0,0,1,1,0,1])
    
