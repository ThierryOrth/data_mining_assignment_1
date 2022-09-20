import numpy as np
import os

DIR = os.getcwd()
FILENAME = "CREDIT.TXT"
CREDIT_DATA = np.genfromtxt(DIR+"/"+FILENAME, delimiter=",", skip_header=True)


if __name__ == "__main__":
    array=np.array([1,0,1,1,1,0,0,1,1,0,1])
    
