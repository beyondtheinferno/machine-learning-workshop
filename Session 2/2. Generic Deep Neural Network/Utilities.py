import cv2
import datetime as dt
import h5py
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import numpy as np
import os
import pandas as pd
from glob import glob

def LoadDataset(fileName:str):
    '''
    Imports the dataset and returns XTrain, XTest, YTrain, YTest
    
    Returns:
        XTrain --> np.array() : Train data image of shape (,64,64,3)
        XTest --> np.array(): Test data image of shape (,64,64,3)
        YTrain --> np.array(): Train data label of shape (,1)
        YTest --> np.array(): Test data label of shape (,1)
    '''
    XTrain = []
    YTrain = []
    XTest = []
    YTest = []
    with h5py.File(fileName, 'r') as hf:
        for i in hf['TrainingSet']["Images"].keys():
            index = str(i).split("_")[-1]
            XTrain.append(hf['TrainingSet']["Images"][i].value)
            YTrain.append(hf['TrainingSet']["Labels"]['Label_' + str(index)].value)
        for i in hf['TestSet']["Images"].keys():
            index = str(i).split("_")[-1]
            XTest.append(hf['TestSet']["Images"][i].value)
            YTest.append(hf['TestSet']["Labels"]["Label_" + str(index)].value)
    
    return np.array(XTrain), np.array(XTest), np.array(YTrain), np.array(YTest)