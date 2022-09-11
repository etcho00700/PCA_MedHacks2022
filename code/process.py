import pdb

import numpy
import pandas as pd
import os
import glob
import xlrd
import openpyxl
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler


ECGIDPath= "../data/test_data/Diagnostics.csv"
ECGIDDataDenoised = "../data/test_data/ECGDataDenoised"

RhythmNameDict = {
    "AFIB": [1,0,0,0,0,0,0,0,0,0,0,0],
    "SVT": [1,0,0,0,0,0,0,0,0,0,0,0],
    "SR": [0,0,1,0,0,0,0,0,0,0,0,0],
    "AT":[0,1,0,0,0,0,0,0,0,0,0,0],
    "ST":[0,1,0,0,0,0,0,0,0,0,0,0],
}

def dataProcess():
    ecgID = []
    Dataset = []
    Label = []

    f = open('../data/test_data/SignalLabel.csv', 'w')
    writer = csv.writer(f)


    with open("../data/test_data/Diagnostics.csv", 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            #index 0 == file name index 1 == rhythm type
            if row[0][7].isdigit():
                if row[1] in RhythmNameDict:
                    ecgID.append([row[0],RhythmNameDict.get(row[1])]);

    for file in os.listdir(ECGIDDataDenoised):
        raw = pd.read_csv("../data/test_data/ECGDataDenoised/" + file)
        raw = raw.to_numpy()
        pad = np.zeros([1,12])
        raw = np.vstack((raw, pad))
        for i in range(len(ecgID)):
            if ecgID[i][0] == file[0:len(file)-4]:
                Dataset.append(raw);
                Label.append(ecgID[i][1])

    #scaler = StandardScaler()
    #scaler.fit(np.vstack(Dataset).flatten()[:, np.newaxis].astype(float))

    return Dataset, Label







