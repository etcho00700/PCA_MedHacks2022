import pdb

import sklearn.metrics

from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *
import torch
from models.fastai_model import fastai_model
from utils import utils
torch.device("cpu")
import random
import numpy as np
from process import dataProcess


def perf_metrics(y_actual, y_hat):
    tp = 0
    fn = 0

    for i in range(len(y_hat)):
        max = np.argmax(y_hat[i])
        if y_actual[i][max] == 1:
            tp += 1;
        else:
            fn += 1;


    # We find the True positive rate and False positive rate based on the threshold

    tpr = tp / (tp + fn)

    return tpr

def main():


    experiment = "exp3"
    modelname = 'fastai_xresnet1d101'
    pretrainedfolder = '../output/' +  '/models/' + modelname + '/'
    mpath = '../output/'  # <=== path where the finetuned model will be stored
    n_classes_pretrained = 12  # <=== because we load the model from exp0, this should be fixed because this depends the experiment


    sampling_frequency = 500
    datafolder = '../data/ptbxl/'
    task = 'rhythm'
    outputfolder = '../output/'
    #Rhythm -> Identifies arrythmia (irregular heart beat rhythm patterns)

    data, Y = dataProcess()
    data = np.array(data)
    Y = np.array(Y)
    data = np.nan_to_num(data)


    # Load PTB-XL data
    #rdata, raw_labels = utils.load_dataset(datafolder, sampling_frequency)
    # Preprocess label data
    #labels = utils.compute_label_aggregations(raw_labels, datafolder, task)
    # Select relevant data and convert to one-hot
    #rdata, labels, rY, _ = utils.select_data(rdata, labels, task, min_samples=0, outputfolder=outputfolder)
    #pdb.set_trace()
    #pdb.set_trace()

    # 1-9 for training
    X_train = data[0:4000]
    y_train = Y[0:4000]
    # 10 for validation
    X_val = data[4001:len(data)]
    y_val = Y[4001:len(Y)]

    num_classes = 12  # <=== number of classes in the finetuning dataset
    input_shape = [5000, 12]  # <=== shape of samples, [None, 12] in case of different lengths

    model = fastai_model(
        modelname,
        num_classes,
        sampling_frequency,
        mpath,
        input_shape=input_shape,
        pretrainedfolder=pretrainedfolder,
        n_classes_pretrained=n_classes_pretrained,
        pretrained=True,
        epochs_finetuning=5,
    )

    model.fit(X_train, y_train, X_val, y_val);
    y_val_pred = model.predict(X_val)
    results = perf_metrics(y_val, y_val_pred);
    print(results)

if __name__ == "__main__":
    main()
