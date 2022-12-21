import math
import numpy as np
import pandas as pd
import csv
import keras
import tensorflow
from keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.gaussian_process import GaussianProcessRegressor
import scipy.optimize
from sklearn.gaussian_process.kernels import Matern

import GPy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from keras.wrappers.scikit_learn import KerasRegressor

import os
os.chdir("/u/paige/asinha/T-BOL/")

# extract Lacus Mortis temperature profiles
files = ['lacus_mortis-tb-' + str(format(num, '03')) + '.xyz' for num in range(1,241)]

def extract(file):
    X, Y, Z = [], [], []
    f = open(file, 'r')
    for row, line in enumerate(f):
        values = line.strip().split('\t')
        X.append(float(values[0]))
        Y.append(float(values[1]))
        Z.append(float(values[2]))
    data = [X, Y, Z]
    return data

data = [extract(fileX) for fileX in files]

X = [0.1*jj - 0.05 for jj in range(1,len(data)+1)]

x_fit = np.linspace(0, 24, 121)
x_fit = x_fit[0:-1] # removed hour 24 = hour 0 (added later with padding)
x_raw, x_train = [], []

# assumes each file has the same number of pixels
for ii in range(len(data[0][0])):
    Y = [data[jj][2][ii] for jj in range(0, len(data))]
    x_raw.append(np.array(Y))
    mask = [index for index, val in enumerate(Y) if not np.isnan(val)]
    x, y = [], []
    for m in mask:
        x.append(X[m])
        y.append(Y[m])
    kernel = GPy.kern.Matern32(1, lengthscale = 6.0, variance = 100.0)
    gpr = GPy.models.GPRegression(np.array(x).reshape(-1,1), np.array(y).reshape(-1,1), kernel)
    gpr.constrain_positive()
    gpr.optimize()
    y_pred = gpr.predict(x_fit.reshape(-1,1))
    x_train.append(np.transpose(y_pred[0]))
    x_train[ii] = x_train[ii][0]

def periodic_padding(array, pad):
    N = len(array)
    M = N + 2*pad
    output = np.zeros(M)
    for index in range(pad, N+pad):
        output[index] = array[index - pad]
    for index in range(0, pad):
        output[index] = array[index + N - pad]
        output[index + N + pad] = array[index]
    return output

x_train = np.asarray(x_train)
x_train2 = []
for ix, xx in enumerate(x_train):
    xx = periodic_padding(xx, 4)
    x_train2.append(xx.astype('float32'))
x_train2 = np.asarray(x_train2)

x_train_dump = np.asarray(x_train2)

import os
os.chdir("/u/paige/asinha/projectdir/")
np.savetxt('GPR-padded-dump.csv', x_train_dump, fmt = '%1.3f')
