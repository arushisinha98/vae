#######################################################################
## USE !pip install INSTEAD OF IMPORT FOR UNSATISTIFIED REQUIREMENTS ##
#######################################################################

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

original_dim = 128
beta = 0.2
batch_size = 200
latent_dim = 4 # 4 latent variables (per Moseley, et al. (2020))

from tensorflow.keras.layers import BatchNormalization

##############################################
## NEED TO EDIT TO POINT TO T-BOL DIRECTORY ##
##############################################

import os
os.chdir("/u/paige/asinha/T-BOL/")

# object-oritented version
# https://wwww.tensorflow.org/guide/keras/custom_layers_and_models#setup

# another source
# https://keras.io/examples/generative/vae/

class Sampling(layers.Layer):
    # uses (z_mean, z_log_var) to sample z, the vector encoding an input array

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape = (batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

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

"""
TIME ESTIMATE FOR THIS CELL OF CODE: 16 MINUTES
"""

import os
os.chdir("/u/paige/asinha/projectdir/")

with open('GPR-padded-dump.csv') as file_name:
    x_train_dump = np.loadtxt(file_name, delimiter = " ")
    
x_train2 = np.asarray(x_train_dump)
print(x_train2.shape)

x_raw = []
# assumes each file has the same number of pixels
for ii in range(len(data[0][0])):
    Y = [data[jj][2][ii] for jj in range(0, len(data))]
    x_raw.append(np.array(Y))

## find the second derivative and output relevant values

dx = []
d2x = []
d3x = []

for ii in range(0,len(x_train2)):
    x = x_train2[ii]
    dx.append([(x[jj+1] - x[jj])/12 for jj in range(len(x)-1)])
    d2x.append([(dx[ii][kk+1] - dx[ii][kk])/12 for kk in range(len(x)-2)])

"""
0 1 2 3 4 5 6 7 8 9 10 … 124 125 126 127 = x
  0 1 2 3 4 5 6 7 8 9 10 … 124 125 126 = dx
    0 1 2 3 4 5 6 7 8 9 10 … 124 125 = d2x

in the ideal case:
FIND MAX(d2x) IN [0, 125/2]
FIND MIN(d2x)
FIND MAX(d2x) in [125/2, 125]

in our case:
FIND MAX(d2x) IN [0, 42]
FIND MIN(d2x)
FIND MAX(d2x) IN [83, 125]

"""

t_imp = []
imp = []

for ii in range(0,len(x_train2)):
    imp.append(np.argmax(d2x[ii][0:int(len(d2x[ii])/3)]))
    imp.append(np.argmin(d2x[ii]))
    imp.append(int(2*len(d2x[ii])/3) + np.argmax(d2x[ii][int(2*len(d2x[ii])/3):int(len(d2x[ii]))]))
    t_imp.append(imp)

import os
os.chdir("/u/paige/asinha/projectdir/")
dx_dump = np.asarray(dx)
np.savetxt('dx.csv', dx_dump, fmt = '%1.3f')
d2x_dump = np.asarray(d2x)
np.savetxt('d2x.csv', d2x_dump, fmt = '%1.3f')
t_imp_dump = np.asarray(t_imp)
np.savetxt('t_imp.csv', t_imp_dump, fmt = '%1.3f')
