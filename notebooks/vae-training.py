import math
import numpy as np
import pandas as pd
import csv
import keras
import tensorflow
from keras import backend as K
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
import pickle
import datetime

original_dim = 128
beta = 0.2
batch_size = 200
latent_dim = 4 # 4 latent variables (per Moseley, et al. (2020))
epochs = 1000

from tensorflow.keras.layers import BatchNormalization

# confirm that TensorFlow is using GPU
sess = tf.Session(config = tf.ConfigProto(log_device_placement = True))
gpus = tf.config.list_physical_devices('GPU')
print("Number of GPUs Available: ", len(gpus))
K.tensorflow_backend._get_available_gpus()

# object-oritented version
# https://wwww.tensorflow.org/guide/keras/custom_layers_and_models#setup

# another source
# https://keras.io/examples/generative/vae/

os.chdir("/u/paige/asinha/projectdir/")
with open('padded-data-dump.csv') as file_name:
    data = np.loadtxt(file_name, delimiter = " ")
x_train = np.asarray(data)

## CREATE PROFILE DIRECTORY
timestamp = datetime.datetime.now().strftime("%Y%m%d")
os.mkdir("/u/paige/asinha/projectdir/model" + timestamp + "/")
    
class Sampling(layers.Layer):
    """ uses (z_mean, z_log_var) to sample z, the vector encoding an input array """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape = (batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
encoder_inputs = keras.Input(shape = (original_dim,))
x = tf.reshape(encoder_inputs, [-1, original_dim, 1])
x = layers.BatchNormalization()(x)
x = layers.Conv1D(filters = 32, kernel_size = 3, strides = 1, padding = "same", activation = "relu", name = "3e")(x)
x = layers.Conv1D(filters = 32, kernel_size = 2, strides = 2, padding = "same", activation = "relu", name = "4e")(x)
x = layers.Conv1D(filters = 32, kernel_size = 2, strides = 2, padding = "same", activation = "relu", name = "5e")(x)
x = layers.Conv1D(filters = 32, kernel_size = 2, strides = 2, padding = "same", activation = "relu", name = "6e")(x)
x = layers.Conv1D(filters = 32, kernel_size = 2, strides = 2, padding = "same", activation = "relu", name = "7e")(x)
x = layers.Conv1D(filters = 32, kernel_size = 2, strides = 2, padding = "same", activation = "relu", name = "8e")(x)
x = layers.Conv1D(filters = 32, kernel_size = 2, strides = 2, padding = "same", activation = "relu", name = "9e")(x)
x = layers.Conv1D(filters = 32, kernel_size = 2, strides = 2, padding = "same", activation = "relu", name = "10e")(x)
x = layers.Flatten()(x)
z_mean = layers.Dense(units = latent_dim, activation = None, name = "z_mean")(x)
z_log_var = layers.Dense(units = latent_dim, activation = None, name = "z_log_var")(x)
z = Sampling()((z_mean, z_log_var))
z = tf.reshape(z, [-1, 1, 4])
encoder = tf.keras.Model(inputs = encoder_inputs, outputs = z, name = "encoder")
encoder.summary()

latent_inputs = keras.Input(shape = (1, latent_dim))
x = layers.Conv1DTranspose(filters = 32, kernel_size = 2, strides = 2, padding = "same", activation = "relu", name = "1d")(latent_inputs)
x = layers.Conv1DTranspose(filters = 32, kernel_size = 2, strides = 2, padding = "same", activation = "relu", name = "2d")(x)
x = layers.Conv1DTranspose(filters = 32, kernel_size = 2, strides = 2, padding = "same", activation = "relu", name = "3d")(x)
x = layers.Conv1DTranspose(filters = 32, kernel_size = 2, strides = 2, padding = "same", activation = "relu", name = "4d")(x)
x = layers.Conv1DTranspose(filters = 32, kernel_size = 2, strides = 2, padding = "same", activation = "relu", name = "5d")(x)
x = layers.Conv1DTranspose(filters = 32, kernel_size = 2, strides = 2, padding = "same", activation = "relu", name = "6d")(x)
x = layers.Conv1DTranspose(filters = 32, kernel_size = 2, strides = 2, padding = "same", activation = "relu", name = "7d")(x)
outputs = layers.Conv1D(filters = 1, kernel_size = 1, strides = 1, padding = "same", activation = "relu", name = "8d")(x)
decoder = tf.keras.Model(inputs = latent_inputs, outputs = outputs, name = "decoder")
decoder.summary()

outputs = decoder(z)
autoencoder = tf.keras.Model(encoder_inputs, outputs)
# add KL divergence regularization loss
D_KL = beta * -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
autoencoder.add_loss(D_KL)
optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
autoencoder.compile(optimizer, loss = tf.keras.losses.MeanSquaredError())

scaler = MinMaxScaler()
x_norm = scaler.fit_transform(x_train)
x_vae = autoencoder.fit(x_train, x_train, epochs = epochs, batch_size = 200, validation_split = 0.2, shuffle = True) ## CHANGE EPOCHS
autoencoder.save('VAE-' + str(epochs) + 'e') ## CHANGE SAVED MODEL
decoder.save('Decoder-' + str(epochs) + 'e') ## CHANGE SAVED MODEL
encoder.save('Encoder-' + str(epochs) + 'e') ## CHANGE SAVED MODEL

X = [0.1*jj - 0.05 for jj in range(1, len(data)+1)]
x_fit = np.linspace(0, 24, 121)
x_fit = x_fit[0:-1] # removed hour 24 = hour 0 (added later with padding)
predictions = autoencoder.predict(x_train)
transformed = np.squeeze(predictions, axis = 2)
x_hat = scaler.inverse_transform(transformed)

os.mkdir("/u/paige/asinha/projectdir/model" + timestamp + "/profiles/")

z_Sample = []
for ix, xx in enumerate(x_train):
    fig = plt.figure(figsize = (5,5))
    plt.scatter(X, x_raw[ix])
    plt.plot(x_fit, xx[4:124], label = "GPR")
    x_predict = transformed[ix]
    plt.plot(x_fit, x_predict[4:124], label = "VAE")
    plt.legend()
    plt.ylim(0,400)
    plt.xlim(0,25)
    fig.savefig('x_test_row' + str(ix) + '.png')
    plt.close()
    z = encoder.predict(x_norm[ix:ix+1])
    z_Sample.append(z)
z_arrays = [z[0][0] for z in z_Sample]

os.chdir("/u/paige/asinha/projectdir/model" + timestamp + "/")

print("Mean latent values:")
print("z0: " + str(np.mean(z_arrays[0])))
print("z1: " + str(np.mean(z_arrays[1])))
print("z2: " + str(np.mean(z_arrays[2])))
print("z3: " + str(np.mean(z_arrays[3])))

fig = plt.figure(figsize = (5, 5))
vals = np.linspace(min(z_arrays[0]), max(z_arrays[0]), 6)
#vals = [0.5, 1, 1.5, 2]
vals = np.array(vals)
for iv, val in enumerate(vals):
    z = np.array([[[val, 0, 0, 0]]])
    prediction = decoder.predict(z)
    prediction[0].transpose()
    plt.plot(x_fit, prediction[0][4:124])
    plt.ylim(0,400)
    plt.xlim(0,25)
fig.legend()
fig.savefig("z0_exercise" + str(epochs) + ".jpg") ## ADD SUFFIX

fig = plt.figure(figsize = (5, 5))
vals = np.linspace(min(z_arrays[1]), max(z_arrays[1]), 6)
#vals = [-0.5, -1, -1.5, -2, -2.5]
vals = np.array(vals)
for iv, val in enumerate(vals):
    z = np.array([[[0, val, 0, 0]]])
    prediction = decoder.predict(z)
    prediction[0].transpose()
    plt.plot(x_fit, prediction[0][4:124])
    plt.ylim(0,400)
    plt.xlim(0,25)
fig.legend()
fig.savefig("z1_exercise" + str(epochs) + ".jpg") ## ADD SUFFIX

fig = plt.figure(figsize = (5, 5))
vals = np.linspace(min(z_arrays[2]), max(z_arrays[2]), 6)
#vals = [-1, -1.5, -2, -2.5]
vals = np.array(vals)
for iv, val in enumerate(vals):
    z = np.array([[[0, 0, val, 0]]])
    prediction = decoder.predict(z)
    prediction[0].transpose()
    plt.plot(x_fit, prediction[0][4:124])
    plt.ylim(0,400)
    plt.xlim(0,25)
fig.legend()
fig.savefig("z2_exercise" + str(epochs) + ".jpg") ## ADD SUFFIX

fig = plt.figure(figsize = (5, 5))
vals = np.linspace(min(z_arrays[3]), max(z_arrays[3]), 6)
#vals = [0, -0.5, -1, -1.5, -2, -2.5]
vals = np.array(vals)
for iv, val in enumerate(vals):
    z = np.array([[[0, 0, 0, val]]])
    prediction = decoder.predict(z)
    prediction[0].transpose()
    plt.plot(x_fit, prediction[0][4:124])
    plt.ylim(0,400)
    plt.xlim(0,25)
fig.legend()
fig.savefig("z3_exercise" + str(epochs) + ".jpg") ## ADD SUFFIX

os.chdir("/u/paige/asinha/projectdir/")
latent_dump = np.asarray(z_arrays)
np.savetxt('latent-values.csv', latent_dump, fmt = '%1.5f') ## CHANGE FILE NAME

os.chdir("/u/paige/asinha/projectdir/pickle_folder/")
with open(f'VAE_{epochs}e_{timestamp}.pickle', 'wb') as filename:
    pickle.dump(autoencoder, filename, protocol = pickle.HIGHEST_PROTOCOL)
