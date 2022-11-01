#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 14:32:01 2022

@author: asinha
"""

import numpy as np

filename = 'diviner_learn_data_c7_processed_Xf_7_2000.npy'

data = np.load(filename)

data = np.asarray(data)
print(data.shape)

if data.ndim > 2:
	data2 = data[:, 0, :]
	print(data2.shape)

data_dump = np.asarray(data2)
np.savetxt('data-dump.csv', data_dump, fmt = '%1.3f')
