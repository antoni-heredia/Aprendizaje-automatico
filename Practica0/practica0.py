#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:32:39 2019

@author: antonio
"""
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, 2:]
y = iris.target
y_axis  = X[:,:1]
x_axis = X[:,1:]

# recorro todos las plantas y segun su tipo, pongo un color y una etiqueta
for pos in range(0,y.size):
    if(y[pos] == 0):
        plt.scatter(x_axis[pos], y_axis[pos], color='blue', label="Clase 0" )
    elif (y[pos] == 1):
        plt.scatter(x_axis[pos], y_axis[pos], color='red', label="Clase 1")
    else:
        plt.scatter(x_axis[pos], y_axis[pos], color='green', label="Clase 2")

#codigo para no repetir las etiquetas
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()

