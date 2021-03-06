#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:46:32 2019

@author: antonio
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.decomposition import PCA 


# Fija las semillas aleatorias para la reproducibilidad
np.random.seed(7)
# carga los datos

def leerArchivo(ruta):
    f = open(ruta)
    data = np.genfromtxt(f, delimiter=",")
    return data[:,:-1],data[:,-1:]


X_test,y_test = leerArchivo("datos/pendigits.tes")
X,y = leerArchivo("datos/pendigits.tra")

scaler = StandardScaler()  
# Don't cheat - fit only on training data
scaler.fit(X)  
X = scaler.transform(X)  
# apply same transformation to test data
X_test = scaler.transform(X_test)  





clf = MLPClassifier(solver='adam',activation="relu", hidden_layer_sizes=(3, 256), batch_size="auto", verbose=1)

clf.fit(X, y)

print("Acierto en train: " + str(clf.score(X,y)))
print("Acierto en test: " + str(clf.score(X_test,y_test)))