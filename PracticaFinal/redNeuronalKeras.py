#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:31:24 2019

@author: antonio
"""

# Crea tu primer MLP en Keras
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras.utils import to_categorical
import numpy as np
# Importing PCA 
# Importing standardscalar module  
from sklearn.preprocessing import StandardScaler 
  

# Fija las semillas aleatorias para la reproducibilidad
np.random.seed(7)
# carga los datos

def leerArchivo(ruta):
    f = open(ruta)
    data = np.genfromtxt(f, delimiter=",")
    return data[:,:-1],data[:,-1:]


n_classes = 10

X_test,y_test = leerArchivo("datos/pendigits.tra")
y_test = to_categorical(y_test, num_classes=None)


X,y = leerArchivo("datos/pendigits.tra")
y = to_categorical(y, num_classes=None)


  

# crea el modelo
model = Sequential()
model.add(Dense(24, input_dim=16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='sigmoid'))

# Compila el modelo
model.compile(loss=losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

# Ajusta el modelo
model.fit(X, y, epochs=150, batch_size=256)
# evalua el modelo
scores = model.evaluate(X, y)
print("En el train \n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

scores = model.evaluate(X_test, y_test)
print("En el test \n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
