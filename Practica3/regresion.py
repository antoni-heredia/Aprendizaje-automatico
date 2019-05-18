#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 23:59:29 2019

@author: antonio
"""

# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

def leerArchivo(ruta):
    f = open(ruta)
    data = np.genfromtxt(f, delimiter='\t')
    return data[:,:-1],data[:,-1:]






X,y = leerArchivo("datos/air/airfoil_self_noise.dat")


#----------------Preprocesado de datos
#Normalizo los datos

#Junto el train y test para poder normalizar los valores de forma correcta

min_max_scaler = preprocessing.MinMaxScaler()

X = min_max_scaler.fit_transform(X)

#Elimina las columnas que no aportan nada
selector = VarianceThreshold(threshold=0.01)
X = selector.fit_transform(X)

poly = PolynomialFeatures(degree=4)
X = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

modelo = linear_model.LinearRegression()
modelo.fit(X_train,y_train)
print("Ein: " + str(modelo.score(X_train,y_train)))
print("Eout: " + str(modelo.score(X_test,y_test)))