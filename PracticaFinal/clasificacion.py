# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

import matplotlib.pyplot as plt
from sklearn import linear_model

def leerArchivo(ruta):
    f = open(ruta)
    data = np.genfromtxt(f, delimiter=",")
    return data[:,:-1],data[:,-1:]

def confusionMatrix(prediccion, y_test, titulo):
    cm = metrics.confusion_matrix(y_test, prediccion)
    
    plt.figure(figsize=(9,9))
    plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
    plt.title(titulo, size = 15)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], rotation=45, size = 10)
    plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size = 10)
    plt.tight_layout()
    plt.ylabel('Etiquetas reales', size = 15)
    plt.xlabel('Etiquetas predecidas', size = 15)
    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
            horizontalalignment='center',
            verticalalignment='center')
    plt.show()



X_test,y_test = leerArchivo("datos/pendigits.tes")
X_train,y_train = leerArchivo("datos/pendigits.tra")


#----------------Preprocesado de datos
#Normalizo los datos

#Junto el train y test para poder normalizar los valores de forma correcta
X = np.concatenate((X_test, X_train))

min_max_scaler = preprocessing.MinMaxScaler()

X = min_max_scaler.fit_transform(X)

#Elimina las columnas que no aportan nada
selector = VarianceThreshold()
X = selector.fit_transform(X)
poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(X)
#Una vez normalizados vuelvo a dejar el mismo grupo de train y test
X_test = X[:X_test.shape[0],:]
X_train = X[X_test.shape[0]:,:]

#-------------Eleccion de clase de funciones


model = linear_model.LogisticRegression( penalty='l2', multi_class='ovr', C=1)
model.fit(X_train,y_train.ravel())
print("---------Regresion logistica---------")

prediccion = model.predict(X_test)
confusionMatrix(prediccion, y_test, "CM de la Regresion Logistica")
print("E_in: "+str(model.score(X_train,y_train)))
print("E_out: "+str(model.score(X_test,y_test)))


print("---------Perceptron---------")
perceptron = linear_model.Perceptron(tol=0, penalty='l1')
perceptron.fit(X_train,y_train.ravel())
prediccion = perceptron.predict(X_test)
confusionMatrix(prediccion, y_test, "CM del Perceptron")

print("E_in: "+str(perceptron.score(X_train,y_train)))
print("E_out: "+str(perceptron.score(X_test,y_test)))