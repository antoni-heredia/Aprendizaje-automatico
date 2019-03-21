#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 14:50:35 2019

@author: Antonio Jesus Heredia Castillo
"""
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import pandas as pd
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 
from sklearn import linear_model
from sklearn.utils import shuffle

def calcular_pseudoInversa(_matrix):
    return np.linalg.inv(_matrix.T @ _matrix)  @ _matrix .T

def cambiar_intervalo_target(_target, _inicio, _final):
    _target = _target.astype(np.float64)
    _valores = np.unique(_target)
    _valores = np.sort(_valores)
    linealspace = np.linspace(_inicio,_final, _valores.size)
    for ix, xf in enumerate(linealspace):
        _target[_target == _valores[ix]] = xf 
    return _target

def errin(_X_train, _y_train, w):
    _wt = w.T
    _Xt = _X_train.T
    _yt = _y_train.T
    _N = _y_train.shape[0]
    return (1/_N)*( _wt @ _Xt @ _X_train - 2* _wt @ _Xt @ _y_train + _yt @ _y_train )

def gradiente_estocastico(_X_train, _y_train, _lr=0.01,_iteraciones=1000, _epsilon=1e-10):
    _w = np.zeros([3])
    N = _y_train.shape[0]

    _gradiente = lambda x, _X, _y: (2/N)*( _X.T @ ( _X  @ x)- _X.T@ _y)
    error = errin(_X_train, _y_train,_w)
    while error[0] > _epsilon:
        minibatch, y_mini = shuffle(_X_train, _y_train, n_samples=64)
        _iter = 0
        while True:

            #si hay mas iteraicones de las que yo quiero paro
            if _iter >= _iteraciones:
                break;

            #calculo el gradiente
            _w = _w - _lr * _gradiente(_w, minibatch, y_mini)


            _iter =_iter+1
        error = errin(_X_train, _y_train,_w)

    return _w

#Cargo los datos tanto de test como de entrenamiento
X_train = np.load("datos/X_train.npy")
y_train = np.load("datos/y_train.npy")
X_test = np.load("datos/X_test.npy")
y_test = np.load("datos/y_test.npy")

#clasifico los datos quedandome solo del 1 al 5
X_train = X_train[(y_train== 1) | (y_train == 5)]
y_train = y_train[(y_train== 1) | (y_train == 5)]
X_test = X_test[(y_test== 1) | (y_test == 5)]
y_test = y_test[(y_test== 1) | (y_test == 5)]

#cambio el intervalo del 1 al 5 a -1 al 1
y_train = cambiar_intervalo_target(y_train,-1.,1.)
#y_test = cambiar_intervalo_target(y_test,-1.,1.)
X_train = np.insert(X_train,0,1,axis=1)

#calculo el valor de w con la pseudoinversa
wi =  calcular_pseudoInversa(X_train) @ y_train

#calculo la wi con el descenso del gradiente estocastico
clf = linear_model.SGDRegressor(shuffle=True,tol=1e-10, max_iter=1000)
clf.fit(X_train, y_train)
sgd_scikit = clf.coef_

sgd_mio = gradiente_estocastico(X_train,y_train)



 #Mostrar datos
plt.figure(1)
plt.title('Ejercicio 1')
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetria')
max_val = 1
t = np.arange(0., max_val+0.5,0.5)
plt.plot(X_train[(y_train== 1),1],X_train[(y_train== 1),2], 'ro', c='blue', alpha=0.6, label="5")
plt.plot(X_train[(y_train == -1),1],X_train[(y_train == -1),2], 'ro', c='red', alpha=0.6, label="1")
plt.plot(t,-wi[0]/wi[2]- wi[1]/wi[2]*t, label="PseudoInversa")
plt.plot(t,-sgd_scikit[0]/sgd_scikit[2] - sgd_scikit[1]/sgd_scikit[2]*t, label="SGD")
plt.plot(t,-sgd_mio[0]/sgd_mio[2] - sgd_mio[1]/sgd_mio[2]*t, label="Mi SGD")

plt.legend(numpoints=1)
plt.show()
"""

fig = plt.figure()
ax = Axes3D(fig)
# Creamos una malla, sobre la cual graficaremos el plano
xx, yy = np.meshgrid(np.linspace(0, 0.8, num=10), np.linspace(-8, 0, num=10))

# calculamos los valores del plano para los puntos x e y
nuevoX = (wi[1] * xx)
nuevoY = (wi[2] * yy) 

# calculamos los correspondientes valores para z. Debemos sumar el punto de intercepción
z = (nuevoX + nuevoY + wi[0])

# Graficamos el plano
ax.plot_surface(xx, yy, z, alpha=0.2, cmap='hot')

# Graficamos en azul los puntos en 3D
ax.scatter(X_train[:,1],X_train[:,2],y_train, c='r', alpha=0.05, marker='o')



# con esto situamos la "camara" con la que visualizamos
ax.view_init(elev=30., azim=65)
        

ax.set_xlabel('Cantidad de Palabras')
ax.set_ylabel('Cantidad de Enlaces,Comentarios e Imagenes')
ax.set_zlabel('Compartido en Redes')

ax.set_title('Regresión Lineal con Múltiples Variables')
"""