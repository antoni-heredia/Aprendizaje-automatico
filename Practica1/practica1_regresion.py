#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 14:50:35 2019

@author: Antonio Jesus Heredia Castillo
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

#funcion para calcular la pseudoinversa
def calcular_pseudoInversa(_matrix):
    return np.linalg.inv(_matrix.T @ _matrix)  @ _matrix .T

#funcion para cambiar el intervalo en el que se encuentra un target
def cambiar_intervalo_target(_target, _inicio, _final):
    _target = _target.astype(np.float64)
    _valores = np.unique(_target)
    _valores = np.sort(_valores)
    linealspace = np.linspace(_inicio,_final, _valores.size)
    for ix, xf in enumerate(linealspace):
        _target[_target == _valores[ix]] = xf
    return _target

#funcion para calcular el error
def error(_X, _y, w):
    _wt = w.T
    _Xt = _X.T
    _yt = _y.T
    _N = _y.shape[0]
    return (1/_N)*( _wt @ _Xt @ _X @ w - 2* _yt @ _X @ w + _yt @ _y )

def gradiente_estocastico(_X_train, _y_train,_minibatch_size=None, _lr=0.01, _iteraciones=1000, _epsilon=0.08):
    _w = np.zeros([3])
    _N = _y_train.shape[0]

    _gradiente = lambda x, _X, _y: (2/_N)*( _X.T @ ( _X  @ x)- _X.T@ _y)
    _error =  error(_X_train, _y_train,_w)
    _iter = 0
    #en el caso de que el usuario no introduzca un tamaño lo calculo yo
    if _minibatch_size == None:
        _minibatch_size = np.int(np.ceil(np.log2(len(_X_train))*5))

    #mientras el error no sea menor a lo que yo quiero o no se alcance el numero maximo de iteraciones
    while _error >_epsilon and _iter < _iteraciones:

        #realizo la mezcla de los datos
        _batch, _y_batch = shuffle(_X_train, _y_train)

        #cojo minibatchs hasta que se acaben
        while _batch.shape[0] != 0:
            #cojo un minibatch y sus etiquetas correspondientess
            _minibatch = _batch[:_minibatch_size,:]
            _y_mini = _y_batch[:_minibatch_size]

            #elemino ese minibatch para no coger en la siguiente iteracion
            _batch = np.delete(_batch,np.s_[:_minibatch_size:], axis = 0)
            _y_batch = np.delete(_y_batch,np.s_[:_minibatch_size:])

            #calculo el gradiente
            _w = _w - _lr * _gradiente(_w, _minibatch, _y_mini)
        #aumento el numero de iteraciones
        _iter =_iter+1
    #calculo el error de este batch
    _error = error(_X_train, _y_train,_w)
    #devuelvo el mejor _w y el error conseguido con ese _w
    return _w, _error

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))


def experimento(iteracioens):
	_suma_error = 0
	_suma_error_out = 0
	_RUIDO = 0.1
	_N = 1000
	#realizo el gradiente tantas veces como me digan
	f = lambda x,y: np.sign((x-0.2)*(x-0.2) + y*y - 0.6)

	for x in range(0, iteracioens):
		#genero los puntos
		_X = simula_unif(_N,2,-1)
		_y_puntos = np.empty(_N)
		#le añado ruido a los puntos
		for _ix,_xf in enumerate(_X):
			_y_puntos[_ix] = f(_xf[0],_xf[1])
			if _ix % int(100*_RUIDO) == 0:
				_y_puntos[_ix] = -_y_puntos[_ix]
		#añado variable independiente
		_X = np.insert(_X,0,1,axis=1)
		#genero el gradiente y sumo el error a la variable para luego obtener la media
		_W, error_in = gradiente_estocastico(_X, _y_puntos,_iteraciones=50, _minibatch_size=64)
		_suma_error += error_in

		#creuo un conjunto de datos como test
		_X_tes = simula_unif(_N,2,-1)
		_y_tes = np.empty(1000)
		for ix, xf in enumerate(_X_tes):
			_y_tes[ix] = f(xf[0],xf[1])
		_X_tes = np.insert(_X_tes,0,1,axis=1)
		#obtengo el error que tiene el test desde el modelo
		_suma_error_out += error(_X_tes,_y_tes,_W)

	return _suma_error/_N, _suma_error_out/_N


#Cargo los datos tanto de test como de entrenamiento
X_train = np.load("datos/X_train.npy")
y_train = np.load("datos/y_train.npy")
X_test = np.load("datos/X_test.npy")
y_test = np.load("datos/y_test.npy")

#clasifico los datos quedandome solo del 1 al 5
X_train = X_train[(y_train == 1) | (y_train == 5)]
y_train = y_train[(y_train== 1) | (y_train == 5)]
X_test = X_test[(y_test== 1) | (y_test == 5)]
y_test = y_test[(y_test== 1) | (y_test == 5)]

#cambio el intervalo del 1 al 5 a -1 al 1
y_train = cambiar_intervalo_target(y_train,-1.,1.)
y_test = cambiar_intervalo_target(y_test,-1.,1.)
#añado el valor para el w0
X_train = np.insert(X_train,0,1,axis=1)
X_test = np.insert(X_test,0,1,axis=1)

#calculo el valor de w con la pseudoinversa
wi =  calcular_pseudoInversa(X_train) @ y_train

#calculo la wi con el descenso del gradiente estocastico
w_sgd, error_sgd = gradiente_estocastico(X_train,y_train,64)

print("----------------------Ejercicio 1----------------------")
print("------Errores Pseudoinversa-------")
print("El E_in es:" + str(error(X_train,y_train,wi)))
print("El E_our es:" + str(error(X_test,y_test,wi)))
print("------Errores SGD-------")
print("El E_in es:" + str(error_sgd))
print("El E_our es:" + str(error(X_test,y_test,w_sgd)))




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
plt.plot(t,-w_sgd[0]/w_sgd[2] - w_sgd[1]/w_sgd[2]*t, label="SGD")

plt.legend(numpoints=1)
plt.show()

input("Pulsa  Enter para continuar al siguiente apartado...")
print("----------------------Ejercicio 2----------------------")
#variables globales para la configuracion
N=1000
ruido = 0.1
#creo los puntos, la funcion y el array del target
X_puntos = simula_unif(N,2,-1)
f = lambda x,y: np.sign(((x-0.2)*(x-0.2)) + y*y - 0.6)
y_puntos = np.empty(1000)

#relleno el array del target
for ix, xf in enumerate(X_puntos):
	y_puntos[ix] = f(xf[0],xf[1])
	if ix % int(100*ruido) == 0:
		y_puntos[ix] = -y_puntos[ix]

#meto la variable independiente
X_puntos = np.insert(X_puntos,0,1,axis=1)

print("----------------------Muestro los puntos conseguido----------------------")


plt.figure(2)
plt.title('Ejercicio 2')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(X_puntos[(y_puntos== 1),1],X_puntos[(y_puntos== 1),2], 'ro', c='blue', alpha=0.6, label="1")
plt.plot(X_puntos[(y_puntos== -1),1],X_puntos[(y_puntos== -1),2], 'ro', c='red', alpha=0.6, label="-1")
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.legend(numpoints=1)
plt.show()


#genero el gradiente para esa entrada
w_ej2, error_eje2 = gradiente_estocastico(X_puntos, y_puntos,_iteraciones=50, _minibatch_size=64)

plt.figure(3)
plt.title('Modelo para el ejercicio 2s')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(X_puntos[(y_puntos== 1),1],X_puntos[(y_puntos== 1),2], 'ro', c='blue', alpha=0.6, label="1")
plt.plot(X_puntos[(y_puntos== -1),1],X_puntos[(y_puntos== -1),2], 'ro', c='red', alpha=0.6, label="-1")
plt.plot( [-1,1],[(-w_ej2[0]+w_ej2[1])/w_ej2[2],(-w_ej2[0]-w_ej2[1])/w_ej2[2]],'k-',c='yellow',label="modelo")
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.legend(numpoints=1)
plt.show()
#muestro los datos que se piden
print("-----------Apartado C-----------")
print("El vector de pesos w es: "+str(w_ej2))
print("El err_in del modelo es: "+str(error_eje2))

#realizo el experimento y muestro lo obtenido
media_in, medio_out = experimento(N)
print("El valor medio para el Ein en las"+str(N)+" iteraciones es de: "+str(media_in))
print("El valor medio para el Eout en las"+str(N)+" iteraciones es de: "+str(medio_out))

