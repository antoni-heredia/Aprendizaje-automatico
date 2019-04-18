# -*- coding: utf-8 -*-
"""
TRABAJO 2.
Nombre Estudiante: Antonio Jesus Heredia Castillo
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import math


# Fijamos la semilla
np.random.seed(1)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0
    out = np.zeros((N,dim),np.float64)
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para
        # la primera columna se usará una N(0,sqrt(5)) y para la segunda N(0,sqrt(7))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)

    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.

    return a, b
    

def ajusta_PLA(datos, label, max_iter, vini):
    _w = vini
    _X =  np.insert(datos,0,1,axis=1)
    _iteraciones = 0
    _cant_cambios = 1
    while _iteraciones < max_iter and _cant_cambios != 0:
        _cant_cambios = 0
        _iteraciones+= 1
        for _ix, xf in enumerate(_X):
            if np.sign(_w.T @ xf) != label[_ix]:
                _w = _w+label[_ix]*xf
                _cant_cambios += 1
    return _w,_iteraciones

def grandienteSumatorio(_X, _y, _w):
    sumatorio = 0
    for xi in range(0,_y.size):
        sumatorio += -_y[xi]*_X[xi]*( (math.exp( -_y[xi]*_w.T@_X[xi])) / (1+math.exp( -_y[xi]*_w.T@_X[xi])) )
    return (1/_y.size)*sumatorio

def gradiente_estocastico(_X_train, _y_train,_minibatch_size=None, _lr=0.01, _iteraciones=1000, _epsilon=0.01):
    w = np.zeros([3])

    _iter = 0
    #en el caso de que el usuario no introduzca un tamaño lo calculo yo
    if _minibatch_size == None:
        _minibatch_size = np.int(np.ceil(np.log2(len(_X_train))*5))
    w_anterior = w+1
    #mientras el error no sea menor a lo que yo quiero o no se alcance el numero maximo de iteraciones
    while  np.linalg.norm(w_anterior-w) >_epsilon and _iter < _iteraciones:

        #realizo la mezcla de los datos
        _batch, _y_batch = shuffle(_X_train, _y_train)
        w_anterior = np.copy(w)

        #cojo minibatchs hasta que se acaben
        while _batch.shape[0] != 0:
            #cojo un minibatch y sus etiquetas correspondientess
            _minibatch = _batch[:_minibatch_size,:]
            _y_mini = _y_batch[:_minibatch_size]

            #elemino ese minibatch para no coger en la siguiente iteracion
            _batch = np.delete(_batch,np.s_[:_minibatch_size:], axis = 0)
            _y_batch = np.delete(_y_batch,np.s_[:_minibatch_size:])
            #calculo el gradiente
            w = w - _lr * grandienteSumatorio(_minibatch, _y_mini,w)
        #aumento el numero de iteraciones
        _iter =_iter+1
    #calculo el error de este batch
    #devuelvo el mejor _w y el error conseguido con ese _w
    return  w


print("-----------------Modelos Lineales-----------------")
print("-----------------Ejercicio 1----------------")
N=50
rango = [-50,50]
puntos_uni = simula_unif(N,2,rango)

intervalo = [[-50,-50],[50,50]]
v = simula_recta(intervalo)
f = lambda x, y,: y-v[0]*x-v[1]
X = np.copy(puntos_uni)
y = f(X[:,0],X[:,1])
x = np.linspace(-50,50,2)
y[(y>0)] = 1
y[(y<0)] = -1

pesos = np.zeros((X[0].size+1))
pesos,iteracionesSIN = ajusta_PLA(X,y,1000,pesos)
print("------Apartado a------")
print("Cantidad de iteraciones inicializado a 0: "+str(iteracionesSIN))
A = (-(pesos[0] / pesos[2]) / (pesos[0] / pesos[1]))
B =  (-pesos[0] / pesos[2])
fig2, (ax1,ax2) = plt.subplots(nrows=1, ncols=2)
fig2.suptitle("Ejercicio 1-Perceptron-Apartado A")
ax1.set_title("Inicializacion fija")
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.plot(X[(y>0),0],X[(y>0),1],'ro',c='red', label="positivos")
ax1.plot(X[(y<0),0],X[(y<0),1],'ro',c='blue', label="negativos")
ax1.plot(x, A*x+B,'--m', label='recta')

iteracionesSumatoria = 0
pesos = 0
for _xi in range(10):
    pesos = np.random.randint(2, size=X[0].size+1)
    pesosNuevos,iteracionesSIN = ajusta_PLA(X,y,1000,pesos)
    iteracionesSumatoria += iteracionesSIN


print("La media de iteraciones con inializacion aleatoria es: "+str(iteracionesSumatoria/10))
A = (-(pesosNuevos[0] / pesosNuevos[2]) / (pesosNuevos[0] / pesosNuevos[1]))
B =  (-pesosNuevos[0] / pesosNuevos[2])
ax2.set_title("Inicializacion aleatoria")
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.plot(X[(y>0),0],X[(y>0),1],'ro',c='red', label="positivos")
ax2.plot(X[(y<0),0],X[(y<0),1],'ro',c='blue', label="negativos")
ax2.plot(x, A*x+B,'--m', label='recta')

#lo mezclo aleatoriamente para cambiar las etiqueas de forma aleatoria
X, y = shuffle(X, y)
cantpos = int(y[(y>0)].size)*0.1
cantneg = int(y[(y<0)].size)*0.1
for index, _x in np.ndenumerate(y):
	if(_x>0 and cantpos > 0):
		cantpos-= 1
		y[index] = -y[index]
	elif(_x<0 and cantneg > 0):
		cantneg-= 1
		y[index] = -y[index]

pesos = np.zeros((X[0].size+1))
pesos,iteracionesCON = ajusta_PLA(X,y,1000,pesos)
print("------Apartado b------")
print("Cantidad de iteraciones inicializado a 0: "+str(iteracionesSIN))

fig2, (ax1,ax2) = plt.subplots(nrows=1, ncols=2)
fig2.suptitle("Ejercicio 1-Perceptron-Apartado B")
ax1.set_title("Inicializacion fija")
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.plot(X[(y>0),0],X[(y>0),1],'ro',c='red', label="positivos")
ax1.plot(X[(y<0),0],X[(y<0),1],'ro',c='blue', label="negativos")
ax1.plot(x, A*x+B,'--m', label='recta')

iteracionesSumatoria = 0
pesos = 0
for _xi in range(10):
    pesos = np.random.randint(2, size=X[0].size+1)
    pesosNuevos,iteracionesSIN = ajusta_PLA(X,y,1000,pesos)
    iteracionesSumatoria += iteracionesSIN


print("La media de iteraciones con inializacion aleatoria es: "+str(iteracionesSumatoria/10))
A = (-(pesosNuevos[0] / pesosNuevos[2]) / (pesosNuevos[0] / pesosNuevos[1]))
B =  (-pesosNuevos[0] / pesosNuevos[2])
ax2.set_title("Inicializacion aleatoria")
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.plot(X[(y>0),0],X[(y>0),1],'ro',c='red', label="positivos")
ax2.plot(X[(y<0),0],X[(y<0),1],'ro',c='blue', label="negativos")
ax2.plot(x, A*x+B,'--m', label='recta')


print("-----------------Ejercicio 2----------------")

datos = simula_unif(100,2,(0,2))
v = simula_recta((0,2))
x = np.linspace(-0,2,2)
f = lambda x, y,: y-v[0]*x-v[1]
y = f(datos[:,0],datos[:,1])
y[(y>0)] = 1
y[(y<0)] = -1



_X =  np.insert(datos,0,1,axis=1)
pesos = gradiente_estocastico(_X,y,1)

A = (-(pesos[0] / pesos[2]) / (pesos[0] / pesos[1]))
B =  (-pesos[0] / pesos[2])

fig2, ax = plt.subplots(nrows=1, ncols=1)
fig2.suptitle("Ejercicio 2")

ax.set_title("Sin ruido")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.plot(x, v[0]*x+v[1] ,'--m', label='recta')
ax.plot(datos[(y>0),0],datos[(y>0),1],'ro',c='red', label="positivos")
ax.plot(datos[(y<0),0],datos[(y<0),1],'ro',c='blue', label="negativos")

x = np.linspace(-0,2,2)
ax.plot(x, A*x+B,'--m',c='yellow', label='sgd')

###############################################################################
###############################################################################
###############################################################################
###############################################################################
print('EJERCICIO BONUS\n')

label4 = 1
label8 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	# Solo guardamos los datos cuya clase sea la 4 o la 8
	for i in range(0,datay.size):
		if datay[i] == 4 or datay[i] == 8:
			if datay[i] == 4:
				y.append(label4)
			else:
				y.append(label8)
			x.append(np.array([1, datax[i][0], datax[i][1]]))

	x = np.array(x, np.float64)
	y = np.array(y, np.float64)

	return x, y