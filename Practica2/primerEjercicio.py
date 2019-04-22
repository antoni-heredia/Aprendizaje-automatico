# -*- coding: utf-8 -*-
"""
TRABAJO 2.
Nombre Estudiante: Antonio Jesus Heredia Castillo
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle



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

print("-----------------Complejidad de H y ruido-----------------")
print("-----------------Ejercicio 1-----------------")
N=50
dim = 2
rango = [-50,50]
sigma = [5,7]
intervalo = [[-50,-50],[50,50]]

puntos_uni = simula_unif(N,dim,rango)

#simulo aqui la recta del ejercicio 2 de complejidad
#para tener la misma recta que en el ejercicio de modelos lineales
v = simula_recta(intervalo)

puntos_gau = simula_gaus(N,dim,sigma)

fig, ax1 = plt.subplots(nrows=1, ncols=1)
fig.suptitle("Ejercicio 1")
ax1.plot(puntos_uni[:,0],puntos_uni[:,1],  'ro',c='blue', label="normal")
ax1.set_title('Uniforme')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

fig, ax2 = plt.subplots(nrows=1, ncols=1)
ax2.plot(puntos_gau[:,0],puntos_gau[:,1],'ro',c='red', label="normal")
ax2.set_title('Gaussiana')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
input("Pulsa  Enter para continuar al siguiente apartado...")

print("-----------------Ejercicio 2----------------")
f = lambda x, y,: y-v[0]*x-v[1]
X = np.copy(puntos_uni)
y = f(X[:,0],X[:,1])
x = np.linspace(-50,50,2)
fig2, ax3 = plt.subplots(nrows=1, ncols=1)
fig2.suptitle("Ejercicio 2")
ax3.set_title("Sin ruido")
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.plot(X[(y>0),0],X[(y>0),1],'ro',c='red', label="positivos")
ax3.plot(X[(y<0),0],X[(y<0),1],'ro',c='blue', label="negativos")
ax3.plot(x, v[0]*x+v[1],'--m', label='y=2x-3')
ax3.legend( fancybox=True, shadow=True)
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


fig2, ax4 = plt.subplots(nrows=1, ncols=1)

ax4.set_title("Con ruido")
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.plot(X[(y>0),0],X[(y>0),1],'ro',c='red', label="positivos")
ax4.plot(X[(y<0),0],X[(y<0),1],'ro',c='blue', label="negativos")
ax4.plot(x, v[0]*x+v[1],'--m', label='y=2x-3')
ax4.legend( fancybox=True, shadow=True)
input("Pulsa  Enter para continuar al siguiente apartado...")

print("-----------------Ejercicio 3----------------")
print("-----------------Apartado A----------------")
x0 = np.arange(-50,50,0.2)
y0 = np.arange(-50,50,0.2)
x0 = np.linspace(-50, 50, 50)
y0 = np.linspace(-50, 50, 50)
x0, y0 = np.meshgrid(x0, y0)

fig3, ax = plt.subplots(1,1)
fig3.suptitle("Ejercicio 3")
ax.set_title("Primera funcion")
f = lambda x, y,: ((x-10)*(x-10))+((y-20)*(y-20))-400
funcion = f(x0,y0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.plot(X[(y>0),0],X[(y>0),1],'ro',c='red', label="positivos")
ax.plot(X[(y<0),0],X[(y<0),1],'ro',c='blue', label="negativos")
ax.contour(x0,y0,funcion,[0])
fig3, ax = plt.subplots(1,1)

ax.set_title("Segunda funcion funcion")
f = lambda x, y,: (0.5*(x+10)*(x+10))+((y-20)*(y-20))-400
funcion = f(x0,y0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.plot(X[(y>0),0],X[(y>0),1],'ro',c='red', label="positivos")
ax.plot(X[(y<0),0],X[(y<0),1],'ro',c='blue', label="negativos")
ax.contour(x0,y0,funcion,[0])

fig3, ax = plt.subplots(1,1)

ax.set_title("Tercera funcion funcion")
f = lambda x, y,: (0.5*(x-10)*(x-10))-((y+20)*(y+20))-400
funcion = f(x0,y0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.plot(X[(y>0),0],X[(y>0),1],'ro',c='red', label="positivos")
ax.plot(X[(y<0),0],X[(y<0),1],'ro',c='blue', label="negativos")
ax.contour(x0,y0,funcion,[0])
fig3, ax = plt.subplots(1,1)

ax.set_title("Cuarta funcion funcion")
f = lambda x, y,: y-(20*x*x)-(5*x)+3
funcion = f(x0,y0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.plot(X[(y>0),0],X[(y>0),1],'ro',c='red', label="positivos")
ax.plot(X[(y<0),0],X[(y<0),1],'ro',c='blue', label="negativos")
ax.contour(x0,y0,funcion,[0])
###############################################################################
###############################################################################
###############################################################################
###############################################################################
