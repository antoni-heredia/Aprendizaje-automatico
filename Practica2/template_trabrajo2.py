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

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
fig.suptitle("Ejercicio 1")
ax1.plot(puntos_uni[:,0],puntos_uni[:,1],  'ro',c='blue', label="normal")
ax1.set_title('Normal')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

ax2.plot(puntos_gau[:,0],puntos_gau[:,1],'ro',c='red', label="normal")
ax2.set_title('Gaussiana')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

print("-----------------Ejercicio 2----------------")
f = lambda x, y,: y-v[0]*x-v[1]
X = np.copy(puntos_uni)
y = f(X[:,0],X[:,1])
x = np.linspace(-50,50,2)
fig2, (ax3,ax4) = plt.subplots(nrows=1, ncols=2)
fig2.suptitle("Ejercicio 2")
ax3.set_title("Sin ruido")
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.plot(X[(y>0),0],X[(y>0),1],'ro',c='red', label="positivos")
ax3.plot(X[(y<0),0],X[(y<0),1],'ro',c='blue', label="negativos")
ax3.plot(x, v[0]*x+v[1],'--m', label='y=2x-3')
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=3)
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



ax4.set_title("Con ruido")
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.plot(X[(y>0),0],X[(y>0),1],'ro',c='red', label="positivos")
ax4.plot(X[(y<0),0],X[(y<0),1],'ro',c='blue', label="negativos")
ax4.plot(x, v[0]*x+v[1],'--m', label='y=2x-3')
ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=3)

print("-----------------Ejercicio 3----------------")
print("-----------------Apartado A----------------")
x0 = np.arange(-50,50,0.2)
y0 = np.arange(-50,50,0.2)
x0 = np.linspace(-50, 50, 50)
y0 = np.linspace(-50, 50, 50)
x0, y0 = np.meshgrid(x0, y0)

fig3, ax = plt.subplots(2,2)
fig3.suptitle("Ejercicio 3")
ax[0,0].set_title("Primera funcion")
f = lambda x, y,: ((x-10)*(x-10))+((y-20)*(y-20))-400
funcion = f(x0,y0)
ax[0,0].set_xlabel('x')
ax[0,0].set_ylabel('y')
ax[0,0].plot(X[(y>0),0],X[(y>0),1],'ro',c='red', label="positivos")
ax[0,0].plot(X[(y<0),0],X[(y<0),1],'ro',c='blue', label="negativos")
ax[0,0].contour(x0,y0,funcion,[0])

ax[0,1].set_title("Segunda funcion funcion")
f = lambda x, y,: (0.5*(x+10)*(x+10))+((y-20)*(y-20))-400
funcion = f(x0,y0)
ax[0,1].set_xlabel('x')
ax[0,1].set_ylabel('y')
ax[0,1].plot(X[(y>0),0],X[(y>0),1],'ro',c='red', label="positivos")
ax[0,1].plot(X[(y<0),0],X[(y<0),1],'ro',c='blue', label="negativos")
ax[0,1].contour(x0,y0,funcion,[0])


ax[1,0].set_title("Tercera funcion funcion")
f = lambda x, y,: (0.5*(x-10)*(x-10))-((y+20)*(y+20))-400
funcion = f(x0,y0)
ax[1,0].set_xlabel('x')
ax[1,0].set_ylabel('y')
ax[1,0].plot(X[(y>0),0],X[(y>0),1],'ro',c='red', label="positivos")
ax[1,0].plot(X[(y<0),0],X[(y<0),1],'ro',c='blue', label="negativos")
ax[1,0].contour(x0,y0,funcion,[0])

ax[1,1].set_title("Cuarta funcion funcion")
f = lambda x, y,: y-(20*x*x)-(5*x)+3
funcion = f(x0,y0)
ax[1,1].set_xlabel('x')
ax[1,1].set_ylabel('y')
ax[1,1].plot(X[(y>0),0],X[(y>0),1],'ro',c='red', label="positivos")
ax[1,1].plot(X[(y<0),0],X[(y<0),1],'ro',c='blue', label="negativos")
ax[1,1].contour(x0,y0,funcion,[0])
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