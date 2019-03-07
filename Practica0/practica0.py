#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:32:39 2019

@author: Antonio Jesus Heredia Castillo
"""
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import random
import math
from sklearn import datasets
#cargo los datos
iris = datasets.load_iris()
#guardamos todos los datos, tanto las caracteristicas de la flor como 
#a la clase que pertenece 
#(mas tarde me di cuenta de la tonteria de guardarlo junto para luego separarlos)
array_datos = np.column_stack((iris.data[:, 2:],iris.target))
#inicializo variables que voy a usar
y_axis  = array_datos[:,1]
x_axis = array_datos[:,0]
target =  array_datos[:,2]
clase_0 = np.empty((0,3))
clase_1 = np.empty((0,3))
clase_2 = np.empty((0,3))

# recorro todos las plantas y segun su tipo, pongo un color y una etiqueta 
#y ademas la añado a un array de su clase
for pos in range(0,target.size):
    if(target[pos] == 0):
        plt.scatter(x_axis[pos], y_axis[pos], color='blue', label="Clase 0" )
        clase_0 = np.append(clase_0, array_datos[pos:pos+1,[0,1,2]],axis=0)
    elif (target[pos] == 1):
        plt.scatter(x_axis[pos], y_axis[pos], color='red', label="Clase 1")
        clase_1 = np.append(clase_1, array_datos[pos:pos+1,[0,1,2]],axis=0)
    else:
        plt.scatter(x_axis[pos], y_axis[pos], color='green', label="Clase 2")
        clase_2 = np.append(clase_2, array_datos[pos:pos+1,[0,1,2]],axis=0)

#codigo para no repetir las etiquetas
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()


#Apartado 2
test = np.empty((0,3))
entrenamiento = np.empty((0,3))

#cojo aleatoriamente el 20% de cada clase y lo añado a la variable test
#ademas cada vez que elimino uno lo borro de la variable donde estaba
tam = int(len(clase_0)*0.2)
for pos in range(0,tam):
    rand = random.randrange(len(clase_0))
    test = np.append(test, clase_0[rand:rand+1,[0,1,2]],axis=0)
    clase_0 = np.delete(clase_0,[rand],axis=0)
tam = int(len(clase_1)*0.2)
for pos in range(0,tam):
    rand = random.randrange(len(clase_1))
    test = np.append(test, clase_1[rand:rand+1,[0,1,2]],axis=0)
    clase_1 = np.delete(clase_1,[rand],axis=0)
tam = int(len(clase_2)*0.2)
for pos in range(0,tam):
    rand = random.randrange(len(clase_2))
    test = np.append(test, clase_2[rand:rand+1,[0,1,2]],axis=0)
    clase_2 = np.delete(clase_2,[rand],axis=0)

#ahora en las variables solo quedan los datos que usaremos en el entrenamiento
#asi que lo añado a la variable entrenamiento
entrenamiento = np.append(entrenamiento, clase_0,axis=0)
entrenamiento = np.append(entrenamiento, clase_1,axis=0)
entrenamiento = np.append(entrenamiento, clase_2,axis=0)
print("Total de flores: "+str(target.size))
print("--------Datos entrenamiento--------")
print("Cantidad total flores en el array de entrenamiento: "+str(len(entrenamiento)))
print("Cantidad de flores en el array de entrenamiento: "+str(len(entrenamiento)))
print("Cantidad de flores de tipo 0 el array de entrenamiento: "+str(len(entrenamiento[entrenamiento[:,2] == 0])))
print("Cantidad de flores de tipo 1 el array de entrenamiento: "+str(len(entrenamiento[entrenamiento[:,2] == 1])))
print("Cantidad de flores de tipo 2 el array de entrenamiento: "+str(len(entrenamiento[entrenamiento[:,2] == 2])))
print("--------Datos test--------")
print("Cantidad total flores en el array de test: "+str(len(test)))
print("Cantidad de flores en el array de test: "+str(len(test)))
print("Cantidad de flores de tipo 0 el array de test: "+str(len(test[test[:,2] == 0])))
print("Cantidad de flores de tipo 1 el array de test: "+str(len(test[test[:,2] == 1])))
print("Cantidad de flores de tipo 2 el array de test: "+str(len(test[test[:,2] == 2])))

#Apartado 3
#creo una lista de 100 valores desde 0 a 2PI
valores =np.linspace(0,2*math.pi,num=100)
#creo los arrays donde guardare los valores calculados
valores_sin = np.array([])
valores_cos = np.array([])
valores_mix = np.array([])
#recorro los 100 valores y añado los valores del seno, el coseno y la suma
for pos in range(0,valores.size):
    valores_sin = np.append(valores_sin, math.sin(valores[pos])) 
    valores_cos = np.append(valores_cos, math.cos(valores[pos])) 
    valores_mix = np.append(valores_mix, math.sin(valores[pos])+math.cos(valores[pos]))
#pinto con plot los 3 arrays
plt.plot(valores, valores_sin, "--", color='black', label="Sin" )
plt.plot(valores, valores_cos, "--", color='blue', label="Cos" )
plt.plot(valores, valores_mix, "--", color='red', label="Sin+Cos" )
#muestro la legenda
plt.legend(numpoints=1)
plt.show()
