#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 18:16:57 2019

@author: Antonio Jesús Heredia Castillo
Practica 1
"""
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

#varialbe para mostrar(o no) las graficas, en false no se muestran
MOSTRAR_GRAFICAS = True

def gradiente_descendiente(_lr, _punto, _funcion, _iteraciones, _grafias=True, minimo=None):
    """
    Esta funcion realiza el algoritmo del gradiente descendiente


    Parameters:
    lr (float): Tasa de aprendizaje
    punto (array[x,y]): Punto desde el cual se quiere aplicar el algoritmo
    funcion (float): Funcion a la que se le quiere aplicar el gradiente descendente
    iteraciones (int): Numero maximo de iteraciones que aplicara el algoritmo
    Returns:
    array[x, y]: Punto en el que ha convergido el algoritmo
    _grfias:Si queremos mostrar los  puntos
    minimo = un float64 que indica a que minimo parar
    """
    _iter = 0 #Contador de iteraciones
    _dx = sp.diff(_funcion, x) #Derivadas parciales
    _dy = sp.diff(_funcion, y)
    _lam_dx = sp.lambdify((x,y),_dx) #Lambdifyco las derivadas parciales para calcular mas rapido el valor
    _lam_dy = sp.lambdify((x,y),_dy)
    _lam_f = sp.lambdify((x,y),funcion)

    while True:
        #creo una copia del punto, para que a la hora de evuluar el x, no lo pierda para y
        _punto_copia = np.copy(_punto)
        #si hay mas iteraicones de las que yo quiero paro
        if _iter >= _iteraciones:
            print ("La funcion ha acabado a las " + str(_iter) + " iteraciones")
            break;
        if minimo != None:
            if _lam_f(_punto[0],_punto[1]) < minimo:
                print ("La funcion ha acabado a las " + str(_iter) + " iteraciones")
                break
        #calculo el gradiente
        _punto[0] = _punto[0]-_lr*_lam_dx(_punto_copia[0],_punto_copia[1])
        _punto[1] = _punto[1]-_lr*_lam_dy(_punto_copia[0],_punto_copia[1])
        #si esta activada la opcion de mostrar graficas las muestro
        if MOSTRAR_GRAFICAS and _grafias:
            plt.plot(_punto[0],_punto[1],".",c="red")
            #ax.scatter(_punto[0], _punto[1], funcion.evalf(subs={x:_punto[0],y:_punto[1]}), c='r', marker='^')

        _iter =_iter+1
    return _punto

#Las varialbes que voy a usar en mi funcion
x, y = sp.symbols('x y')
# creo la funcion
funcion = (((x**2)*(np.e**y))-(2*(y**2)*(np.e**-x)))**2
_lam_f = sp.lambdify((x,y),funcion)

#punto en el que inicia
punto = np.array([1,1],  dtype = float)
#la tasa de aprendizaje
lr = 0.01

#Si quiero mostrar como funciona el gradiente
if MOSTRAR_GRAFICAS:
    plt.figure(1)
    plt.title('Ejercicio 2')
    plt.xlabel('Variable x')
    plt.ylabel('Variable y')
    #resolucion con la que quiero mostrar el contorno
    res = 100
    #puntos equiespaciados para calcular la funcion
    _X = np.linspace( punto[0]-0.4,punto[0]+0.01,res)
    _Y = np.linspace(punto[1]-0.06,punto[1]+0.01,res)
    _Z = np.zeros((res,res))
    for ix, xf in enumerate(_X):
        for iy, yf in enumerate(_Y):
            _Z[iy,ix]=_lam_f(xf,yf) # calculo la altura de la funcion de los puntos equiespaciados


    #descomentar si se quiere ver en 3D
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.contourf(_X,_Y,_Z,res)
    ax.scatter(punto[0], punto[1], funcion.evalf(subs={x:punto[0],y:punto[1]}), c='r', marker='^')
    """
    #Muestro el contorno
    plt.contourf(_X,_Y,_Z,res)
    plt.colorbar()
    plt.plot(punto[0],punto[1],".",c="yellow")

#calculo el punto final donde termina el algoritmo
print ("---------Apartado 2 del ejercicio de gradiente---------")
print ("La altura en la que comienza es: "+str(_lam_f(punto[0],punto[1])))
punto = gradiente_descendiente(lr,punto,funcion,100,True,np.float64(10**(-14)))
print ("El punto en el que acaba es: "+str(punto))
print ("La altura en la que acaba es: "+str(_lam_f(punto[0],punto[1])))
#muestro donde a acabado el algoritmo
if MOSTRAR_GRAFICAS:

    """
    ax.scatter(punto[0], punto[1], funcion.evalf(subs={x:punto[0],y:punto[1]}), c='w', marker='^')
    ax.show()
    """
    plt.plot(punto[0],punto[1],".",c="white")
    plt.show()

input("Pulsa  Enter para continuar al siguiente apartado...")

#funcion y punto
funcion = (x**2)+(2*y**2)+(2*sp.sin(2*np.pi*x)*sp.sin(2*np.pi*y))
_lam_f = sp.lambdify((x,y),funcion)
punto = np.array([0.1,0.1],  dtype = float)
#la tasa de aprendizaje
lr = 0.1


#Si quiero mostrar como funciona el gradiente
if MOSTRAR_GRAFICAS:
    plt.figure(2)
    plt.title('Ejercicio 3')
    plt.xlabel('Variable x')
    plt.ylabel('Variable y')
    #resolucion con la que quiero mostrar el contorno
    res = 100
    #puntos equiespaciados para calcular la funcion
    _X = np.linspace( punto[0]-2,punto[0]+2,res)
    _Y = np.linspace(punto[1]-2,punto[1]+2,res)
    _Z = np.zeros((res,res))
    for ix, xf in enumerate(_X):
        for iy, yf in enumerate(_Y):
            _Z[iy,ix]=_lam_f(xf,yf) # calculo la altura de la funcion de los puntos equiespaciados
    #Muestro el contorno
    plt.contourf(_X,_Y,_Z,res)
    plt.colorbar()
    plt.plot(punto[0],punto[1],".",c="yellow")


print ("---------Apartado 3.a del ejercicio de gradiente---------")
print ("El lr es: "+str(lr))
print ("La altura en la que comienza es: "+str(_lam_f(punto[0],punto[1])))
punto = gradiente_descendiente(lr,punto,funcion,50)
print ("El punto en el que acaba es: "+str(punto))
print ("La altura en la que acaba es: "+str(_lam_f(punto[0],punto[1])))

#muestro donde a acabado el algoritmo
if MOSTRAR_GRAFICAS:
    plt.plot(punto[0],punto[1],".",c="white")
    plt.show()


punto = np.array([0.1,0.1],  dtype = float)
#la tasa de aprendizaje
lr = 0.01

#Si quiero mostrar como funciona el gradiente
if MOSTRAR_GRAFICAS:
    plt.figure(3)
    plt.title('Ejercicio 3')
    plt.xlabel('Variable x')
    plt.ylabel('Variable y')
    #resolucion con la que quiero mostrar el contorno
    res = 100
    #puntos equiespaciados para calcular la funcion
    _X = np.linspace( punto[0]-1,punto[0]+1,res)
    _Y = np.linspace(punto[1]-1,punto[1]+1,res)
    _Z = np.zeros((res,res))
    for ix, xf in enumerate(_X):
        for iy, yf in enumerate(_Y):
            _Z[iy,ix]=_lam_f(xf,yf) # calculo la altura de la funcion de los puntos equiespaciados

    #Muestro el contorno
    plt.contourf(_X,_Y,_Z,res)
    plt.colorbar()
    plt.plot(punto[0],punto[1],".",c="yellow")


#la tasa de aprendizaje
lr = 0.01
print ("El lr es: "+str(lr))
print ("La altura en la que comienza es: "+str(_lam_f(punto[0],punto[1])))
punto = gradiente_descendiente(lr,punto,funcion,50)
print ("El punto en el que acaba es: "+str(punto))
print ("La altura en la que acaba es: "+str(_lam_f(punto[0],punto[1])))

#muestro donde a acabado el algoritmo
if MOSTRAR_GRAFICAS:
    plt.plot(punto[0],punto[1],".",c="white")
    plt.show()

input("Pulsa  Enter para continuar al siguiente apartado...")
#la tasa de aprendizaje
lr = 0.01
punto = np.array([0.1,0.1],  dtype = float)
print ("---------Apartado 3.b del ejercicio de gradiente---------")
print ("---------Punto "+ str(punto) +"---------")
print ("La altura en la que comienza es: "+str(_lam_f(punto[0],punto[1])))
punto = gradiente_descendiente(lr,punto,funcion,50, False)
print ("El punto en el que acaba es: "+str(punto))
print ("La altura en la que acaba es: "+str(_lam_f(punto[0],punto[1])))

punto = np.array([1,1],  dtype = float)
print ("---------Punto "+ str(punto) +"---------")
print ("La altura en la que comienza es: "+str(_lam_f(punto[0],punto[1])))
punto = gradiente_descendiente(lr,punto,funcion,50, False)
print ("El punto en el que acaba es: "+str(punto))
print ("La altura en la que acaba es: "+str(_lam_f(punto[0],punto[1])))

punto = np.array([-0.5,-0.5],  dtype = float)
print ("---------Punto "+ str(punto) +"---------")
print ("La altura en la que comienza es: "+str(_lam_f(punto[0],punto[1])))
punto = gradiente_descendiente(lr,punto,funcion,50, False)
print ("El punto en el que acaba es: "+str(punto))
print ("La altura en la que acaba es: "+str(_lam_f(punto[0],punto[1])))
punto = np.array([-1,-1],  dtype = float)

print ("---------Punto "+ str(punto) +"---------")
print ("La altura en la que comienza es: "+str(_lam_f(punto[0],punto[1])))
punto = gradiente_descendiente(lr,punto,funcion,50, False)
print ("El punto en el que acaba es: "+str(punto))
print ("La altura en la que acaba es: "+str(_lam_f(punto[0],punto[1])))
