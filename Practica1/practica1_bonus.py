#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Bonus: Metodo de Newton
Created on Sat Mar 23 18:53:30 2019

@author: antonio
"""

def gradiente_descendiente(_lr, _punto, _funcion, _iteraciones, _grafias=True, minimo=None):

    _iter = 0 #Contador de iteraciones
    _dx = sp.diff(_funcion, x) #Derivadas parciales
    _dy = sp.diff(_funcion, y)
    _lam_dx = sp.lambdify((x,y),_dx) #Lambdifyco las derivadas parciales para calcular mas rapido el valor
    _lam_dy = sp.lambdify((x,y),_dy)
    _lam_f = sp.lambdify((x,y),_funcion)

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

