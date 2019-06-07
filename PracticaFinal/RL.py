#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:09:13 2019

@author: jose
"""
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from time import time
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

fichero_train = "datos/pendigits.tra"
fichero_test = "datos/pendigits.tes"

def normalizarDatos(datos):
    normalizada = datos / 100
    return normalizada

def loadCSV(fichero, delimitador = ','):
    
    my_data = np.genfromtxt(fichero, delimiter= delimitador)
    #dibujarMatrizCorrelacion(my_data)
    
    clases = my_data[:, -1]     
    datos = my_data[:, :-1]

    #datos = normalizarDatos(datos)
    
    return datos,clases

 
    
def plot_confusion_matrix(df_confusion, title='Matriz de confusion', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    plt.title(title)
    plt.colorbar()
    #tick_marks = np.arange(len(df_confusion.columns))
    #plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    #plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel("Etiquetas")
    plt.xlabel("Prediccion")
    
def regresionLogisticaEntrenarDefecto(X,y,X_test, y_test):
    
    model = LogisticRegression( )
    #model = LogisticRegression(penalty = None,solver = 'newton-cg')
    
    model.fit(X,y)
    
    tasa_acierto_test = model.score(X_test,y_test)
    tasa_acierto_train = model.score(X,y)
    
    return tasa_acierto_test, tasa_acierto_train, model.predict(X_test)

def regresionLogisticaEntrenarMejorado(X,y,X_test, y_test,regurlarizacion = 'l2' , ajuste = 1):
    
    model = LogisticRegression(solver = 'liblinear', multi_class='ovr', penalty = regurlarizacion, C = ajuste, random_state=0 )
    #model = LogisticRegression(penalty = None,solver = 'newton-cg')
    
    model.fit(X,y)
    print(model.n_iter_)
    tasa_acierto = model.score(X_test,y_test)
    tasa_acierto_train = model.score(X,y)
    return tasa_acierto, tasa_acierto_train, model.predict(X_test)
    
def plot_confusion_matrix2(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=[0,1,2,3,4,5,6,7,8,9], yticklabels=classes,
           title=title,
           ylabel='Valores validos',
           xlabel='Valores predecidos')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def anadirInformacionPolinomial(train_X, test_X, grado = 2):
    poly = PolynomialFeatures(grado)
    train_X = poly.fit_transform(train_X)
    test_X = poly.fit_transform(test_X)
    
    return train_X, test_X

def primerDataSet():
    datos_X, datos_y = loadCSV(fichero_train)
    test_X, test_y = loadCSV(fichero_test)
    
    #datos = normalizarDatos(datos)   
    
    
    #test_X = normalizarDatos(test_X)
    
    #train_X,test_X = eliminarVarianza(train_X,test_X, 0);
    
    datos_X,test_X = anadirInformacionPolinomial(datos_X,test_X,2)
    train_X = datos_X[:5621, ]
    train_y = datos_y[:5621, ]
    
    validacion_X = datos_X[5621:, ]
    validacion_y = datos_y[5621:, ]
    regularizacion = 'l2'
    
    
    start_time = time()
    acieto_test, acierto_train,y_predecido_mejorada = regresionLogisticaEntrenarMejorado(train_X,train_y,test_X,test_y,regularizacion,100)
    print("Regresion logistica")
    print("Tiempo=", time() - start_time )
    print("Ein = ",(1-acierto_train)*100)
    print("Eout = ",(1-acieto_test)*100)
    print()
    
    

    matriz_confusion = confusion_matrix(test_y, y_predecido_mejorada)
    matriz_confusion =  matriz_confusion /  matriz_confusion.sum(axis=1) 
    
    #plot_confusion_matrix(matriz_confusion)
    plot_confusion_matrix2(test_y, y_predecido_mejorada, classes=[0,1,2,3,4,5,6,7,8,9], normalize=True,
                      title='Matriz de confusion regresion logistica')
    
    
    
   
    

"""Utilizada para la medicion de los datos de la grafica. Los datos que muestra han sido 
copieados en los ficheros .dat"""
def medirParametroRegularizacion():
    datos_X, datos_y = loadCSV(fichero_train)
    test_X, test_y = loadCSV(fichero_test)
    
    #datos = normalizarDatos(datos)   
    
    
    #test_X = normalizarDatos(test_X)
    
    #train_X,test_X = eliminarVarianza(train_X,test_X, 0);
    
    datos_X,test_X = anadirInformacionPolinomial(datos_X,test_X,2)
    train_X = datos_X[:5621, ]
    train_y = datos_y[:5621, ]
    
    validacion_X = datos_X[5621:, ]
    validacion_y = datos_y[5621:, ]
    regularizacion = 'l1'
    
    i = 1
    while i <= 100:
        
        start_time = time()
        acierto_validacion, acierto_train,y_predecido_mejorada = regresionLogisticaEntrenarMejorado(train_X,train_y,validacion_X,validacion_y,regularizacion,i)
       
    
        print(i, "," , (1-acierto_validacion)*100, ",", time() - start_time)
        i += 2
        

"""La he usado para cargar los datos de los csv y mostrar la grafica"""  
def imprimirEstadisticas():
      l1 = np.genfromtxt("datos/rlregularizacionl1.dat", delimiter= ",")
      l2 = np.genfromtxt("datos/rlregularizacionl2.dat", delimiter= ",")
      
      # evenly sampled time at 200ms intervals
      

      plt.figure()

      #plt.plot(l1[:,0], l1[:,1], 'r-') 
      #plt.plot(l2[:,0], l2[:,1], 'b-')
      p = np.polyfit(l1[:,0],l1[:,1], 1)
      p2 = np.polyfit(l2[:,0],l2[:,1], 1)
      y_ajuste = p[0]*l1[:,0] + p[1]
      y_ajuste2 =p2[0]*l2[:,0] + p2[1]

      plt.plot(l1[:,0], y_ajuste, 'r-')
      plt.plot(l2[:,0], y_ajuste2, 'b-')
      
      
      plt.gca().legend(('l1','l2'))
      plt.ylabel("Error en validación(0%-100%)")
      plt.xlabel("Parametro de regularacion")
      plt.title("Error en validacion")
      
      plt.show()
      
      plt.figure()

      plt.plot(l1[:,0], l1[:,2], 'r-') 
      plt.plot(l2[:,0], l2[:,2], 'b-')
      plt.gca().legend(('l1','l2'))
      plt.ylabel("Tiempo de ejecución(s)")
      plt.xlabel("Parametro de regularacion")
      plt.title("Tiempos de ejecución para regularizacion")
      
      plt.show()    
    
def main():
 
    print("--------------------------------------------------")
    print("Pen-Based Recognition of Handwritten Digits Data Set con RL")
    primerDataSet()
    print("--------------------------------------------------")
    
    #medirParametroRegularizacion()
    imprimirEstadisticas()
    #segundoDataSet()
    

 
    
    
    
    
    
if __name__== "__main__":
  main()
  