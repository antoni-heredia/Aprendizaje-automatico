#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:09:13 2019

@author: jose
"""

from sklearn.preprocessing import PolynomialFeatures
from time import time
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


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

    start_time = time()
    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    clf.fit(train_X, train_y)  


    #print(clf.feature_importances_)
    print("Tiempo =", time() - start_time )
    print("E_in = ", 100 - (100 * clf.score(train_X, train_y)))
    print("E_out = ", 100 - ( 100 * clf.score(test_X, test_y)))
    
   
    y_predecido = clf.predict(test_X);
    matriz_confusion = confusion_matrix(test_y, y_predecido)
    matriz_confusion =  matriz_confusion /  matriz_confusion.sum(axis=1) 
    plot_confusion_matrix2(test_y, y_predecido, classes=[0,1,2,3,4,5,6,7,8,9], normalize=True,
                    title='Matriz de confusion RF')
    

def calcularParticion():
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
    
    
    i = 10;
    #num_variables = np.sqrt(i)
    while (i <= 250):
        
        start_time = time()
        clf = RandomForestClassifier(n_estimators=i,  random_state=0)
        clf.fit(train_X, train_y)  
        print( i,",", 100 - (100 * clf.score(validacion_X, validacion_y)), ",", time() - start_time )
        i += 10

    #print(clf.feature_importances_)


def imprimirEstadisticas():
      raiz = np.genfromtxt("datos/rfraiz.dat", delimiter= ",")
      mitad = np.genfromtxt("datos/rfmitad.dat", delimiter= ",")
      igual = np.genfromtxt("datos/rfigual.dat", delimiter= ",")
      
      # evenly sampled time at 200ms intervals
      plt.figure()

      plt.plot(raiz[:,0], raiz[:,1], 'r-') 
      plt.plot(mitad[:,0], mitad[:,1], 'b-')
      plt.plot(igual[:,0], igual[:,1], 'g-')
      plt.gca().legend(('m=sqrt(p)','m=p/2', 'm=p'))
      plt.ylabel("Error en validación")
      plt.xlabel("Nuero de árboles")
      plt.title("Error en muestra para parametro m")
      
      plt.show()
      
      plt.figure()

      plt.plot(raiz[:,0], raiz[:,2], 'r-') 
      plt.plot(mitad[:,0], mitad[:,2], 'b-')
      plt.plot(igual[:,0], igual[:,2], 'g-')
      plt.gca().legend(('m=sqrt(p)','m=p/2', 'm=p'))
      plt.ylabel("Tiempo de ejecución(s)")
      plt.xlabel("Nuero de árboles")
      plt.title("Tiempos de ejecución para parametro m")
      
      plt.show()
 
   
    
    
    
    
    
    
    
    
    
    
    

def main():
    print("--------------------------------------------------")
    print("Pen-Based Recognition of Handwritten Digits Data Set con RF")
    primerDataSet()
    print("--------------------------------------------------")
    imprimirEstadisticas()
    #calcularParticion()
    

    
    
    
    
    
    
if __name__== "__main__":
  main()
  