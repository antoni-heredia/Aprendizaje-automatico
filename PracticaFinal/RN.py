#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:31:24 2019

@author: antonio
"""

# Crea tu primer MLP en Keras
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras.utils import to_categorical
import numpy as np
# Importing standardscalar module  

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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

# Fija las semillas aleatorias para la reproducibilidad
np.random.seed(7)
# carga los datos

def leerArchivo(ruta):
    f = open(ruta)
    data = np.genfromtxt(f, delimiter=",")
    return data[:,:-1],data[:,-1:]


n_classes = 10

X_test,y_test = leerArchivo("datos/pendigits.tes")
y_test = to_categorical(y_test, num_classes=None)


X,y = leerArchivo("datos/pendigits.tra")


validacion_X = X[5621:, ]
validacion_y = y[5621:, ]
X =X[:5621, ]
y = y[:5621, ]

y = to_categorical(y, num_classes=None)
validacion_y = to_categorical(validacion_y, num_classes=None)

cantNeuronas = [
                [1,1,10],
                [5,5,10],
                [15,15,10],
                [20,20,10],
                [25,25,10],
                [30,30,10],
                [100,100,10],
                [150,150,10],
                [200,200,10],
                [250,250,10]

]
  

#Bucles que tardan muchoi tiempo en ejecutar, por eso estan comentados, los uso para recopiar datos
'''
puntosX = []
puntosY = []
print("------------------Activacion Relu--------------------")
for i in cantNeuronas:
    # crea el modelo
    model = Sequential()
    model.add(Dense(i[0], input_dim=16, activation='relu'))
    model.add(Dense(i[1], activation='relu'))
    model.add(Dense(i[0], activation='relu'))
    model.add(Dense(i[2], activation='softmax'))
    
    # Compila el modelo
    model.compile(loss=losses.binary_crossentropy, optimizer='adam' ,metrics=['accuracy'])
    # Ajusta el modelo
    model.fit(X, y, epochs=150, batch_size=256,verbose=0)
    
    # evalua el modelo
    scores = model.evaluate(X, y,verbose=0)
    scores_test = model.evaluate(X_test, y_test,verbose=0)
    print("------------------------"+str(i[0])+"------------------------")
    print("En el train %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("En el test %s: %.2f%%" % (model.metrics_names[1], scores_test[1]*100))
    puntosY.append(1-scores_test[1])
    puntosX.append(i[1])
plt.figure(1)    
plt.plot(puntosX, puntosY, 'r-.')
print("------------------Activacion Sigmoid--------------------")
puntosX = []
puntosY = []
for i in cantNeuronas:
    # crea el modelo
    model = Sequential()
    model.add(Dense(i[0], input_dim=16, activation='sigmoid'))
    model.add(Dense(i[1], activation='sigmoid'))
    model.add(Dense(i[0], activation='sigmoid'))
    model.add(Dense(i[2], activation='softmax'))
    
    # Compila el modelo
    model.compile(loss=losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
    # Ajusta el modelo
    model.fit(X, y, epochs=150, batch_size=256,verbose=0)
    
    # evalua el modelo
    scores = model.evaluate(X, y,verbose=0)
    scores_test = model.evaluate(X_test, y_test,verbose=0)
    print("------------------------"+str(i[0])+"------------------------")    
    print("En el train %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("En el test %s: %.2f%%" % (model.metrics_names[1], scores_test[1]*100))
    puntosY.append(1-scores_test[1])
    puntosX.append(i[1])
    
plt.title("Variación del error")
plt.xlabel("Cantidad de neuronas por capa")  
plt.ylabel("E_out")
plt.plot(puntosX, puntosY, 'g-.')
plt.show() 

'''

model = Sequential()
model.add(Dense(20, input_dim=16, activation='sigmoid'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))

model.add(Dense(10, activation='softmax'))

# Compila el modelo
model.compile(loss=losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
# Ajusta el modelo
model.fit(X, y, epochs=150, batch_size=256,verbose=0)

# evalua el modelo
scores = model.evaluate(X, y,verbose=0)
scores_test = model.evaluate(X_test, y_test,verbose=0)
y_predecido = model.predict(X_test)
y_numericos = np.argmax(y_predecido, axis=1)
y_test = np.argmax(y_test, axis=1)       

print("El E_in = "+str(1-scores[1]))
print("El E_out en el test = "+str(1-scores_test[1]))
scores_validacion = model.evaluate(validacion_X, validacion_y,verbose=0)
print("El E_out en la validacion = "+str(1-scores_validacion[1]))

plot_confusion_matrix2(y_test, y_numericos, classes=[0,1,2,3,4,5,6,7,8,9], normalize=True,
                    title='Matriz de confusion RN')