from sklearn import svm
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# Fija las semillas aleatorias para la reproducibilidad
np.random.seed(7)
# carga los datos
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

def leerArchivo(ruta):
    f = open(ruta)
    data = np.genfromtxt(f, delimiter=",")
    return data[:,:-1],data[:,-1:]


n_classes = 10

X_test,y_test = leerArchivo("datos/pendigits.tes")
y_test = y_test.ravel()

X,y = leerArchivo("datos/pendigits.tra")

validacion_X = X[5621:, ]
validacion_y = y[5621:, ]
X =X[:5621, ]
y = y[:5621, ]

validacion_y = validacion_y.ravel()
y = y.ravel()
#Bucles que tardan muchoi tiempo en ejecutar, por eso estan comentados, los uso para recopiar datos
'''
grados = np.linspace(0 , 10, num=11)
puntosX = []
puntosY = []
for i,xi in enumerate(grados):
    clf = svm.SVC(kernel='poly',coef0=44,degree=xi)
    clf.fit(X, y)
    print("Acierto en test con Degree= "+str(xi)+": " + str(clf.score(X_test,y_test)))
    puntosX.append(xi)
    puntosY.append(1-clf.score(X_test,y_test))
    
print("Acierto en test con Degree=2: " + str(clf.score(X_test,y_test)))
plt.figure(1)
plt.title("Variación del error")
plt.xlabel("Grado del polinomio")  
plt.ylabel("E_out")
plt.plot(puntosX, puntosY, 'g-.')
plt.show() 

coeficientes = np.linspace(0, 100, num=100)
puntosX = []
puntosY = []
for i,xi in enumerate(coeficientes):
    clf = svm.SVC(kernel='poly',coef0=xi,degree=2)
    clf.fit(X, y)
    puntosX.append(xi)
    puntosY.append(1-clf.score(X_test,y_test))
plt.figure(2)
plt.title("Variación del error")
plt.xlabel("Coeficiente independiente")  
plt.ylabel("E_out")
plt.plot(puntosX, puntosY, 'g-.')
plt.show() 
'''

clf = svm.SVC(kernel='poly',coef0=5,degree=2)
clf.fit(X, y)
y_numericos = clf.predict(X_test)
print("E_in : " + str(1-clf.score(X,y)))
print("E_out en test : " + str(1-clf.score(X_test,y_test)))
print("E_out en validacion : " + str(1-clf.score(validacion_X,validacion_y)))

plot_confusion_matrix2(y_test, y_numericos, classes=[0,1,2,3,4,5,6,7,8,9], normalize=True,
                    title='Matriz de confusion RN')