import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
from sklearn.datasets import make_blobs

# Fija las semillas aleatorias para la reproducibilidad
np.random.seed(7)
# carga los datos

def leerArchivo(ruta):
    f = open(ruta)
    data = np.genfromtxt(f, delimiter=",")
    return data[:,:-1],data[:,-1:]


n_classes = 10

X_test,y_test = leerArchivo("datos/pendigits.tra")
y_test = y_test.ravel()

X,y = leerArchivo("datos/pendigits.tra")
y = y.ravel()


clf = svm.SVC(kernel='rbf', verbose=1)
clf.fit(X, y)

print(clf.score(X_test,y_test))