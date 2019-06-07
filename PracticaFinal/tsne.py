#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 19:23:25 2019

@author: antonio
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
def leerArchivo(ruta):
    f = open(ruta)
    data = np.genfromtxt(f, delimiter=",")
    return data[:,:-1],data[:,-1:]


n_classes = 10


X,y = leerArchivo("datos/pendigits.tra")
y = y.ravel()



tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X)




dataset = pd.DataFrame({'EjeX':tsne_results[:,0],'EjeY':tsne_results[:,1],'y':y})


plt.figure(figsize=(16,10))
sns.scatterplot(
    x="EjeX", y="EjeY",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=dataset,
    legend="full",
    alpha=0.3
)