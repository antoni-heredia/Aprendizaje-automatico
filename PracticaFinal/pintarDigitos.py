#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 18:25:56 2019

@author: antonio
"""

import matplotlib.pyplot as plt


X=[
273,323
,272,327
,269,331
,266,334
,262,337
,255,338
,249,338
,241,336
,235,333
,228,330
,224,327
,218,320
,218,315
,221,311
,224,306
,229,301
,236,296
,243,290
,251,285
,259,279
,265,274
,272,267
,277,262
,280,257
,282,252
,283,247
,283,240
,282,236
,279,232
,276,227
,270,224
,265,222
,259,221
,251,222
,246,224
,239,227
,234,233
,230,239
,229,247
,229,254
,231,263
,235,271
,238,278
,243,286
,250,293
,255,299
,260,304
,264,310
,268,313
,272,317
,275,320
,277,324
,278,326
,279,328
,280,329
,280,330
,280,332
,279,332]

x = []
y = []
for i in range(len(X)):
    if i%2 == 0:
        x.append(X[i])
    else:
        y.append(X[i])
plt.xlim(200,400)
plt.ylim(200,400)
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(x, y, 'r--')
plt.show()
X = [88, 92,  2, 99, 16, 66, 94, 37, 70,  0,  0, 24, 42, 65,100,100]
x = []
y = []
for i in range(len(X)):
    if i%2 == 0:
        x.append(X[i])
    else:
        y.append(X[i])
plt.xlim(0,100)
plt.ylim(0,100)
plt.plot(x, y, 'r--')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

X = [
 192 , 318,
 197  ,322,
 199 , 322,
 199 , 318,
 197 , 311,
 192 , 301,
 188 , 288,
 185 , 274,
 182 , 258,
 179 , 244,
 178 , 229]

x = []
y = []
for i in range(len(X)):
    if i%2 == 0:
        x.append(X[i])
    else:
        y.append(X[i])

plt.xlim(0,500)
plt.ylim(0,500)
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(x, y, 'r--')
plt.show()

X = [70,100,100, 97, 70, 81, 45, 65, 30, 49, 20, 33,  0, 16,  0,  0]
x = []
y = []
for i in range(len(X)):
    if i%2 == 0:
        x.append(X[i])
    else:
        y.append(X[i])

plt.xlim(0,200)
plt.ylim(0,200)
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(x, y, 'r--')
plt.show()