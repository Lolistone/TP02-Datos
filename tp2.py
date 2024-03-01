# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 19:52:48 2024

@author: lolocrsn 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Guardamos la ruta a la carpeta donde está el csv.
carpeta = '~/Dropbox/UBA/2024/LaboDeDatos/TP02/'

# Importamos el archivo .csv
df_sign = pd.read_csv(carpeta + 'sign_mnist_train.csv')

# Creo un diccionario label -> letra.
letras : dict = {'0' : 'A', 1 : 'B', 2 : 'C', 3 : 'D', 4 : 'E', 5 : 'F', 6 : 'G',
                 7: 'H', 8: 'I', 9: 'J', 10 : 'K', 11 : 'L', 12 : 'M', 13 : 'N',
                 14 : 'O', 15 : 'P', 16 : 'Q', 17: 'R', 18 : 'S', 19: 'T', 20: 'U',
                 21: 'V', 22 : 'W', 23 : 'X', 24: 'Y', 25 : 'Z'}

# Separo la columna con los label.
Y = df_sign['label']
X = df_sign.drop(['label'], axis=1)

# Convierto en array el dataset.
X = X.values

# Genero una 'Preview' del dataset para ver como son las imagenes.
fig, axe = plt.subplots(3,3)
fig.suptitle('Preview del dataset', size = 14, x= 0.5, y = 0.995)
plt.subplots_adjust(hspace= 0.4, wspace = -0.2)

label : int = 0
for i in range(3):
    for j in range(3):
        axe[i][j].imshow(X[label].reshape(28,28), cmap='gray')
        axe[i][j].set_title('letra: ' + letras[Y[label]])
        label += 1

# Elimino los ticks de todos los subplot.
for ax in axe:
    for i in range(3):
        ax[i].set_xticks([])
        ax[i].set_yticks([])

# Aca me falta el savefig para guardar las figuras, así las agregamos al principio del informe

# Elimino las variables que no uso.
del X, Y
del label
del i, j
del ax, axe

# (Esto iria en el informe, no aca, pero mientras tanto)
# Los atributos que a nuestro criterio, son mas relevantes para predecir una seña son aquellos
# correspondientes a los pixeles 'centrales' dado que dan a conocer mayor información acerca de la mano.
# Si hay atributos que a simple vista podrían descartarse son los pixeles del fondo. Sin embargo, habria
# que tener cierto cuidado pues no todas las imagenes poseen los mismo pixeles de fondo.}

# Filtro en el dataset las letras, L, E y M.
Xl = df_sign[df_sign['label'] == 11] 
Xm = df_sign[df_sign['label'] == 12] 
Xe = df_sign[df_sign['label'] == 4]

Yl = Xl['label']
Ym = Xm['label']
Ye = Xe['label']

# Elimino la columna label y convierto en array
Xl = Xl.drop(['label'], axis = 1).values
Xm = Xm.drop(['label'], axis = 1).values
Xe = Xe.drop(['label'], axis = 1).values

# Comparamos la letra E contra la L.
fig, axe = plt.subplots(2,3)
fig.suptitle('Letra E vs L', size = 14, x= 0.5, y = 0.995)
plt.subplots_adjust(hspace= 0.4, wspace = -0.2)

for j in range(3):
    axe[0][j].imshow(Xl[j].reshape(28,28), cmap='gray')
    axe[0][j].set_title('letra: L')
    
for j in range(3):
    axe[1][j].imshow(Xe[j].reshape(28,28), cmap='gray')
    axe[1][j].set_title('letra: E')

# Elimino los ticks de todos los subplot.
for ax in axe:
    for i in range(3):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        
# Ahora, comparo la E con la M
fig, axe = plt.subplots(2,3)
fig.suptitle('Letra E vs M', size = 14, x= 0.5, y = 0.995)
plt.subplots_adjust(hspace= 0.4, wspace = -0.2)

for j in range(3):
    axe[0][j].imshow(Xm[j].reshape(28,28), cmap='gray')
    axe[0][j].set_title('letra: M')
    
for j in range(3):
    axe[1][j].imshow(Xe[j].reshape(28,28), cmap='gray')
    axe[1][j].set_title('letra: E')

# Elimino los ticks de todos los subplot.
for ax in axe:
    for i in range(3):
        ax[i].set_xticks([])
        ax[i].set_yticks([])

# Elimino las variables que no utilizo.
del Xl, Xm, Xe
del Yl, Ym, Ye
del i,j
        
# (Otra vez, esto va al informe pero mientras lo dejo documentado aca)
# Viendo los dos graficos, se puede ver bien claro que hay letras que se parecen mucho entre si.
# En este caso, la letra E y la letra M son muy similares entre si. Pero, son bastante diferentes 
# a la letra L.

# Ahora, comparemos imagenes de una misma letra. En particular, la letra C.

# Filtro en el dataset la letra C.
Xc = df_sign[df_sign['label'] == 2] 

# Elimino la columna label y convierto en array
Xc = Xc.drop(['label'], axis = 1).values

# Primero, veamos varios ejemplos de una misma letra.
fig, axe = plt.subplots(2,3)
fig.suptitle('Letra C', size = 20, x= 0.5, y = 0.98)
plt.subplots_adjust(hspace= 0.2, wspace = 0.2)

label = 0
for i in range(2):
    for j in range(3):
        axe[i][j].imshow(Xc[label].reshape(28,28), cmap='gray')
        label += 1
        
# Elimino los ticks de todos los subplot.
for ax in axe:
    for i in range(3):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        
# Ahora, podemos ver claramente que hay ciertas diferencias entre una misma letra
# por lo que tenemos que tener cuidado. 

# Una posible solución, para encontrar los pixeles mas característicos es apilar las imagenes.
Xc = df_sign[df_sign['label'] == 2]
Xc = Xc.drop(['label'], axis = 1)

# Sumo todos los pixeles de cada columnas y lo convierto en array.
Xc = Xc.sum(axis= 0)
Xc = Xc.values

# Graficamos.
fig, ax = plt.subplots()
fig.suptitle('Letras C apiladas', size = 14, x= 0.5, y = 0.98)

ax.imshow(Xc.reshape(28,28), cmap = 'gray')
ax.set_xticks([])
ax.set_yticks([])

# Eliminamos variables
del i, j
del Xc

# Podemos ver ahora, cuales son exactamente los atributos que identifican a la leta C. Y, además
# determinar con esto donde esta la mayor semejanza entre estas.



