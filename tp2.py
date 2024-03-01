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
df_label = df_sign['label']
df_sign = df_sign.drop(['label'], axis=1)

# Convierto en array el dataset.
X = df_sign.values

# Genero una 'Preview' del dataset para ver como esta compuesto.
fig, axe = plt.subplots(3,3)
fig.suptitle('Preview del dataset', size = 14, x= 0.5, y = 0.995)
plt.subplots_adjust(hspace= 0.3, wspace = -0.2)

label : int = 0
for i in range(3):
    for j in range(3):
        axe[i][j].imshow(X[label].reshape(28,28), cmap='gray')
        axe[i][j].set_title('letra: ' + letras[df_label[label]])
        label += 1

# Elimino los ticks de todos los subplot.
for ax in axe:
    for i in range(3):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        
# Ahora que entiendo el dataset, puedo comenzar el analisis exploratorio.

        



