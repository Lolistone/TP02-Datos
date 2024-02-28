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

####### Probando funciones ##########

# Separo la columna con los label.
df_label = df_sign['label']
df_sign = df_sign.drop(['label'],axis=1)

# Genero un dataset a entrenar
X = df_sign.values

# Genero una 'Preview' del dataset
fig,axe=plt.subplots(1,2) # Genero una 'matriz', accedo con axe[i,j], en este caso axe[i] pues es fila.
fig.suptitle('Preview del dataset', size = 20)

axe[0].imshow(X[0].reshape(28,28), cmap='gray')
axe[0].set_title('label: 3  letter: D')
axe[1].imshow(X[1].reshape(28,28), cmap='gray')
axe[1].set_title('label: 6  letter: G')

#######################################