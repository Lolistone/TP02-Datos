# -*- coding: utf-8 -*-
"""
Funciones Auxiliares
"""
import numpy as np
import six
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#%% Funcion para testear la variacion de la precision en funcion de la #Atributos
 
def vecinosAleatorios(kv: int, reps: int, cant_atributos: int, X, Y):
    cant_vecinos = range(1, kv) 
    
    resultados_test  = np.zeros((reps, len(cant_vecinos)))
    resultados_train = np.zeros((reps, len(cant_vecinos)))
    
    # Cantidad de veces que elijo al azar pixeles.
    for i in range(reps):
    
        # Cantidad de vecinos que tomo en cada iteración.
        for k in cant_vecinos:
            
            #Separamos en train y test. Tomo un sample de 3 pixeles.
            X_train, X_test, y_train, y_test = train_test_split(X.sample(cant_atributos, axis = 1), Y, test_size = 0.3, shuffle=True)
            
            # Declaramos el tipo modelo
            clf = KNeighborsClassifier(n_neighbors = k)
            
            # Entrenamos el modelo
            clf.fit(X_train, y_train)
            
            # Evaluamos el modelo con datos de train y luego de test
            resultados_train[i, k-1] = clf.score(X_train, y_train)
            resultados_test[i, k-1]  = clf.score(X_test , y_test)
    
    # Saco el promedio para cantidad de vecinos.
    resultados_train = np.mean(resultados_train, axis = 0) 
    resultados_test  = np.mean(resultados_test , axis = 0)
    
    return resultados_train, resultados_test

#%% Funcion auxiliar para tablas

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax