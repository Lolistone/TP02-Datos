# -*- coding: utf-8 -*-
"""
Trabajo Practico N°2: clasificación y selección de modelos, utilizando validación cruzada.
Materia: Laboratorio de Datos - Verano 2024

Contenido: Modelos de clasificacion KNN y Arbol de decision. Generacion de graficos y tablas.

Integrantes: Chapana Puma Joselin , Martinelli Lorenzo, Padilla Ramiro Martin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz
import seaborn as sns
import six # Para graficar tablas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from utils import vecinosAleatorios, render_mpl_table

# Guardamos la ruta a la carpeta donde está el csv.
carpeta = '~/Dropbox/UBA/2024/LaboDeDatos/TP02/'

# Importamos el archivo .csv
df_sign = pd.read_csv(carpeta + 'sign_mnist_train.csv')

# Creo un diccionario label -> letra.
letras : dict = {0 : 'A', 1 : 'B', 2 : 'C', 3 : 'D', 4 : 'E', 5 : 'F', 6 : 'G',
                 7: 'H', 8: 'I', 9: 'J', 10 : 'K', 11 : 'L', 12 : 'M', 13 : 'N',
                 14 : 'O', 15 : 'P', 16 : 'Q', 17: 'R', 18 : 'S', 19: 'T', 20: 'U',
                 21: 'V', 22 : 'W', 23 : 'X', 24: 'Y', 25 : 'Z'}

#%% Análisis Exploratorio de Datos.

# Separo la columna con los label.
Y = df_sign['label']
X = df_sign.drop(['label'], axis=1)

# Convierto en array el dataset.
X = X.values

# Genero una 'Preview' del dataset para ver como son las imagenes.
fig, axe = plt.subplots(3,3)
fig.suptitle('Preview del dataset', size = 14)
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
        
plt.rcParams['figure.autolayout'] = True
# Guarda la imagen y aumenta su dpi
plt.savefig('atributos_relevantes.png', dpi = 400)
plt.show()
plt.close(fig)

# Analizo la distribucion de clases.
distribucion = df_sign.groupby(['label']).size().reset_index(name = 'cantidad_de_imagenes')

# Realizamos un gráfico de barras.
fig, ax = plt.subplots()

plt.rcParams['font.family'] = 'sans-serif'
ax.bar(data= distribucion, 
       x='label', 
       height='cantidad_de_imagenes',
       color = 'maroon',
       width = 0.8)
       
ax.set_title('Distribución de Clases')                    
ax.set_xlabel('Letras', fontsize='medium')                      
ax.set_ylabel('Cantidad de Imagenes', fontsize='medium')

l = list(letras.keys())
label = list(letras.values())

ax.set_xticks(l)
ax.set_xticklabels(label) # Cambio por las letras los parametros del eje x

plt.savefig('distribucion.png', dpi = 400)

# Elimino las variables que no uso.
del X, Y
del label
del i, j
del ax, axe, fig, l
del distribucion

# Filtro en el dataset las letras, L, E y M.
Xl = df_sign[df_sign['label'] == 11] 
Xm = df_sign[df_sign['label'] == 12] 
Xe = df_sign[df_sign['label'] == 4]

# Elimino la columna label y convierto en array
Xl = Xl.drop(['label'], axis = 1).values
Xm = Xm.drop(['label'], axis = 1).values
Xe = Xe.drop(['label'], axis = 1).values

# Comparamos la letra E contra la L.
fig, axe = plt.subplots(2,3)
fig.suptitle('Letra E vs L', size = 14)
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
        
plt.rcParams['figure.autolayout'] = True       
plt.savefig('letra_E_vs_L.png', dpi = 400)
plt.show()
plt.close(fig)

# Ahora, comparo la E con la M
fig, axe = plt.subplots(2,3)
fig.suptitle('Letra E vs M', size = 14)
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
        
plt.rcParams['figure.autolayout'] = True
plt.savefig('letra_E_vs_M.png', dpi = 400)
plt.show()
plt.close(fig)

# Elimino las variables que no utilizo.
del Xl, Xm, Xe
del i,j
del ax, axe, fig

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
     
plt.savefig('letra_C.png', dpi = 400)
plt.show()
plt.close(fig)     

# Una posible solución, para encontrar los pixeles mas característicos es apilar las imagenes.
Xc = df_sign[df_sign['label'] == 2]
Xc = Xc.drop(['label'], axis = 1)

# Sumo todos los pixeles de cada columnas y lo convierto en array.
Xc = Xc.sum(axis= 0)
Xc = Xc.values

# Graficamos.
fig, ax = plt.subplots()
fig.suptitle('Letras C apiladas', size = 14, x= 0.5, y = 0.96)

# Notemos que imshow rescala solo la imagen.
ax.imshow(Xc.reshape(28,28), cmap = 'gray')
ax.set_xticks([])
ax.set_yticks([])

plt.savefig('letra_C_apilada.png', dpi = 400)
plt.show()
plt.close(fig)  

# Eliminamos variables
del i, j
del Xc
del label, ax, axe, fig

#%% Construccion de modelo.

# Armo un nuevo dataframe que contiene unicamente a la A y la L
df_al = df_sign.loc[df_sign['label'].isin([0, 11])]

# Miramos la distribución.
cantA = len(df_al[df_al['label'] == 0])
cantL = len(df_al[df_al['label'] == 11])

# Realizamos un gráfico de barras.
fig = plt.figure(figsize = (7, 6))

plt.bar(['A', 'L'], [cantA, cantL], color ='maroon', 
        width = 0.5)
 
for i, valor in enumerate([cantA, cantL]):
    plt.text(i, valor, str(valor), ha='center', va='bottom')

plt.xlabel("Letras")
plt.ylabel("Cantidad de Imágenes")
plt.title("Cantidad de imágenes por clase")
plt.savefig('cantidad_letras_A_L.png', dpi = 400)
plt.show()
plt.close(fig)

# Busco los pixeles mas significativos para distinguir entre la A y la L.

# Primero apilo las letras A y L.
Xa = df_sign[df_sign['label'] == 0]
Xa = Xa.drop(['label'], axis = 1)
Xl = df_sign[df_sign['label'] == 11]
Xl = Xl.drop(['label'], axis = 1)

# Sumo todos los pixeles de cada columnas y los resto.
Xa = Xa.sum(axis= 0)
Xl = Xl.sum(axis= 0)

# A través de este df podremos encontrar las regiones de mayor varianza.
restaAL = Xl - Xa

# Me guardo en una lista los 10 pixeles de mayor varianza.
maxima_varianza = restaAL.nlargest(n=10)

# Convertimos en array los df.
Xa = Xa.values
Xl = Xl.values
restaAL = restaAL.values

# Graficamos A y L apiladas
fig, axe = plt.subplots(1, 2)
fig.suptitle('Letras A y L apiladas', size = 16)

# Notemos que imshow reescala solo la imagen.
axe[0].imshow(Xa.reshape(28,28), cmap = 'gray')
axe[1].imshow(Xl.reshape(28,28), cmap = 'gray')

for ax in axe:
    ax.set_xticks([])
    ax.set_yticks([])

plt.rcParams['figure.autolayout'] = True
plt.savefig('letras_A_L_apiladas.png', dpi = 400)    
plt.show()
plt.close(fig)

# Graficamos A - L.
fig, ax = plt.subplots()
fig.suptitle('Varianza entre A y L', size = 14, x= 0.5, y = 0.96)
ax.imshow(restaAL.reshape(28,28), cmap = 'gray')

# Borro las variables que ya no necesito.
del Xa, Xl, restaAL
del cantA, cantL
del ax, axe, i, fig
del valor

# Separo los datos a predecir
Y = df_al['label']
X = df_al.drop(['label'], axis=1)

# Probemos distintas cantidades de atributos al azar para determinar su rendimiento.
cant_atributos = range(1,15)
repeticiones = 10

resultados_test  = np.zeros((repeticiones, len(cant_atributos)))
resultados_train = np.zeros((repeticiones, len(cant_atributos)))

# Cantidad de veces que elijo al azar pixeles. 
for i in range(repeticiones):
    
    # Cantidad de atributos que tomo en cada iteración.
    for j in cant_atributos:
        
        #Separamos en train y test
        X_train, X_test, y_train, y_test = train_test_split(X.sample(j, axis = 1), Y, test_size = 0.3, shuffle=True)
        
        # Declaramos el tipo modelo
        clf = KNeighborsClassifier(n_neighbors = 3)
        
        # Entrenamos el modelo
        clf.fit(X_train, y_train)
        
        # Evaluamos el modelo con datos de train y luego de test
        resultados_train[i, j-1] = clf.score(X_train, y_train)
        resultados_test[i, j-1]  = clf.score(X_test , y_test)
        
resultads_train = np.mean(resultados_train, axis = 0) 
resultads_test  = np.mean(resultados_test , axis = 0)

# Graficamos los resultados anteriores.
sns.set_style('whitegrid')

plt.plot(cant_atributos, resultads_train, label = 'Train')
plt.plot(cant_atributos, resultads_test, label = 'Test')
plt.legend()
plt.title('Performance en función de la cantidad de atributos')
plt.xlabel('Cantidad de atributos')
plt.ylabel('Precisión')
plt.xticks(cant_atributos)
plt.ylim(0.60,1.05)
plt.savefig('pixelesvsacc.png', dpi = 400)
plt.show()

# Ahora que sabemos como influye la cantidad de atributos en la precision, veamos la cantidad de vecinos.

# Separamos en Train y Test.
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, shuffle=True)

# Veamos el caso de 3 pixeles, tomando primero aquellos de mayor varianza.
valores_k = range(1, 10)
pixeles = [300, 273, 245]

resultados_train = np.zeros(len(valores_k))
resultados_test = np.zeros(len(valores_k))

# K representa #Vecinos
for k in valores_k:
    
    # Declaramos el tipo de modelo
    clf = KNeighborsClassifier(n_neighbors = k)
    
    # Entrenamos el modelo (con datos de train)
    clf.fit(X_train.iloc[:, pixeles], y_train) 
    
    # Evaluamos el modelo con datos de train y luego de test
    resultados_train[k-1] = clf.score(X_train.iloc[:, pixeles], y_train)
    resultados_test[k-1]  = clf.score(X_test.iloc[:, pixeles] , y_test )

# Performance con tres pixeles (Muy significativos).
plt.plot(valores_k, resultados_train, label = 'Train')
plt.plot(valores_k, resultados_test, label = 'Test')
plt.legend()
plt.title('Performance del modelo de knn (3 pixeles)')
plt.xlabel('Cantidad de vecinos')
plt.ylabel('Precisión')
plt.xticks(valores_k)
plt.ylim(0.95,1.00)
plt.savefig('3pixeles.png', dpi = 400)
plt.show()

# 10 pixeles, tomando primero aquellos de mayor varianza.
valores_k = range(1, 10)
pixeles = [300, 273, 245, 301, 298, 272, 270, 326, 329, 298]

resultados_train = np.zeros(len(valores_k))
resultados_test = np.zeros(len(valores_k))

# K representa #Vecinos
for k in valores_k:
    
    # Declaramos el tipo de modelo
    clf = KNeighborsClassifier(n_neighbors = k)
    
    # Entrenamos el modelo (con datos de train)
    clf.fit(X_train.iloc[:, pixeles], y_train) 
    
    # Evaluamos el modelo con datos de train y luego de test
    resultados_train[k-1] = clf.score(X_train.iloc[:, pixeles], y_train)
    resultados_test[k-1]  = clf.score(X_test.iloc[:, pixeles] , y_test )

# Performance con tres pixeles (Muy significativos).
plt.plot(valores_k, resultados_train, label = 'Train')
plt.plot(valores_k, resultados_test, label = 'Test')
plt.legend()
plt.title('Performance del modelo de knn (10 pixeles)')
plt.xlabel('Cantidad de vecinos')
plt.ylabel('Precisión')
plt.xticks(valores_k)
plt.ylim(0.98,1.005)
plt.savefig('10pixeles.png', dpi = 400)
plt.show()

# Probemos, con 3 atributos aleatorios.
cant_vecinos = range(1,10)
k = len(cant_vecinos) + 1

resultados_train, resultados_test = vecinosAleatorios(k, 20, 3, X, Y)

# Graficamos los resultados anteriores.
sns.set_style('whitegrid')

plt.plot(cant_vecinos, resultados_train, label = 'Train')
plt.plot(cant_vecinos, resultados_test, label = 'Test')
plt.legend()
plt.title('Performance del modelo Knn (3 pixeles aleatorios)')
plt.xlabel('Cantidad de atributos')
plt.ylabel('Precisión')
plt.xticks(cant_vecinos)
plt.ylim(0.60,1.05)
plt.savefig('3pixelesRandom.png', dpi = 400)
plt.show()

# Para cerrar, podemos probar con 10 pixeles.
resultados_train, resultados_test = vecinosAleatorios(k, 20, 10, X, Y)

plt.plot(cant_vecinos, resultados_train, label = 'Train')
plt.plot(cant_vecinos, resultados_test, label = 'Test')
plt.legend()
plt.title('Performance del modelo Knn (10 pixeles aleatorios)')
plt.xlabel('Cantidad de atributos')
plt.ylabel('Precisión')
plt.xticks(cant_vecinos)
plt.ylim(0.95,1.01)
plt.savefig('10pixelesRandom.png', dpi = 400)
plt.show()

# Elimino variables sin usar.
del clf
del df_al
del i, k
del pixeles, maxima_varianza
del resultados_test, resultados_train
del valores_k, y_train

#%% Clasificación multiclase

# Filtro las vocales a,e,i,o,u 
Xvocal = df_sign[df_sign["label"].isin([0,4,8,14,20])]
Yvocal = Xvocal[["label"]]

# Separamos los label
Xvocal = Xvocal.drop(columns = ['label'])


# Dividimos en test(30%) y train(70%)
X_train, X_test, Y_train, Y_test = train_test_split(Xvocal, Yvocal, 
                                                    test_size = 0.3,
                                                    shuffle=True,
                                                    random_state=50)

# La idea, es no utilizar los datos de test sino hasta el final.
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, 
                                                    test_size = 0.3,
                                                    shuffle=True,
                                                    random_state = 13)

# Arboles de diferentes profundidades.
profundidad = range(1,15)
resultados_val  = np.zeros(len(profundidad))
resultados_train = np.zeros(len(profundidad))

for i in profundidad:
    # Armo el arbol de profundidad i y lo entreno.
    arbol= DecisionTreeClassifier(max_depth = i)
    arbol.fit(X_train, Y_train)
    
    # Genero y guardo la precision de mi modelo
    Y_pred = arbol.predict(X_val)
    precision = accuracy_score(Y_val, Y_pred)
    
    # Guardo resultados de train y validation.
    resultados_train[i-1] = arbol.score(X_train, Y_train)
    resultados_val[i-1]  = precision
    
    print(f"Precisión para árbol con profundidad {i}: {precision}")

plt.plot(profundidad, resultados_train, label = 'Train')
plt.plot(profundidad, resultados_val, label = 'Validation')
plt.legend()
plt.title('Performance del Arbol de decision')
plt.xlabel('Nivel de profundidad')
plt.ylabel('Accuracy')
plt.xticks(profundidad)
plt.ylim(0.35,1.05)
plt.savefig('primer_prueba_arbol.png', dpi = 400)
plt.show()

# Notamos que a partir de k = 10, la precision se estabiliza notoriamente.
# Cuando tenemos modelos con performance similares, nos quedamos siempre con el mas sencillo.

# Recuperamos el conjunto de train original.
X_train = X_train.append(X_val, ignore_index=True)
Y_train = Y_train.append(Y_val, ignore_index=True)

# Elimino variables que no uso.
del Xvocal, Yvocal
del X_val, Y_val

# Hacemos el mismo test, pero esta vez utilizando k-fold cross validation.
profundidad = range(1,15)
resultados_cross = np.zeros(len(profundidad))

for i in profundidad:
    arbol=DecisionTreeClassifier(max_depth = i)
    arbol.fit(X_train,Y_train)
    
    # cross_val_score(arbol,X_train,Y_train, cv=5) La comente porque no entendia para q esta.
    score = cross_val_score(arbol, X_train, Y_train, cv=5).mean() 
    resultados_cross[i-1]  = score
    
    print(f"Rendimiento para profundidad {i} :  {score}")

plt.plot(profundidad, resultados_cross)
plt.title('Performance del Arbol de decision con K-Fold Cross Validation')
plt.xlabel('Nivel de profundidad')
plt.ylabel('Rendimiento')
plt.xticks(profundidad)
plt.ylim(0.35,1.05)
plt.savefig('KFold_prueba_arbol.png', dpi = 400)
plt.show()  

# El resultado es similar, a partir de ~10 se es estabiliza.

# Graficamos los resultados con kfold y los de resultado_val.
plt.plot(profundidad, resultados_cross, label = 'K-fold Cross Validation')
plt.plot(profundidad, resultados_val, label = 'Validation')
plt.title('Performance del Arbol de decision')
plt.xlabel('Nivel de profundidad')
plt.ylabel('Accuracy')
plt.xticks(profundidad)
plt.ylim(0.35,1.04)
plt.legend()
plt.savefig('tercer_Grafico_relacion l.png', dpi = 400)
plt.show()

# Para buscar el mejor modelo exploramos distintas combinaciones de hiperparametros.
# Ya vimos antes que alturas eran mas relevantes por eso tomamos solamente un subconjunto.
hyper_params = {'criterion' : ["gini", "entropy"],
                 "max_depth" : [10, 11, 12, 13, 14] } 

# Realizamos un Grid Search.
arbol= DecisionTreeClassifier()
# Notemos que, al no aclarar el parametro cv, por default realiza un Stratified KFold con k = 5.
clf = GridSearchCV(arbol, hyper_params)
clf.fit(X_train, Y_train)

# Pedimos los mejores parametros. 
clf.best_params_
clf.best_score_

# Los resultados del grid search se pueden ver una tabla
resultados = pd.DataFrame(clf.cv_results_)
resultados = resultados[['param_criterion', 'param_max_depth', 'mean_test_score']]

# Renombro las columnas que me interesan.
resultados.rename(columns = {'param_criterion' : 'Criterio',
                             'param_max_depth' : 'Altura',
                             'mean_test_score': 'Precision'},
                  inplace = True)

resultados = resultados.round(4)

# Exporto la tabla.
render_mpl_table(resultados, header_columns=0, col_width=6.2)

# Recall/Precision
vocalesB = np.array(['A','E','I','O','U'])
print(classification_report(Y_test, Y_pred, target_names=vocalesB)) # recall/precision

#%% Genero el mejor arbol a nuestro criterio.
arbol = DecisionTreeClassifier(criterion='entropy', max_depth= 10) 
arbol.fit(X_train, Y_train)

# Una vez, ya estamos seguros de que tenemos el mejor modelo, hacemos una prueba con los datos de test.
Y_pred = arbol.predict(X_test)
arbol.score(X_test, Y_test)

# Grafico el modelo
dot_data = export_graphviz(arbol, out_file=None, 
                           feature_names= X_train.columns,
                           class_names= vocalesB,
                           filled=True, rounded=True,
                           special_characters=True)

graph = graphviz.Source(dot_data) 
graph.render("Que_vocal_es", format= "png")

# Armamos matriz de confusion para el mejor modelo
ConfusionMatrixDisplay.from_estimator(arbol, X_test, Y_test,
                                      display_labels=vocalesB)

plt.tight_layout()
plt.xlabel("Vocal predecida")
plt.ylabel("Vocal verdadera")
plt.title('Matriz de confusión sobre datos de Test')
plt.gcf().set_size_inches(7,6)
plt.savefig('matriz_confusion_arbol.png', dpi = 400)
plt.show()

# Eliminamos las variables que ya no usamos
del arbol, clf, hyper_params
del precision, profundidad
del resultados, resultados_train, resultados_cross, resultados_val
del score
del vocalesB, i
del X_test, Y_test, X_train, Y_train, Y_pred