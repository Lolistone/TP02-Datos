# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 19:52:48 2024

@author: lolocrsn 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import six
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier,plot_tree,export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score # lo uso para la precision
from sklearn.metrics import ConfusionMatrixDisplay, classification_report # genera matriz confusion

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
del ax, axe

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
     
plt.savefig('letra_C.png', dpi = 400)
plt.show()
plt.close(fig)     

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

# Podemos ver ahora, cuales son exactamente los atributos que identifican a la leta C. Y, además
# determinar con esto donde esta la mayor semejanza entre estas.

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

# Podemos observar que la cantidad de estas letras esta bastante balanceada.

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

plt.show()
plt.close(fig)
# Con esta imagen podemos ver, claramente donde están los pixeles mas significativos.

# Borro las variables que ya no necesito.
del Xa, Xl, restaAL
del cantA, cantL
del ax, axe
del i, label
del fig

# Separo los datos a predecir
Y = df_al['label']
X = df_al.drop(['label'], axis=1)

# Separamos en train y test. Utilizo shuffle para garantizar alternancia de clases.
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, shuffle = True, random_state=0)

# Armamos el modelo
clf = KNeighborsClassifier(n_neighbors=3)

# Elijo que pixeles usar. Obviamente, teniendo encuenta la lista 'maxima_varianza'
pixeles = [[300, 273, 245], [301, 298, 272], [270, 327, 328]]
resultados = []
for i in range(len(pixeles)):
    X_train_1 = X_train.iloc[:, pixeles[i]]
    X_test_1 = X_test.iloc[:,pixeles[i]]

    # Entrenamos el modelo.
    clf.fit(X_train_1, y_train)
    
    # Evaluamos la exactitud.
    print("Test set accuracy: {:.2f}".format(clf.score(X_test_1, y_test)))
    
    # Guardo los resultados
    resultados += [round(clf.score(X_test_1, y_test), 2)]

# Utilizando los pixeles de mayor varianza, obtenemos resultados realmente buenos.

# Ahora, probemos, para distintas cantidades de atributos, distintos valores de k.

# Rango de valores por los que se va a mover k
valores_k = range(1, 10)

resultados_test  = np.zeros(len(valores_k))
resultados_train = np.zeros(len(valores_k))

X_train_1 = X_train.iloc[:, pixeles[0]]
X_test_1 = X_test.iloc[:,pixeles[0]]

for k in valores_k:
    # Declaramos el tipo de modelo
    clf = KNeighborsClassifier(n_neighbors = k)
    # Entrenamos el modelo (con datos de train)
    clf.fit(X_train_1, y_train) 
    # Evaluamos el modelo con datos de train y luego de test
    resultados_train[k-1] = clf.score(X_train_1, y_train)
    resultados_test[k-1]  = clf.score(X_test_1 , y_test )

# Performance con tres pixeles.
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

# Probemos ahora con conjuntos mas grandes de pixeles. Por ejemplo, 50 pixeles.
valores_k = range(1, 15)

resultados_test  = np.zeros(len(valores_k))
resultados_train = np.zeros(len(valores_k))

# Intente agarrar pixeles no tan cercanos a los de mayor varianza.
X_train_1 = X_train.iloc[:,600:650]
X_test_1 = X_test.iloc[:,600:650]

for k in valores_k:
    # Declaramos el tipo de modelo
    clf = KNeighborsClassifier(n_neighbors = k)
    # Entrenamos el modelo (con datos de train)
    clf.fit(X_train_1, y_train) 
    # Evaluamos el modelo con datos de train y luego de test
    resultados_train[k-1] = clf.score(X_train_1, y_train)
    resultados_test[k-1]  = clf.score(X_test_1 , y_test )

# Performance con 50 pixeles.
plt.plot(valores_k, resultados_train, label = 'Train')
plt.plot(valores_k, resultados_test, label = 'Test')
plt.legend()
plt.title('Performance del modelo de knn (50 pixeles)')
plt.xlabel('Cantidad de vecinos')
plt.ylabel('Precisión')
plt.xticks(valores_k)
plt.ylim(0.95,1.00)
plt.savefig('50pixeles.png', dpi = 400)
plt.show()

# Parece ser que el mejor k, sin importar la cantidad de pixeles es 1. Probemos con 1 pixel.
valores_k = range(1, 15)

resultados_test  = np.zeros(len(valores_k))
resultados_train = np.zeros(len(valores_k))

X_train_1 = X_train.iloc[:,300:301]
X_test_1 = X_test.iloc[:,300:301]

for k in valores_k:
    # Declaramos el tipo de modelo
    clf = KNeighborsClassifier(n_neighbors = k)
    # Entrenamos el modelo (con datos de train)
    clf.fit(X_train_1, y_train) 
    # Evaluamos el modelo con datos de train y luego de test
    resultados_train[k-1] = clf.score(X_train_1, y_train)
    resultados_test[k-1]  = clf.score(X_test_1 , y_test )

# Performance con 1 pixel.
plt.plot(valores_k, resultados_train, label = 'Train')
plt.plot(valores_k, resultados_test, label = 'Test')
plt.legend()
plt.title('Performance del modelo de knn (1 pixel)')
plt.xlabel('Cantidad de vecinos')
plt.ylabel('Precisión')
plt.xticks(valores_k)
plt.ylim(0.65,1.00)
plt.savefig('1pixelalto.png', dpi = 400)
plt.show()


# Tomando un solo pixel en una zona con bastante varianza, vemos que a medida que hay mas 
# vecinos la precision aumenta. Donde parece que 9 seria un buen k.

# Probemos una zona menos signficativa
valores_k = range(1, 15)

resultados_test  = np.zeros(len(valores_k))
resultados_train = np.zeros(len(valores_k))

# Intente agarrar pixeles no tan cercanos a los de mayor varianza.
X_train_1 = X_train.iloc[:,400:401]
X_test_1 = X_test.iloc[:,400:401]

for k in valores_k:
    # Declaramos el tipo de modelo
    clf = KNeighborsClassifier(n_neighbors = k)
    # Entrenamos el modelo (con datos de train)
    clf.fit(X_train_1, y_train) 
    # Evaluamos el modelo con datos de train y luego de test
    resultados_train[k-1] = clf.score(X_train_1, y_train)
    resultados_test[k-1]  = clf.score(X_test_1 , y_test )

# Performance con 1 pixel.
plt.plot(valores_k, resultados_train, label = 'Train')
plt.plot(valores_k, resultados_test, label = 'Test')
plt.legend()
plt.title('Performance del modelo de knn (1 pixel)')
plt.xlabel('Cantidad de vecinos')
plt.ylabel('Precisión')
plt.xticks(valores_k)
plt.ylim(0.45,0.70)
plt.savefig('1pixelbajo.png', dpi = 400)
plt.show()

# Si bien la precision bajó notablemente, el resultado es el mismo, con un k = 9 el modelo mejora notoriamente.
del clf
del X, X_test, X_test_1, X_train, X_train_1, Y, y_test
del df_al
del i, k
del pixeles, maxima_varianza
del resultados, resultados_test, resultados_train

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

# Exporto la tabla.
render_mpl_table(resultados, header_columns=0, col_width=6.2)

#%% Recall/Precision
vocalesB = np.array(['A','E','I','O','U'])
print(classification_report(Y_test, Y_pred, target_names=vocalesB)) # recall/precision

# Genero el mejor arbol a nuestro criterio.
arbol = DecisionTreeClassifier(criterion='entropy', max_depth= 10) 
arbol.fit(X_train, Y_train)

# Una vez, ya estamos seguros de que tenemos el mejor modelo, hacemos una prueba con los datos de test.
Y_pred = arbol.predict(X_test)
arbol.score(X_test, Y_test)

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

# Grafico el modelo
dot_data = export_graphviz(arbol, out_file=None, 
                           feature_names= X_train.columns,
                           class_names= vocalesB,
                           filled=True, rounded=True,
                           special_characters=True)

graph = graphviz.Source(dot_data) 
graph.render("Que_vocal_es", format= "png")
