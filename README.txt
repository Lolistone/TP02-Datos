## MNIST Classifier

## Bibliotecas necesarias:

- numpy 1.26.4
- pandas 1.5.3
- matplotlib.pyplot 3.8.3
- scikit-learn 1.4.1.post1
- six 1.16.0
- seaborn 0.13.2

De la biblioteca scikit-learn será necesario importar los siguientes módulos con dichas funciones:

- from sklearn.model_selection: train_test_split, GridSearchCV, cross_val_score
- from sklearn.neighbors: KNeighborsClassifier
- from sklearn.tree: DecisionTreeClassifier
- from sklearn.metrics: accuracy_score, ConfusionMatrixDisplay

## Obs: En el archivo utils.py encontramos algunas funciones auxiliares.
## Obs: Por algun motivo, al utilizar grillas en algunos gráfico (con Seaborn), se agregan grillas en graficos que no deberian.
## Por lo tanto, dejo comentado las funciones que utilice para poner las grillas.

## Como correr el código: 

- Primero importamos las librerias, carpeta y datafrme. 

## Obs: Luego de importar las librerias, a la variable carpeta debera asignarle la ruta a donde este su csv.

- Luego, podemos ejecutar cada sección del codigo, dividida en Analisis Explotatorio, Clasificación binaria y Clasificacion multiclase por separado, preferentemente, en el orden antes mencionado. 
