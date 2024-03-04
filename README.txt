# Trabajo Práctico 02: Clasificación y validación cruzada

## Bibliotecas necesarias:

- numpy
- pandas
- matplotlib.pyplot
- scikit-learn
- six
- graphviz

De la biblioteca scikit-learn será necesario importar los siguientes módulos con dichas funciones:

- from sklearn.model_selection: train_test_split, GridSearchCV, cross_val_score
- from sklearn.neighbors: KNeighborsClassifier
- from sklearn.tree: DecisionTreeClassifier,export_graphviz
- from sklearn.metrics: accuracy_score, ConfusionMatrixDisplay, classification_report

## Instrucciones del código

El código esta separado en distintos bloques, cada uno debe ser corrido individualmente para generar gráficos, reportes e información de cada fragmento.

Los dos ultimos bloques corresponden respectivamente a: 
- Función necesaria para obtener el gráfico de la tabla 
- Generación del mejor modelo del arbol de decision junto a la matriz de confusión. 