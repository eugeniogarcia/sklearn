# Decission Trees

Los árboles de decisión son árboles binarios que permiten clasificar los datos en una u otra categoría. El árbol se construye desde un nodo _raíz_ que se va bifurcando hasta llegar a nodos _hoja_. En cada nodo tenemos una serie de parámetros:
- Número total de instancias que se han clasificado por debajo de él. Llamemoslas _m_
- KPI de pureza del nodo - típicamente medida con la función __gini__. Determina cuan puro es el nodo. Un nodo _hoja_ será aquél para el que no podemos dividir las _m_ instancias en dos grupos con un gini menor
- La distribución de las _m_ instancias en las distintas clases
- La clase predominante

Podemos representar gráficamente el árbol asociado a un determinado árbol, `tree_clf`, con `export_graphviz`. 

```py
from sklearn.tree import export_graphviz

export_graphviz(tree_clf,
    out_file=image_path("iris_tree.dot"),
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)
```
## Clasificación (DecisionTreeClassifier)

Podemos usar decission trees para clasificar datos. Para construir el árbol se usa el algoritmo __CART__. El algoritmo procesa los datos dividiendo cada conjunto en dos bloques de forma que se minimice una función de coste que calcula el __gini__ o la __entropría__. El algoritmo divide de forma recursiva las instancias hasta que se alcance el nivel más profundo - no se pueda devidir más las instancias, o no se consiga mejorar el gini/entropía con una división adicional.

El gini se mide como:

```
gini = 1 - suma ( prob_clase_i^2 ) = 1 - suma ( (num_inst_clase_i / num_tot_inst)^2 )
```

Cuando hacemos una división, el Gini resultante de la división se calcula como sigue:

```
Gini_division = Gini_left * (num_int_left / num_tot_inst) + Gini_right * (num_int_right / 
num_tot_inst)
```

Si tenemos un nodo con un _num_tot_inst_ y un gini _G_, y podemos dividir las instancias en dos grupos que dan lugar a un _Gini_division_ < _G_, entonces procedemos a la división. Sino es así, el nodo será un nodo _hoja_.

Este tipo de algoritmos van a producir on over-fit a menos que restrinjamos el universo de exploración - eventualmente podríamos tener tantas hojas como instancias. Para hacer esta restricción usamos los hiper-parámetros del modelo:

- __criterion__. Por defecto `gini`. También se admite `entropy`
- __splitter__. Por defecto 'best'. Admite `best` y `random`. Con este parámetro definimos la estratégia a seguir para dividir cada nodo. Con _best_ es un __greedy algorithm__. Con _random_ introducimos la posibilidad de __explorar__ 
- __max_depth__. Por defecto None. Determina la máxima profundidad del árbol
- __min_samples_split__. Por defecto 2. Valor mínimo de muestras que se requieren para que un nodo se pueda dividir en dos. Sino se supera el valor mínimo, el nodo se convertirá en _leaf_. Si se informase un decimal en lugar de un entero, el número mínimo de muestras para poder dividir un nodo se calculará como _ceil(min_samples_split * n_samples)_
- __min_samples_leaf__. Por defecto 1. Mínimo número de muestras que se requieren en un nodo para que se le considere _leaf_. Para poder hacer una división, en cualquier nivel del árbol, los nodos resultantes a derecha e izquierda tienen que tener al menos este número de muestras. Si se informase un decimal en lugar de un entero, el número mínimo de muestras para poder dividir un nodo se calculará como _ceil(min_samples_leaf * n_samples)_
- __min_weight_fraction_leaf__. Por defecto 0.0. Similar a _min_samples_leaf_, pero expresado en porcentaje sobre el total de las muestras - se puede especificar el parámetro _sample_weight_ si no todas las muestras tienen el mismo peso
- __max_leaf_nodes__. Por defecto None. El árbol irá creciendo hasta el punto en el que se supere este número de nodos _leaf_. Los nodos que se eligen son los que más mejoran la pureza gini/entropía
- __max_features__. Por defecto None. Máximo número de features que se explorarán para decidir un split. Antes de que este parámetro se tenga en cuenta tiene que haber al menos una partición válida. Si se informase un decimal en lugar de un entero, el número mínimo de muestras para poder dividir un nodo se calculará como _ceil(max_features * n_samples)_. Admite los siguientes valores:
    - 'auto'. max_features=sqrt(n_features)
    - 'sqrt'. max_features=sqrt(n_features)
    - 'log2'. max_features=log2(n_features)
    - 'None'. max_features=n_features
- __random_state__ int, RandomState instance, default=None
Controls the randomness of the estimator. The features are always randomly permuted at each split, even if splitter is set to "best". When max_features < n_features, the algorithm will select max_features at random at each split before finding the best split among them. But the best found split may vary across different runs, even if max_features=n_features. That is the case, if the improvement of the criterion is identical for several splits and one split has to be selected at random. To obtain a deterministic behaviour during fitting, random_state has to be fixed to an integer. See Glossary for details.
- __min_impurity_decrease__. Por defecto 0.0. Un nodo se dividirá en dos siempre que el gini/entropía se decrementen al menos en este valor

Podemos ver un ejemplo:

```py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)

clf.fit(X_train, y_train)

clf.predict_proba([[5, 1.5]])
```

## Regresión (DecisionTreeRegressor)

Con __DecisionTreeRegressor__ podemos crear un árbol para hacer la regresión. Es un mecanismo muy parecido al usado en el _DecisionTreeClassifier_. La diferencia es que a los nodos no se les asocia una clase sino un valor. __En lugar de optimizar el gini/entropía, lo que optimizaremos con cada división es el MSE__.

Los parámetros disponibles son los mismos que en el _DecisionTreeClassifier_, con la excepción de:

- __criterion__. Por defecto 'mse'. Se admite:
- 'mse' - error cuadrático
- 'friedman_mse'
- 'mae' - error absoluto

Podemos ver un ejemplo:

```py
from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor(max_leaf_nodes=3, random_state=0)

reg.fit(X_train, y_train)

reg.predict([[5, 1.5]])
```