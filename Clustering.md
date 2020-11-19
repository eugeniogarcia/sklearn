# Clustering

## KMeans

Agrupa los datos alreadedor de varios clusters. Hay que especificar el número de clusters. Por ejemplo, en este caso elegimos 5:

```py
from sklearn.cluster import KMeans

k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)
```

Podemos ver como los datos se agruparán alreadedor de alguno de los clusters. Notese que la salida es el índice del centroide. La primera instancia esta asociada al índice 0, por ejemplo:

```py
y_pred

array([0, 4, 1, ..., 2, 1, 4])
```

Podemos ver la etiqueta que tiene cada instancia asociada. Es lo mismo que tenemos asociado a las predicciones - * con *y_pred* en nuestro ejemplo:

```py
kmeans.labels_

array([0, 4, 1, ..., 2, 1, 4])
```

Y cuales son las coordenadas de cada uno de los centroides:

```py
kmeans.cluster_centers_

array([[-2.80037642,  1.30082566],
       [ 0.20876306,  2.25551336],
       [-2.79290307,  2.79641063],
       [-1.46679593,  2.28585348],
       [-2.80389616,  1.80117999]])
```

Con *transform* lo que obtendremos es la distancia de cada instancia a *cada uno* de los centroides:

```py
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])

kmeans.transform(X_new)

array([[2.88633901, 0.32995317, 2.9042344 , 1.49439034, 2.81093633],
       [5.84236351, 2.80290755, 5.84739223, 4.4759332 , 5.80730058],
       [1.71086031, 3.29399768, 0.29040966, 1.69136631, 1.21475352],
       [1.21567622, 3.21806371, 0.36159148, 1.54808703, 0.72581411]])
```

Veamos que efectivamente cuando hacemos la predicción, lo que obtenemos es el centroide más cercano:

```py
kmeans.predict(X_new)

array([1, 1, 2, 2])
```

Podemos ver cual es el score del modelo con los datos de entrenamiento:

```py
kmeans.score(X)

-211.5985372581683
```

El score es la inercia con signo negativo. La inercia es la suma de los cuadrados de las distancias de cada instancia al centroide más próximo:

```py
kmeans.inertia_

211.5985372581683
```

Podemos ver que el score de los datos de prueba:

```py
kmeans.score(X_new)

-8.180246021073435
```

Es efectivamente la suma de los cuadrados de las distancias:

```py
kmeans.transform(X_new)[0,1]**2+kmeans.transform(X_new)[1,1]**2+kmeans.transform(X_new)[2,2]**2+kmeans.transform(X_new)[3,2]**2

8.18024602107344
```

### Inicialización de pesos

En el ejemplo anterior los centroides se eligieron al azar. Fue equivalente a:

```py
kmeans_rnd_10_inits = KMeans(n_clusters=5, init="random", n_init=10, algorithm="full", random_state=11)

kmeans_rnd_10_inits.fit(X)
```

Se pueden especificar centroides de inicio con el parámetro *init*. Podemos especificar también el número de entrenamientos que se harán con *n_init*.

```py
good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
kmeans = KMeans(n_clusters=5, init=good_init, n_init=10)
```

Con este ejemplo ejecutaremos el algoritmo 10 veces cuando llamemos a fit(). Scikit-Learn utilizará la mejor solución de las 10 intentadas, entendida como mejor aquella que tiene la mejor _inercia_. La _inercia_ se define como la media de la distancia de cada instancia con su centroide más próximo. Podemos ver la inercia:

```py
kmeans.inertia_

211.59853725816856
```

El score estará disponible, como en el resto de predictores. En este caso será la inercia con signo negativo:

```py
kmeans.score(X)

-211.59853725816856
```

### K-Means acelerado

Podemos acelerar el algoritmo k-means si simplificamos los cálculos de la distancia, aprovechando que dados tres puntos A, B y C, _AC < AB + BC_. En esto consiste el algoritmo _elkan_:

```py
KMeans(algorithm="elkan").fit(X)
```

### Minibatchs

Como el nombre sugiere, en minibatchs lo que hacemos es entrenar el modelo que lotes de instancias elegidas al azar. En lugar de usar todas las instancias en cada interaccion de entrenamiento, se utiliza un subconjunto elegido al azar - con o sin reemplazamiento. En sklearn hay una clase especial:

```py
from sklearn.cluster import MiniBatchKMeans

minibatch_kmeans = MiniBatchKMeans(n_clusters=5)
minibatch_kmeans.fit(X)
```

### Elegir número de cluster

#### inercia

Para elegir el mejor cluster podemos ir haciendo clasificaciones con diferente número de centroides, y calcular la inercia en cada caso. La inercia mejorará al incrementar el número de centroides, pero despues de una mejora sustancia, entraremos en tasas de mejora más limitadas. Este "codo" en el que la tasa de mejora deje de ser tan pronunciada, puede determinar el número de centroides a usar.

#### silhouette

Otra forma de determina el número de centroides es usar el silhouette coefficient:

*(b – a) / max(a, b)*

Define la distancia media de una instancia con el resto de instancias del cluster. 

*a*: distancia media de una instancia con el resto de instancias del cluster
*b*: Distancia media al cluster más próximo

El silhouette coefficient varía entre -1 y +1. +1 nos indica que las instancias están "bien dentro" de su propio cluster. -1 que las instancias no están bien clasificadas. 0 indicará que están cerca de la frontera del cluster.

Podemos calcular la silhouette:

```py
from sklearn.metrics import silhouette_score

silhouette_score(X, kmeans.labels_)
0.655517642572828
```
### Casos de Uso

- __Image color segmentation__. Podemos tomar una imagen, y reducir sus colores a número fijo. Usamos tantos centroides como colores queremos tener, y reemplazamos cada color por su centroide

```py
X = image.reshape(-1, 3)

kmeans = KMeans(n_clusters=8, random_state=42).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)
```

- __Clustering as preprocessing tool__
- __SemiSupervised learning__