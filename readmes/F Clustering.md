# 1 Clustering

## 1.1 KMeans

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

### 1.1.1 Inicialización de pesos

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

### 1.1.2 K-Means acelerado

Podemos acelerar el algoritmo k-means si simplificamos los cálculos de la distancia, aprovechando que dados tres puntos A, B y C, _AC < AB + BC_. En esto consiste el algoritmo _elkan_:

```py
KMeans(algorithm="elkan").fit(X)
```

### 1.1.3 Minibatchs

Como el nombre sugiere, en minibatchs lo que hacemos es entrenar el modelo que lotes de instancias elegidas al azar. En lugar de usar todas las instancias en cada interaccion de entrenamiento, se utiliza un subconjunto elegido al azar - con o sin reemplazamiento. En sklearn hay una clase especial:

```py
from sklearn.cluster import MiniBatchKMeans

minibatch_kmeans = MiniBatchKMeans(n_clusters=5)
minibatch_kmeans.fit(X)
```

### 1.1.4 Elegir número de cluster

#### 1.1.4.1 inercia

Para elegir el mejor cluster podemos ir haciendo clasificaciones con diferente número de centroides, y calcular la inercia en cada caso. La inercia mejorará al incrementar el número de centroides, pero despues de una mejora sustancia, entraremos en tasas de mejora más limitadas. Este "codo" en el que la tasa de mejora deje de ser tan pronunciada, puede determinar el número de centroides a usar.

#### 1.1.4.2 silhouette

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
### 1.1.5 Casos de Uso

#### 1.1.5.1 Image color segmentation

Podemos tomar una imagen, y reducir sus colores a número fijo. Usamos tantos centroides como colores queremos tener, y reemplazamos cada color por su centroide. Esto es, si tenemos 8 clusters, la imagen la transformamos a 8 colores

```py
# image es (:,64,3), X es (:,3)
X = image.reshape(-1, 3)

kmeans = KMeans(n_clusters=8, random_state=42).fit(X)
# kmeans.labels_ tiene la clasificación de cada una de las X imagenes
# segmented_img tendrá una imagen que tiene como color el centroida (:,3)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
# (:,64,3)
segmented_img = segmented_img.reshape(image.shape)
```

La imagen *segmented_img* pasa a ser una imagen simplificada de la original, en la que tenemos solo 8 colores diferentes.

#### 1.1.5.2 Clustering as preprocessing tool

Podemos preprocesar los datos, de modo que en lugar de clasificar las instancias, clasifiquemos los centroides. Veamos un ejemplo. Podemos aplicar logistic regression para calcular la probilidad de que una instancia sea un dígito u otro:

```py
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

X_digits, y_digits = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)

log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)

0.9688888888888889
```

Podemos preprocesar los datos con este pipeline. *X_train* es (:,64)

```py
pipeline = Pipeline([
    ("pepe", KMeans(n_clusters=50, random_state=42)),
    ("log_reg", LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)),
])
pipeline.fit(X_train, y_train)

pipeline.score(X_test, y_test)

0.98
```

El primer paso transorma (:,64) en (:,50). Cada una de las 50 features es la distancia de la instancia al centroide. El segundo paso clasificará (:,50), calculando las probabilidades de cada una de las categorias.

Con este método hemos mejorado la precisión en unos puntos.
##### Optimización

Podemos usar GridSearchCV para averiguar cual es el número de centroides óptimo. Definimos el pipeline
- Kmean
- Logistic regression

```py
pipeline = Pipeline([
    ("pepe", KMeans(n_clusters=50, random_state=42)),
    ("log_reg", LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)),
])
pipeline.fit(X_train, y_train)

pipeline.score(X_test, y_test)
```

Podemos buscar el valor óptimo para el número de centroides:

```py
from sklearn.model_selection import GridSearchCV

param_grid = dict(pepe__n_clusters=range(30, 40))
grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_clf.fit(X_train, y_train)

grid_clf.score(X_test, y_test)
```

#### 1.1.5.3 SemiSupervised learning (opción 1)

Con este método vamos a partir de un conjunto de dígitos que no estan etiquetados.

```py
n_labeled = 50

log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", random_state=42)
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
log_reg.score(X_test, y_test)
```

El método anterior nos da un score del 83%. Si en lugar de clasificar la instancia lo que hacemos es:
- definir 50 centroides y agrupar las instancias alrededor de los centroides
- identificar la instancia que esta más cerca de cada centroide, y considerar que esa instancia es la instancia representativa del centroide
- etiquetar la instancia representativa. Son "solo" 50 instancias a clasificar

Con este procedimiento en lugar de tener que etiquetar todas las instancias, lo que hacemos es agrupar las instancias en clusters, elegir una instancia como la representativa del cluster, y etiquetar esa instancia. Lo que hacemos a continuación es clasificar estas 50 instancias clasificadas

```py
k = 50

kmeans = KMeans(n_clusters=k, random_state=42)
# Transformamos X_train de (:,60) a (:,50), donde cada una de las features es la distancia de la instancia a cada uno de los 
#50 centroides
X_digits_dist = kmeans.fit_transform(X_train)

# transoformamos (:,50) en (50). Lo que tendremos en cada una de los 50 valores, es el índice de la imagen que más cercana
# esta al centroide. De esta manera en representative_digit_idx tenemos el indice de la imagen que más cerca esta a cada 
#centroide 
representative_digit_idx = np.argmin(X_digits_dist, axis=0)
#En X_representative_digits tendremos la imagen más representativa, la que esta más cerca de cada centroide
X_representative_digits = X_train[representative_digit_idx]
#Etiquetamos las instancias
y_representative_digits=np.array([0,1,3,2,7,6,4,6,9,5,1,2,9,5,2,7,8,1,8,6,3,1,5,4,5,4,0,3,2,6,1,7,7,9,1,8,6,5,4,8,5,3,3,6,7,9,7,8,4,9])
```

2. Usamos estas instancias clasificadas para clasificar

```py
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_representative_digits, y_representative_digits)
log_reg.score(X_test, y_test)
```

#### 1.1.5.4 SemiSupervised learning (opción 2). Propagar las etiquetas

Una variante sobre el método anterior seria entrenar la logistic regression con todas las instancias, pero en lugar de etiquetarlas manualmente, aplicar a todas las instancias que "pertenezcan" a un centroide, la etiqueta representativa de ese centroide.

```py
#Dimensionamos correctamente nuestro vector de etiquetas
y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
    #Asignamos a todas las imagenes del cluster k, la etiqueta del cluster k 
    y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]
```

El entrenamiento ya se hace con el juego completo de imagenes:

```py
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
# Entrenamos con el juego completo de imagenes
log_reg.fit(X_train, y_train_propagated)
```

Podemos optar por un término medio, y en lugar de usar todas las instancias usar solo las que están más próximas a los centroides.

## 1.2 DBSCAN

Se trata de otro algoritmo no supervisado de segmentación. En este algoritmo no se tiene que especificar el número de clusters, pero por el contrario se define _epsilon_, el radio alrededor de una instancia en la que se buscará la presencia de otras instancias. Sino se encuentra ninguna, se considerará que la instancia es una _anomalía_.

Si encontramos instancias, la instancia se considera como parte de un *e-neighbourhood*. Cuando dentro del vecindario encontramos más de *min_instances*, se considera a la instancia como una *core_instance*. Todas las instancias que estén en el vecindario de una *core_instance* formarán parte del mismo cluster.

```py
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.05)

dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X)
```

- instancia core de cada instancia. *dbscan.labels_*. Cuando el valor es *-1*, significa que la instancia es una anomalía, y que no tiene una instancia core

```py
dbscan.labels_
array([ 0, 2, -1, -1, 1, 0, 0, 0, ..., 3, 2, 3, 3, 4, 2, 6, 3])
```

- instancias. *X*

```py
X[1]

[ 0.97670045 -0.45832306]
```

- instancias core. *dbscan.components_*

```py
dbscan.components_[dbscan.labels_[1]]

[ 0.58930337 -0.32137599]
```

- instancia a la que corresponde cada core. *dbscan.core_sample_indices_*

```py
print(dbscan.core_sample_indices_[dbscan.labels_[1]])

5
```

```py
X[5])

[ 0.58930337 -0.32137599]
```

Vemos que la instancia 1, *[ 0.97670045 -0.45832306]*, pertenece al cluster *[ 0.58930337 -0.32137599]*. Este cluster esta creado alrededor de la instancia 5.

Podemos ver cuantas instancias core hay:

```py
np.unique(dbscan.labels_)

array([-1,  0,  1,  2,  3,  4,  5,  6], dtype=int64)
```

## 1.3 Gaussian

Con esta técnica determinamos la relación de los clusters con las instanacias sigue una distribución normal. La probabilidad de que una instancia pertenezca a un determinado cluster será la probabilidad relativa de pertenecer a cada uno de los cluster. Cada cluster tiene un peso sobre el total, esto, junto con la posición de la instancia con respecto a la los centroiders y la covariancia de cada uno, nos permite calcular la probabilidad de que la instancia pertenezca a cada uno de los clusters:

```py
# Tres centroides, y 10 iteracciones para adiestrar
gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
gm.fit(X)

# Los pesos relativos de cada cluster
gm.weights_

array([0.39054348, 0.2093669 , 0.40008962])

# La media de cada cluster
gm.means_

array([[ 0.05224874,  0.07631976],
       [ 3.40196611,  1.05838748],
       [-1.40754214,  1.42716873]])
       
#La covariancia de cada cluster
gm.covariances_

array([[[ 0.6890309 ,  0.79717058],
        [ 0.79717058,  1.21367348]],

       [[ 1.14296668, -0.03114176],
        [-0.03114176,  0.9545003 ]],

       [[ 0.63496849,  0.7298512 ],
        [ 0.7298512 ,  1.16112807]]])
```

El modelo, como el K-means, se entrena de forma iterativa - en nuestro caso hasta 10 iteracciones -, hasta converger. Podemos ver si ha convergido, y en cuantas interacciones:

```py
gm.converged_

gm.n_iter_
```

Podemos usar el modelo en modo _hard clustering_ - nos dice que cluster a que cluster pertenece la instancia -, o _soft clustering_ - nos dice la probabilidad de pertenecer a uno u otro cluster:

```py
gm.predict(X)

array([0, 0, 2, ..., 1, 1, 1], dtype=int64)

# Probabilidad de que X pertenezca a cada uno de los clusters
gm.predict_proba(X)

array([[9.77227791e-01, 2.27715290e-02, 6.79898914e-07],
       [9.83288385e-01, 1.60345103e-02, 6.77104389e-04],
       [7.51824662e-05, 1.90251273e-06, 9.99922915e-01],
       ...,
       [4.35053542e-07, 9.99999565e-01, 2.17938894e-26],
       [5.27837047e-16, 1.00000000e+00, 1.50679490e-41],
       [2.32355608e-15, 1.00000000e+00, 8.21915701e-41]])
```

Para facilitar podemos imponer restricciones en la forma de la covarianza. Podemos especificar la forma usando *covariance_type*. Por defecto el valor es _full_, podemos usar otros:

```py
gm_full = GaussianMixture(n_components=3, n_init=10, covariance_type="full", random_state=42)
gm_tied = GaussianMixture(n_components=3, n_init=10, covariance_type="tied", random_state=42)
gm_spherical = GaussianMixture(n_components=3, n_init=10, covariance_type="spherical", random_state=42)
gm_diag = GaussianMixture(n_components=3, n_init=10, covariance_type="diag", random_state=42)

gm_full.fit(X)
gm_tied.fit(X)
gm_spherical.fit(X)
gm_diag.fit(X)
```

Este parametro define la forma que tendrá el cluster.

### 1.3.1 Anomaly Detection

Podemos usar el modelo gaussiano para detectar outlayers, anomalías. *score_samples()* retorna el logaritmo de la probabilidad de cada muestra, poderado.

Si por ejemplo queremos identificar outlayers con el 4%, obtenemos el percentil 4, e identificamos las instancias que tienen un score inferior a este threshold:

```py
# Tres centroides, y 10 iteracciones para adiestrar
gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
gm.fit(X)

densities = gm.score_samples(X)
density_threshold = np.percentile(densities, 4)
anomalies = X[densities < density_threshold]
```