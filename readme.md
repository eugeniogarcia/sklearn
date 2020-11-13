# Algoritmos

Algoritmos de aprendizaje supervisado:
- k-Nearest Neighbors
- Linear Regression
- Logistic Regression
- Support Vector Machines (SVMs)
- Decision Trees and Random Forests
- Neural networks

No supervisado:
- Clustering
	- K-Means
	- DBSCAN
	- Hierarchical Cluster Analysis (HCA)
- Anomaly detection and novelty detection
	- One-class SVM
	- Isolation Forest
- Visualization and dimensionality reduction
	- Principal Component Analysis (PCA)
	- Kernel PCA
	- Locally-Linear Embedding (LLE)
	- t-distributed Stochastic Neighbor Embedding (t-SNE)
- Association rule learning
	- Apriori
	- Eclat

Reinforcement Learning

# Preparación de datos

## Estratificar datos

### StratifiedShuffleSplit

Dividimos el conjunto de datos en un bloque para entrenamiento y otro para test. Esto se controla bien con __test_size__ o con __train_size__. Estos parametros toman un valor comprendido entre 0 y 1.

__random_state__ es la semilla, y es un parametro opcional.

__n_splits__ toma por defecto el valor 10. Indica el número de interaciones de splitting y re-.shufling

```py
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(X, y):
	strat_train_set = X.loc[train_index]
	strat_test_set = X.loc[test_index]
	y_train, y_test = y[train_index], y[test_index]
```

### train_test_split

```py
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
```

## Rellena datos no disponibles (na)

Rellenamos los valores no disponibles en un dataset. Podemos usar diferentes estratégias para rellenar los datos:

- __mean__. Rellena con el valor medio de cada columna. Solo es aplicable a columnas numéricas
- __median__. Rellena con la mediana de cada columna. Solo es aplicable a columnas numéricas
- __most_frequent__. Utiliza el valor más frecuente de cada columna. Puede usarse tanto con columnas numéricas como categóricas
- __constant__. Utiliza el valor indicado en _fill_value_. Puede usarse tanto con columnas numéricas como categóricas. El parámetro fill_value por defecto vale None

Con el parámetro _copy_ indicamos si el cambio se hace "in place", o si se crea una copia del dataset. Por defecto vale True, indicando que se crea una copia del dataset.

```py
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median", copy=True)

imputer.fit(X)
```

Podemos ver los valores que se usan para reemplazar en cada columna con:

```py
imputer.statistics_
array([ -118.51 , 34.26 , 29. , 2119.5 , 433. , 1164. , 408. , 3.5409])
```

Aplicamos la transformación como sigue:

```py
X_new = imputer.transform(X)
```

## Encodig

Cuando tenemos datos categóricos vamos a desear convertirlos en númericos. Podemos optar por tres opciones:
- Codificarlos. Cada categoría se reemplaza por un número - ej. cat1->1, cat2->2, cat3->3.... El problema de este enfoque es que los algoritmos de aprendizaje van a "considerar" que un registro con cat1 es más parecido a un registro con cat2 que a uno con cat3. Para resolver esto, tendríamos que usar una codificación en la que la proximidad equivalga a la codificación usada
- Hot-encoding. En este método no tenemos el problema de "proximidad" descrito en el método anterior. En este método convertimos cada categoría en una feature booleana. El problema es que podemos estar creando un número imposible de features
- Embebded. En este método transformamos cada categoría en un vector numerico, usando un diccionario o corpus de valores. 

### codificación

```py
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()

X_new=ordinal_encoder.fit_transform(X)
```

Podemos ver como en el dataset resultante ya no hay categorías, sino sus codificaciones:

```py
X_new[:10]

array([[0.],
[0.],
[4.],
[1.],
[0.],
[1.],
[0.],
[1.],
[0.],
[0.]])
```

Podemos ver la codificación que se ha usado:

```py
ordinal_encoder.categories_

[array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
dtype=object)]
```

### Hot-encoding

```py
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
X_new = cat_encoder.fit_transform(X)
```

Vemos como el resultado es una matriz, en este caso una matriz _sparse_:

```py
X_new

<16512x5 sparse matrix of type '<class 'numpy.float64'>'
with 16512 stored elements in Compressed Sparse Row format>
```

## Custom Transformers & Estimators

Para implementar una transformación custom tenemos que hacer lo siguiente (sklearn no se apoya en herencia):
- implementar tres métodos:
	- fit(). Debe retornar _self_
	- transform()
	- fit_transform()


Si queremos evitar tener que implementar _fit_transform()_, podemos hacer que nuestro custom transformer herede de __TransformerMixin__.

Si añadimos además __BaseEstimator__, y no usamos _*args_ y _**kargs_ en el constructor, se incluirán los métodos _get_params()_ y _set_params()_ que podrán usarse para automatizar la selección de hyperparameters.

### Ejemplo 1

En este ejemplo vamos a transformar un dataset para añadir dos o tres series de datos, que calcularemos usando las series ya presentes en el dataset:

```py
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
	def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
		self.add_bedrooms_per_room = add_bedrooms_per_room # Guardamos este hiperparametro

	def fit(self, X, y=None):
		return self # no hay que hacer ningún entrenamiento

	def transform(self, X, y=None):
		rooms_per_household = X[:, rooms_ix] / X[:, households_ix] # Calculamos la primera serie
		population_per_household = X[:, population_ix] / X[:, households_ix] # Calculamos la segunda serie

		if self.add_bedrooms_per_room: # Vemos si necesitamos calcular una tercera serie
			bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix] # Calculamos la tercera serie
			return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_room] # Retornamos el dataset transformado
		else:
			return np.c_[X, rooms_per_household, population_per_household] # Retornamos el dataset transformado
			
#Usamos el transformer custom			
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)

housing_extra_attribs = attr_adder.transform(housing.values)
```

### Ejemplo 2

### Ejemplo 3

Creamos un clasificador binario que clasifica al azar, siempre retorna un 0:

```py
from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
	def fit(self, X, y=None):
		pass
	
	def predict(self, X):
		return np.zeros((len(X), 1), dtype=bool)
```

## Escalado

Una forma de acelerar el entrenamiento es estandarizar los valores en el dataset para que todas las features manejen una escala similar. Hay dos métodos que se usan habitualmente:

- Min-max. Convertimos todos los valores al ranto 0-1
- Normalización. Transformamos los valores para que sigan una normal de media cero y desviación 1

### Min-Max

Podemos usar los siguientes parámetros:

- __copy__. Por defecto vale True. Nos indica si devolvemos una copia del dataset, o si hacemos las transformaciones "in place"
- __feature_range_. Es una tuple del tipo _(min, max)_, que por defecto vale (0, 1)

```py
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

scaler.fit(X)
```

Podemos ver los resultados de la transformación con:

```py
scaler.min_

scaler.scale_

scaler.data_min_

scaler.data_max_

scaler.data_range_
```

Aplicamos la transformación:

```py
X_new=scaler.transform(X)
```

### Normalización

Podemos usar los siguientes parámetros:

- __copy__. Por defecto vale True. Nos indica si devolvemos una copia del dataset, o si hacemos las transformaciones "in place"
- __with_mean__. Por defecto vale True. Nos indica si normalizamos usando la media, de modo que la media del dataset resultante sea cero
- __with_std__. Por defecto vale True. Nos indica si normalizamos los datos para que la desviación resultante sea 1


```py
from sklearn.preprocessing import StandardScaler

standard=StandardScaler()

standard.fit(X)
```

Podemos ver los resultados de la transformación con:

```py
standard.scale_

standard.mean_

standard.var_
```

Aplicamos la transformación:

```py
X_new=standard.transform(X)
```

## Pipelines

Podemos combinar varios predictors/transformers/estimators en un pipeline. El pipeline expondrá los métodos del último componente definido. Los componentes se definen con un array de tuplas. La tupla será el nombre + predictors/transformers/estimators:

Definimos un pipeline:
 
```py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),('attribs_adder', CombinedAttributesAdder()),('std_scaler', StandardScaler()),])
```

Lo aplicamos:

```py
housing_num_tr = num_pipeline.fit_transform(housing_num)
```

Podemos combinar varios pipelines. Para definir estos pipelines usamos un array con tupla con el nombre+pipeline+series sobre las que deseamos que aplique:

```py
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs),("cat", OneHotEncoder(), cat_attribs),])

housing_prepared = full_pipeline.fit_transform(housing)
```

# Calidad del modelo

Podemos evaluar la calidad del modelo usando diferentes métricas:

- mean_squared_error
- mean_absolute_error
- cross_val_score

## mean_squared_error

Norm 2 distance of two vectors:

```py
from sklearn.metrics import mean_squared_error

#Estimamos los valores con el modelo
housing_predictions = lin_reg.predict(housing_prepared)

#Calculamos el MSE
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)

lin_rmse
68628.19819848922
```

## mean_absolute_error

Norm 1 distance of two vectors:

```py
from sklearn.metrics import mean_absolute_error

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

mean_absolute_error(y_true, y_pred)
0.5
```

## cross_val_score

Dividimos el dataset en K bloques. Usamos K-1 para adiestrar el modelo, y el último bloque para evaluarlo. Hacemos esta operación K veces, eligiendo en cada ocasión un bloque diferente para la evaluación. La precisión del modelo será la media de las K precisiones evaluadas.

Este método requiere que pasemos como argumento el modelo que debe ser entrenado, así como la serie de datos.

```py
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)
```

Podemos evaluar la precisión:

```py
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,scoring="accuracy", cv=10)
```

### Custom cross-validation

Si por las razones que fueran tuvieramos que ejercer un mayor control sobre como se calcula el cross-validation, podríamos hacer una implementación custom. __StratifiedKFold__ nos ofrece un muestreo estratificado. Veamos como hacerlo con un ejemplo:

```py
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

#Calculamos las muestras - tres muestras
skfolds = StratifiedKFold(n_splits=3, random_state=42)

#Para cada muestra...
for train_index, test_index in skfolds.split(X_train, y_train_5):
	#clonamos el modelo
	clone_clf = clone(sgd_clf)

	#Obtenemos los datos de prueba
	X_train_folds = X_train[train_index]
	y_train_folds = y_train_5[train_index]
	
	#Obtenemos los datos de validación
	X_test_fold = X_train[test_index]
	y_test_fold = y_train_5[test_index]

	#Entrenamos el modelo
	clone_clf.fit(X_train_folds, y_train_folds)

	#Evaluamos el modelo
	y_pred = clone_clf.predict(X_test_fold)

	#Calculamos la métrica
	n_correct = sum(y_pred == y_test_fold)
	print(n_correct / len(y_pred)) # prints 0.9502, 0.96565 and 0.96495
```



## Confusion Matrix (Clasificación Binaria)

Para calcular la Confusion Matrix necesitamos un conjunto de predicciones junto con sus valores reales. Para obtener las predicciones usamos __cross_val_predict__:

```py
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
```

Ahora podemos calcular la Confusion Matrix:

```py
from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_train_pred)

array([[53057, 1522],[1325, 4096]])
```

### Precision

```py
from sklearn.metrics import precision_score

precision_score(y_train_5, y_train_pred) 
0.7290850836596654
```

### Recall

```py
from sklearn.metrics import recall_score

recall_score(y_train_5, y_train_pred)
0.7555801512636044
```

### F1_score

```py
from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)
0.7420962043663375
```

### Precision & recall trade-off

En ocasiones podemos estar interesados en mejorar la precision del modelo - por ejemplo, si estamos clasificando videos violentos, queremos que cuando estemos tratando un video violento lo clasifiquemos bien -, otras veces el recall del modelo - por ejemplo, si estamos clasificando videos violentos, queremos que cuando estemos tratando un video no violento lo clasifiquemos bien. Lo que no podemos es mejorar ambas métricas, mejorar una se hará a expensas de la otra.

El clasificador utiliza un threshold. Por defecto el threshold es cero. Pomdemos ver como cambian el precision y el recall con el threshold: 

```py
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```

Si queremos ver que threshold que nos dara una precision del 90%:

```py
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
```

Los casos en los que podemos decir con un 90% de seguridad que se trata de un True son:

```py
y_train_pred_90 = (y_scores >= threshold_90_precision)
```

# Optimizacion de Hiper-parametros

Con GridSearchCV vamos a poder probar un tipo de modelo usando diferentes combinaciones de hiperparámetros.

Especificaremos los parametros que queremos investigar. En este ejemplo estaríamos probando 18 combinaciones diferentes del modelo RandomForestRegressor. Adicionalmente estamos usando __cross-validation__ como se indica con el parámetro _cv_. Esto significa que en este ejemplo el modelo será entrenado 18 x 5 = 90 veces.

```py
from sklearn.model_selection import GridSearchCV

#parametros que queremos investigar
param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}, # Probaremos 3 x 4 = 12 combinaciones
			{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},] # Probaremos 2 x 3 = 6 combinaciones

#Creamos el modelo
forest_reg = RandomForestRegressor()

#Realizamos la búsqueda de parametros
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
```

Podemos ver el resultado de la búsqueda:

```py
grid_search.best_params_

{'max_features': 8, 'n_estimators': 30}
```

Podemos obtener también cual es el mejor modelo:

```py
grid_search.best_estimator_

RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features=8, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=None, oob_score=False, random_state=None, verbose=0, warm_start=False)
```

Si GridSearchCV es inicializada con el parámetro __refit=True__, que es el valor por defecto, una vez que encontramos el mejor estimador, se entrana con todo el conjunto de datos.

Podemos ver los resultados de cada una de las combinaciones que se intentaron:

```py
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
	print(np.sqrt(-mean_score), params)
```

# Regresion

## Linear Regression

Calcula la regresión lineal utilizando la formula matemática.

```py
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
```

Los resultados de este predictor seráan:

```py
lin_reg.intercept_, lin_reg.coef_
(array([4.21509616]), array([[2.77011339]]))
```

Podemos usarlos para hacer estimaciones:

```py
lin_reg.predict(X_new)
array([[4.21509616],[9.75532293]])
```

### Complejidad computacional

- El entrenamiento es __O(n^2)__ con el número de features
- El entrenamiento es __lineal__ con el número de datos
- La estimación es __lineal__

### Regularización

#### Ridge (Norma 2)

```py
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=1, solver="cholesky")

ridge_reg.fit(X, y)

ridge_reg.predict(X_new)
```

#### Lasso (Norma 1)

```py
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)

lasso_reg.fit(X, y)

lasso_reg.predict(X_new)
```

#### Elastic Net

```py
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)

elastic_net.fit(X, y)

elastic_net.predict(X_new)
```

## Stocastic Gradient Descent

Aplica gradient descent, pero en lugar de aplicarlo sobre todos los datos de entrenamiento, lo que hace el algoritmo es elegir uno al azar, y actualizar los pesos con este dato.

Comportamiento estocastico:

- __shuffle__. Indica si los datos deben barajarse en cada epoch. Por defecto es True
- __random_state__. Semilla para el generador de números aleatorios. Por defecto es None. Es un valor entero

función de error:

- __lossstr__. función de error a optimizar. Por defecto es 'squared_loss'. Otras opciones son:
	- 'squared_loss'
	- 'huber'. modifica 'squared_loss' para fozalizarse menos en los outliers. Cuando la distancia supera un determinado valor, epsilon, en lugar de usar norma 2, cuadrática, usara norma 1
	- 'epsilon_insensitive'. Ignora aquellos errores que sean menores que epsilon, y usa una norma 1 para los errores que sean mayores que epsilon
	- 'squared_epsilon_insensitive'. Ignora aquellos errores que sean menores que epsilon, y usa una norma 2 para los errores que sean mayores que epsilon
- __epsilon__. El valor de epsilon. Por defecto es 0.1
- __penalty__. Regularización a utilizar. Por defecto se usa _l2_
	- 'l2'
	- 'l1'
	- 'elasticnet'
- __alpha__. Factor a aplicar al termino de regularización, o _penalty_. Por defecto es 0.0001
- __l1_ratio__. Se usa en Elastic Net. Por defecto es 0.15.  0 <= l1_ratio <= 1. l1_ratio=0 corresponderá a l2 penalty, l1_ratio=1 a l1.
- __fit_intercept__. Indica si debe estimarse el termino constante, el bias, o no. Por defecto es True, se estima

Aprendizaje:

- __max_iter__. número máximo de iteraciones
- __learning_rate__. Define como actualizar la learning rate. Por defecto es invscaling.
	- 'constant': Constante. eta = eta0
	- 'optimal': eta = 1.0 / (alpha * (t + t0)) where t0 is chosen by a heuristic proposed by Leon Bottou.
	- 'invscaling': eta = eta0 / pow(t, power_t)
	- 'adaptive': eta = eta0, mientras el error vaya disminuyendo . Si durante _n_iter_no_change_ epochs consecutivas no se decrementa el error, o no lo hace por encima de la tolerancia, _tol_, y _early_stopping_ es True, divide la tasa actual de aprendizaje por cinco
- __eta0__. Learning rate de partida
- __tol__. Tolerancia. Por debajo de este error, el algoritmo se detiene
- __early_stopping__. Por defecto es False. Si no cambia el error durante _n_iter_no_change epochs, detiene el aprendizaje
- __n_iter_no_change__. Paciencia del algoritmo. Por defecto es 5
- __validation_fraction__. Por defecto 0.1. Proporción de los datos que se apartarán para validar el error en el caso de _early stopping_. Solo se usa si _early_stopping_ es True

```py
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
```

El resultado del entrenamiento:

```
sgd_reg.intercept_, sgd_reg.coef_

(array([4.24365286]), array([2.8250878]))
```

Otros valores interesantes:

```py

sgd_reg.n_iter_ #Numero de epochs que se ejecutaron

sgd_reg.t_ #Numero de veces que se actualizaron los pesos
```

### Complejidad computacional

El coste de tener que entrenar más features se incrementa de forma lineal, de modo que cuando tenemos muchas features el SGD es un método más óptimo que la regresión lineal.

El SGD es __muy sensible a la escala de los datos__, y por lo tanto se requiere siempre una normalización de los mismos.

El coste de adiestrar el SGD es lineal con el número de datos de entrenamiento

### Regularización

La regularización de los pesos es uno de los hiper-parámetros del SGD - _penalty_. Raro será el caso en el que no tengamos que regularizar los pesos, por eso se incluye como hiper-parametro del modelo - en lugar de en un modelo a parte.

## regresión Polinomial

Cuando los datos no tienen una estructura lineal, sino más bien polinomial, ¿qué podemos hacer?. Un truco que podemos aplicar para que los métodos anteriores sigan siendo válidos, es calcular series que contenga las potencias de las features, de modo que el problema pueda ser tratado como un caso lineal.

En este ejemplo estamos solicitando que se añadan los terminos de grado 2. Podemos observar que en el dataset transformado ya tenemos una feature que es básicamente el cuadrado de la feature original:

```py
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

X[0]
array([-0.75275929])

X_poly[0]
array([-0.75275929, 0.56664654])
```

### Ejemplo

Veamos un ejemplo en el que se define un pipeline en el que se hace una regresión polinomial:

```py
from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline([("poly_features", PolynomialFeatures(degree=10, include_bias=False)),("lin_reg", LinearRegression()),])

polynomial_regression.fit(X, y)

polynomial_regression.predict(X)
```

# Clasificadores

## Logistic Regression (binario y multi-class)

Usamos este método para clasificar. Por un lado el modelo que vamos a estimar aplica una función de activación, la __Logistic function__ _( 1/(1+e^-x)_ al resultado del modelo lineal.

La función de coste es la __log loss__ _(-y*log(p)-(1-y)*log(1-p))_.

El modelo admite varios parámetros:

- __solver__. Algoritmo de optimización, Puede ser 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'. Por defecto 'lbfgs'. 
	- Si el dataset es pequeño, 'liblinear' es una buena elección. 'sag' y 'saga' son más rápidos en datasets grandes.
	- Para problemas multiclass solo 'newton-cg', 'sag', 'saga' y 'lbfgs' pueden suportar multinomial loss; 'liblinear' solo sirve para OvA.
	- 'newton-cg', 'lbfgs', 'sag' y 'saga' solo pueden usarse con _penalty_ l2 y None
	- 'liblinear' and 'saga' pueden usar _penalty_ l1
	- 'saga' tambien soporta _penalty_ 'elasticnet'
	- 'liblinear' no soporta _penalty_ 'none'
	- 'sag' y 'saga' convergirán rápidamente solo si las features tienen la misma escala. Habrá que pre-procesar los datos con sklearn.preprocessing
- __penalty__. Regularización a aplicar a los pesos. Las posibilidades son {'l1', 'l2', 'elasticnet', 'none'}. El valor por defecto es 'l2'. 
	- Con el 'newton-cg', 'sag' y 'lbfgs' solo es posible elegir 'l2'. 
	- 'elasticnet' solo se puede usar con 'saga'
	- 'none' no se admite con 'liblinear'
- __l1_ratio__. Por defecto None. Utilizado con Elastic-Net. Toma un valor entre 0 <= l1_ratio <= 1. Si l1_ratio=0 equivale a utilizar 'l2', mientras que l1_ratio=1 equivale a 'l1'
- __max_iter__. Máximo número de interacciones. El valor por defecto es 100
- __tol__. Por defecto el valor es 1e-4. Tolerancia usada para parar el adiestramiento
- __C__. Por defecto 1.0. Es la inversa de la intensidad de la regularización. Es un concepto similar al usado en vector machines, valores pequeños implican una intensidad en la regularización mayor
- __fit_intercept__. Por defecto es True. Determina si debemos incluir el bias 
- __random_state__. Por defecto None. Se usa cuando el algoritmo es 'sag', 'saga' o 'liblinear', para que se barajen los datos
- __multi_class__. Por defecto 'auto'
	- 'ovr'. Se crea un modelo binario por cada etiqueta. 
	- 'multinomial'. Usamos softmax para determinar la probabilidad en cada una de las clases. 'multinomial' no esta disponible cuando el algoritmo es 'liblinear'. 'auto' selecciona 'ovr' si los datos son binarios, o el algoritmo es 'liblinear'. En caso contratrio se usara 'multinomial'.

### Clasificación Binaria

Notese que el método es __predict_proba__. Usamos los valores por defecto

```py
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

log_reg.fit(X, y)

log_reg.predict_proba(X)
```

### Multi-class (softmax)

Notese que el método es __predict_proba__. Indicamos que se trata de un caso multi-class, y al elegir _multinomial_ estamos usando softmax:

```py
from sklearn.linear_model import LogisticRegression

softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)

log_reg.fit(X, y)

log_reg.predict_proba(X)
array([[6.38014896e-07, 5.74929995e-02, 9.42506362e-01]])
```

## Stocastic Gradient Descent (binario y multi-class)

### Binario

```py
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)

sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([some_digit])
array([ True])
```

### Multi-class

Cuando y_train tiene más de una clases, el SGDClassifier automáticamente trabaja en modo multi-class, creando un clasificador para cada una de las clases, y retornando aquella clase en la que el threshold resulto ser más alto. Esto es lo que se denomina Uno contra Todo, o OvR (One vs Rest).

En este ejemplo lo podemos ver. El vector de training en lugar de tener True, False, tiene varias posibles valores, clases, y al hacer la predicción retorna un valor concreto, 5 en el ejemplo:

```py
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)

sgd_clf.fit(X_train, y_train) # y_train, not y_train_5

sgd_clf.predict([some_digit])
array([5], dtype=uint8)
```

Podemos usar __decision_function__ para ver el score que se dio a cada uno de las clases, y podemos comprobar que efectivamente tenemos 10 scores - en nuestro ejemplo había 10 clases, y por lo tanto se crearon 10 modelos "under the hood" -, y el score que corresponde a la clase 5 es el más alto:

```py
some_digit_scores = sgd_clf.decision_function([some_digit])

some_digit_scores

array([[-15955.22627845, -38080.96296175, -13326.66694897,
573.52692379, -17680.6846644 , 2412.53175101,
-25526.86498156, -12290.15704709, -7946.05205023,
-10631.35888549]])
```

#### OneVsOneClassifier

Podemos forzar a que se utilice la estratégia OvO con el siguiente modelo. Con esta estrategia se crea un modelo para cada pareja. Si tenemos N clases, N * (N-1) / 2 modelos. Si tuvieramos 10 clases, el total de modelo serán 45:

```py
from sklearn.multiclass import OneVsOneClassifier

ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))

fit(X_train, y_train)

predict([some_digit])
array[5], dtype=uint8)

len(ovo_clf.estimators_)
45
```

Si bien el número de modelos se incrementa, cada modelo se adiestra con un subconjunto del total.

#### OneVsRestClassifier

Podemos forzar a que se utilice la estratégia OvR con el siguiente modelo. Con esta estrategia se crea un modelo para cada clase. Si tenemos N clases, N modelos. Si tuvieramos 10 clases, el total de modelo serán 10:

```py
from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SGDClassifier(random_state=42))

ovr_clf.fit(X_train, y_train)

ovr_clf.predict([some_digit])
array([5], dtype=uint8)

len(ovr_clf.estimators_)
10
```






