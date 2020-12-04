# Preparación de datos

Vamos a ver una serie de técnicas:

- Estratificar datos
- Rellenar huecos
- Tratar datos categóricos
- Normalizar los datos

Y finalmente veremos como poder crear _pipelines_ que convinen varios de estos pasos y los apliquen como un todo.
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

Esta transformación se encarga de rellenar datos vacios en series categoricas. Lo que hacemos es rellenar aquellos campos vacios con la categoría que sea más frecuente.

```py
# Inspired from stackoverflow.com/questions/25239958
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    #Tiene como argumento un dataframe.
    def fit(self, X, y=None):
        #Calcula una serie
        #La serie tiene como indice los nombres de las columnas del dataframe
        #y como valor una colección con el nombre de la categoría más popular, la que tiene el contador más elevado 
        #tenemos la categoria que es más frecuente
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],index=X.columns)
        return self
		
    def transform(self, X, y=None):
        #Transformamos el dataframe, de modo que en cada columna rellenamos los valores no informados, 
        #con la categoría más frecuente
        return X.fillna(self.most_frequent_)
```

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