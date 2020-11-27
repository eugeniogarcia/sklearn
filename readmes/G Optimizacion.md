# Optimizacion de Hiper-parametros

## GridSearchCV y RandomizedSearchCV

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

Podemos ver el resultado de la búsqueda, cuales son los mejores parámetros:

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

### Pipelines

Si estamos usando una pipeline, el diccionario de valores a usar en el universo de pruebas seguirá una sintaxis especial:

```py
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ("pepe", KMeans(n_clusters=50, random_state=42)),
    ("log_reg", LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)),
])

param_grid = dict(pepe__n_clusters=range(80, 100))

grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_clf.fit(X_train, y_train)

grid_clf.best_params_

grid_clf.score(X_test, y_test)
```

Notese como el parámetro *n_clusters* en el diccionario lo llamamos *pepe__n_clusters*. Esto es así porque la entrada en el pipeline donde hemos definido el kmeans se llama *pepe*.

## Optimizando modelos Keras

Podemos usar RandomizedSearchCV o GridSearchCV para optimizar los hiper-parámetros de un modelo Keras. Para ellos tendremos que:
- Definir el modelo por medio de una función, que tenga como argumentos los hiper-parámetros que queremos optimizar
- Crear un wrapper sklearn, de modo que nuestro modelo pase a ser un predictor más
- Aplicar RandomizedSearchCV o GridSearchCV para optimizar los parámetros

Veamoslo con un ejemplo. En primer lugar definimos la función que crea el modelo:

```py
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    #Creamos un modelo secuencial
	model = keras.models.Sequential()
	#Con un imput de tamaño input_shape
	model.add(keras.layers.InputLayer(input_shape=input_shape))
	#Con n_hidden layers
    for layer in range(n_hidden):
		#Cada capa oculta tiene n_neurons
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    #Finalmente el modelo tiene una capa final para hacer la regresión a un número
	model.add(keras.layers.Dense(1))
	#Usamos SGD con una tasa de aprendizaje learning_rate 
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model
```

A continuación creamos un wrapper para el modelo:

```py
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
```

Podemos hacer el entrenamiento como con cualquier otro modelo Keras:

```py
keras_reg.fit(X_train, y_train, epochs=100,
              validation_data=(X_valid, y_valid),
              callbacks=[keras.callbacks.EarlyStopping(patience=10)])
```

Podemos ver el scoring - en sklearn sería el mse, no la accuracy:

```py
mse_test = keras_reg.score(X_test, y_test)
```

Podemos hacer predicciónes:

```py
y_pred = keras_reg.predict(X_new)
```

Vamos a utilizar RandomizedSearchCV para buscar el número de capas ocultas, neuronas y learning rate más convenientes. Usamos cross_validation para evaluar los modelos, con 3 grupos. El número máximo de interacciones será 10:

```py
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

#Universo que vamos a explorar. Tenemos 4 x 2 x 2 = 16 convinaciones. 
#Como cv=3, significa que haremos 16 x 3 = 48 adiestramientos
param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2),
}

#Lanzamos la búsqueda
rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3, verbose=2)
rnd_search_cv.fit(X_train, y_train, epochs=100,
                  validation_data=(X_valid, y_valid),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])

#Parámetros más óptimos
rnd_search_cv.best_params_

#Mejor score que hemos conseguido
rnd_search_cv.best_score_

#Mejor modelo
rnd_search_cv.best_estimator_

model = rnd_search_cv.best_estimator_.model
model.evaluate(X_test, y_test)
```