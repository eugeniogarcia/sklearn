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