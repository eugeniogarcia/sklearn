# Calidad del modelo

Podemos evaluar la calidad del modelo usando diferentes métricas:

- mean_squared_error
- mean_absolute_error
- cross_val_score
- Confusion Matrix (Clasificación Binaria)

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

NOTA: Con _cross_validation_score_ tenemos un hibrido. Por un lado lo que estamos haciendo es adiestrar un modelo, y por otro lado, una vez adiestrado recuperamos su score. El asunto es que el modelo no se adiestra de cualquier forma, se hacen K adiestramientos __independientes__, como podemos ver en la implementación custom que veremos abajo, cuando se usa _clone_, y obtenemos la calidad _conjunta_ de los K modelos.

Dividimos el dataset en K bloques. Usamos K-1 para adiestrar el modelo, y el último bloque para evaluarlo. Hacemos esta operación K veces, eligiendo en cada ocasión un bloque diferente para la evaluación. La precisión del modelo será la media de las K precisiones evaluadas.

Este método requiere que pasemos como argumento el modelo que debe ser entrenado, así como la serie de datos.

Podemos usar diferentes KPI para medir. En este ejemplo usamos el _error cuadrático medio_. Los modelos de scoring disponibles dependen del modelo elegido:

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

Puntos a destacar:
- Repetimos el training y la evaluación K veces
- En cada interacción se parte de un modelo "limpio". Para ello usamos un clon del modelo en cada interacción
- El scoring puede ser cualquiera. En este caso usamos el desvio 

## accuracy_score

Precisión del modelo:

```py
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression()

log_clf.fit(X_train, y_train)
y_pred = log_clf.predict(X_test)
accuracy_score(y_test, y_pred))
```

## Confusion Matrix (Clasificación Binaria)

Para calcular la Confusion Matrix necesitamos un conjunto de predicciones junto con sus valores reales. Para obtener las predicciones usamos __cross_val_predict__ - como podríamos haber usado cualquier otro clasificador:

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

En la matriz tendremos

```
[[TN FP],[FN, TP]]
```

Cada columna evalua una categóría. Por ejemplo, la primera esta evaluando los _Nagatives_. Con esto podemos definir las siguientes métricas:

```
Precision = TP / ( TP + FP)
```

```
recall = TP / (TP + FN)
```

```
F1 = 1 / ((1 / precision) + (1 / recall))
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

El clasificador utiliza un threshold por defecto que no podemos manipular. Este threshold determina cual es el balance de precision vs recall que estamos usando. Por defecto el threshold es cero. 

```py
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```

Si bien no podemos cambiar el threshold directamente, podemos usar la _decision_function_ para aplicar el threshold que nos interese. Aquí por ejemplo vemos como hacerlo:

```py
y_scores = sgd_clf.decision_function([some_digit])

y_scores
array([2412.53175101])

threshold = 0
y_some_digit_pred = (y_scores > threshold)

array([ True])
```

Pomdemos ver como cambian el precision y el recall con el threshold. Si queremos ver que threshold que nos dara una precision del 90%:

```py
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
```

Los casos en los que podemos decir con un 90% de seguridad que se trata de un True son:

```py
y_train_pred_90 = (y_scores >= threshold_90_precision)
```
