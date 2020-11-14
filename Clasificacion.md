# Clasificadores

## Logistic Regression (binario y multi-class)

Usamos este método para clasificar. Por un lado el modelo que vamos a estimar aplica una función de activación, la __Logistic function__ _( 1/(1+e^-x)_ al resultado del modelo lineal.

El algoritmo se puede usar en problemas de clasificación binaría y en multi-class. En caso de que la clasificación sea binaria, la función de coste es la __log loss__ _(-y*log(p)-(1-y)*log(1-p))_.

En problemas de clasificación multi-clases, se podrán configurar dos estrategías diferentes, según lo que se informe en el parámetro _multi_class_:
- 'ovr'. Se utiliza la estrategia __OvR__, "One vs Rest"
- 'multinomial'. Se utiliza como función de coste la __cross-entropia__. 'multinomial' solo se soporta con los algoritmos 'lbfgs', 'sag', 'saga' y 'newton-cg'

No hay una solución matemática que resuelva este problema de optimización, así que se utiliza Batch Gradient Descent para resolverlo. El modelo admite varios parámetros:

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

Los algoritmos 'newton-cg', 'sag', y 'lbfgs' admiten solo la regularización 'l2' - o trabajar sin regularización. El algoritmo 'liblinear' admite tanto 'l1' como 'l2'.Elastic-Net solo se admite con el algoritmo 'saga'.

### Clasificación Binaria

Notese que el método es __predict_proba__. Usamos los valores por defecto

```py
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

log_reg.fit(X, y)

log_reg.predict_proba(X)

#Clases disponibles
log_reg.classes_
```

### Multi-class (softmax)

Notese que el método es __predict_proba__. Indicamos que se trata de un caso multi-class, y al elegir _multinomial_ estamos usando softmax:

```py
from sklearn.linear_model import LogisticRegression

softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)

log_reg.fit(X, y)

log_reg.predict_proba(X)
array([[6.38014896e-07, 5.74929995e-02, 9.42506362e-01]])

#Clases disponibles
log_reg.classes_
```

### Multi-class (ovr)

Notese que el método es __predict_proba__. Como _y_ tiene varias clases, y no especificamos nada en _multinomial_, aplicará el valor por defecto, 'ovr':

```py
from sklearn.linear_model import LogisticRegression

softmax_reg = LogisticRegression(C=10)

log_reg.fit(X, y)

log_reg.predict_proba(X)
array([[6.38014896e-07, 5.74929995e-02, 9.42506362e-01]])

#Clases disponibles
log_reg.classes_
```

## Stocastic Gradient Descent (binario y multi-class)

Por defecto utiliza una función de coste __hinge__, _y = max(0, 1-t * y)_, para definir el clasificador.

Se configura con los siguientes parámetros:

- Comportamiento estocastico:
	- __shuffle__. Indica si los datos deben barajarse en cada epoch. Por defecto es True
	- __random_state__. Semilla para el generador de números aleatorios. Por defecto es None. Es un valor entero

- Función de error:
	- __loss__. función de error a optimizar. Por defecto es 'hinge'. Otras opciones son:
		- 'hinge'
		- 'log'. Usa Logistic regresion
		- 'modified_huber'. Ofrece una función de coste probabilistica al tiempo que es robusta frente a outliers
		- 'squared_hinge'. Como hinge, pero la penalización es cuadrática
		- 'perceptron'
		- A regression loss. Estas funciones estan diseñadas para hacer regresión, pero pueden ser útiles en algunos casos de clasificación:
			- 'squared_loss'
			- 'huber'. Es una función cuadratica para errores menores a un umbral, y lineal para errores mayores
			- 'epsilon_insensitive'
			- 'squared_epsilon_insensitive'
	- __epsilon__. El valor de epsilon. Por defecto es 0.1. Se utiliza en 'huber', 'epsilon_insensitive', y 'squared_epsilon_insensitive'. En el caso de 'huber', determina el umbral a partir del cual no es tan importante que la predicción sea correcta. Para 'epsilon-insensitive', cualquier diferencia entre la predicción actual y la correcta - según la etiqueta -, es ignorada si es inferior a este threshold.
	- __penalty__. Regularización a utilizar. Por defecto se usa _l2_
		- 'l2'
		- 'l1'
		- 'elasticnet'
	- __alpha__. Factor a aplicar al termino de regularización, o _penalty_. Por defecto es 0.0001
	- __l1_ratio__. Se usa en Elastic Net. Por defecto es 0.15.  0 <= l1_ratio <= 1. l1_ratio=0 corresponderá a l2 penalty, l1_ratio=1 a l1.
	- __fit_intercept__. Indica si debe estimarse el termino constante, el bias, o no. Por defecto es True, se estima
- Aprendizaje:
	- __max_iter__. número máximo de iteraciones
	- __learning_rate__. Define como actualizar la learning rate. Por defecto es invscaling.
		- 'constant': Constante. eta = eta0
		- 'optimal': eta = 1.0 / (alpha * (t + t0)) where t0 is chosen by a heuristic proposed by Leon Bottou.
		- 'invscaling': eta = eta0 / pow(t, power_t)
		- 'adaptive': eta = eta0, mientras el error vaya disminuyendo . Si durante _n_iter_no_change_ epochs consecutivas no se decrementa el error, o no lo hace por encima de la tolerancia, _tol_, y _early_stopping_ es True, divide la tasa actual de aprendizaje por cinco
	- __eta0__. Learning rate de partida
	- __power_tdouble__. Por defecto 0.5. Exponente usado en inverse scaling.
	- __tol__. Tolerancia. Por debajo de este error, el algoritmo se detiene
	- __early_stopping__. Por defecto es False. Si no cambia el error durante _n_iter_no_change epochs, detiene el aprendizaje
	- __n_iter_no_change__. Paciencia del algoritmo. Por defecto es 5
	- __validation_fraction__. Por defecto 0.1. Proporción de los datos que se apartarán para validar el error en el caso de _early stopping_. Solo se usa si _early_stopping_ es True


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

## Support Vector Machine (SVM)

La función de coste en SVM no trata de ajustar los resultados del modelo a las observaciones, sino buscar la recta 


```py
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()

X = iris["data"][:, (2, 3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64) # Iris-Virginica

svm_clf = Pipeline([("scaler", StandardScaler()),("linear_svc", LinearSVC(C=1, loss="hinge")),])

svm_clf.fit(X, y)
```




