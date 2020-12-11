# 1 Regresion

## 1.1 Linear Regression

Calcula la regresión lineal utilizando la formula matemática.

- __fit_intercept__. Por defecto True. Determina si se calculara el término independiente o no

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

### 1.1.1 Complejidad computacional

- El entrenamiento es __O(n^2)__ con el número de features
- El entrenamiento es __lineal__ con el número de datos
- La estimación es __lineal__

### 1.1.2 Regularización

Para evitar el overfit una técnica habitual es limitar la norma de los pesos, tratar que los pesos sean lo más pequeños posible. 

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

Es un hibrido entre L1 y L2. Con el parametro _l1_ratio_ controlamos cuanto tiene de L1 y cuanto de L2. _alpha_ es el factor que se le aplica al L1 o L2 de los pesos.

```py
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)

elastic_net.fit(X, y)

elastic_net.predict(X_new)
```

## 1.2 Stocastic Gradient Descent

Aplica gradient descent, pero en lugar de aplicarlo sobre todos los datos de entrenamiento, lo que hace el algoritmo es elegir uno al azar, y actualizar los pesos con este dato.

Se configura con los siguientes parámetros:

- Comportamiento estocastico:
	- __shuffle__. Indica si los datos deben barajarse en cada epoch. Por defecto es True
	- __random_state__. Semilla para el generador de números aleatorios. Por defecto es None. Es un valor entero
- Función de error:
	- __loss__. función de error a optimizar. Por defecto es 'squared_loss'. Otras opciones son:
		- 'squared_loss'
		- 'huber'. modifica 'squared_loss' para __focalizarse menos en los outliers__. Cuando la distancia supera un determinado valor, epsilon, en lugar de usar norma 2, cuadrática, usara norma 1
		- 'epsilon_insensitive'. Ignora aquellos errores que sean menores que epsilon, y usa una norma 1 para los errores que sean mayores que epsilon
		- 'squared_epsilon_insensitive'. Ignora aquellos errores que sean menores que epsilon, y usa una norma 2 para los errores que sean mayores que epsilon
	- __epsilon__. El valor de epsilon. Por defecto es 0.1. Se utiliza en ciertas funciones de error, como __huber__.
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

### 1.2.1 Complejidad computacional

El coste de tener que entrenar más features se incrementa de forma lineal, de modo que cuando tenemos muchas features el SGD es un método más óptimo que la regresión lineal.

El SGD es __muy sensible a la escala de los datos__, y por lo tanto se requiere siempre una normalización de los mismos.

El coste de adiestrar el SGD es lineal con el número de datos de entrenamiento

### 1.2.2 Regularización

La regularización de los pesos es uno de los hiper-parámetros del SGD - _penalty_. Raro será el caso en el que no tengamos que regularizar los pesos, por eso se incluye como hiper-parametro del modelo - en lugar de en un modelo a parte.

## 1.3 regresión Polinomial

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

### 1.3.1 Ejemplo

Veamos un ejemplo en el que se define un pipeline en el que se hace una regresión polinomial:

```py
from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline([("poly_features", PolynomialFeatures(degree=10, include_bias=False)),("lin_reg", LinearRegression()),])

polynomial_regression.fit(X, y)

polynomial_regression.predict(X)
```

## 1.4 Support Vector Machines

__Los SVR se utilizan fundamentalmente para clasificar__, pero también podemos usarlos para resolver problemas de regresión. La idea aquí es la inversa a la del problema de clasificación, esto es, buscar que todos los puntos entren dentro de el margen de tolerancia, de nuestra "calle", __y que este margen de tolerancia sea lo más pequeño posible__.

```py
from sklearn.svm import LinearSVR

svm_reg = LinearSVR(epsilon=1.5)

svm_reg.fit(X, y)
```

### 1.4.1 Problemas no lineales

De la misma forma que en el caso de clasificación podíamos usar SV__C__, en el de regresión podremos usar SV__R__ para tratar problemas que no sean lineales:

```py
from sklearn.svm import SVR
svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)

svm_poly_reg.fit(X, y)
```