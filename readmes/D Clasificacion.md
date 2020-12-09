# 1 Clasificadores

## 1.1 Multi-Class Estrategias

Cuando se trata hacer una clasificación multi-class con SGD o con SVM, podemos forzar a que se siga una estratágia OvO o OvR utilizando estos wrappers:
- OneVsOneClassifier
- OneVsRestClassifier

### 1.1.1 OneVsOneClassifier

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

### 1.1.2 OneVsRestClassifier

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

## 1.2 Logistic Regression (binario y multi-class)

Usamos este método para clasificar. Por un lado el modelo que vamos a estimar aplica una función de activación, la __Logistic function__ _( 1/(1+e^-x)_ al resultado del modelo lineal. El algoritmo de optimización que se usa es batch gradient descent.

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

### 1.2.1 Clasificación Binaria

Notese que el método es __predict_proba__. Usamos los valores por defecto

```py
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

log_reg.fit(X, y)

log_reg.predict_proba(X)

#Clases disponibles
log_reg.classes_
```

### 1.2.2 Multi-class (softmax)

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

### 1.2.3 Multi-class (ovr)

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

## 1.3 Stocastic Gradient Descent (SGD) (binario y multi-class)

En estos métodos usaremos lo que pretendemos es, como el logistic regression, determinar la probilidad de que una instancia perteneza a una determinada clase - binaria - o clases. Para determinar esta probabilidad usamos una función de coste y la optimización la hacemos usando SGD, en lugar del Batch Gradient Descent que empleamos en la Logistic Regression. La función de coste no está basada en la entropía - de la información; Claude Shannon. Por defecto utiliza una función de coste __[hinge](https://en.wikipedia.org/wiki/Hinge_loss)__, para definir el clasificador. Hinge se define como:

```
Si cada instancia puede clasificarse como +1 o -1, y denominamos _t_ como se clasifica la instancia, e _y_ la predicción del clasificador, la funcion de coste hinge se define como:

y = max(0, 1 - t * y)
```

Otra función de coste que se usa en ocasiones es __[huber](https://en.wikipedia.org/wiki/Huber_loss)__. Huber usa un hibrido de L2 y L1 para medir los errores. La idea es que cuando los errores sean pequeños se use L2, y cuando sea grandes L1. Un parametros determina el umbral a partir del cual se usa L2 o L1:

```
loss = 1/2 * err^2  si |err| < epsilon

loss = epsilon * (|err|- 1/2 * epsilon)  si err >= epsilon
```

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
	- __epsilon__. El valor de epsilon. Por defecto es 0.1. Se utiliza en __'huber'__, 'epsilon_insensitive', y 'squared_epsilon_insensitive'. En el caso de 'huber', determina el umbral a partir del cual no es tan importante que la predicción sea correcta. Para 'epsilon-insensitive', cualquier diferencia entre la predicción actual y la correcta - según la etiqueta -, es ignorada si es inferior a este threshold.
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


### 1.3.1 Binario

```py
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)

sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([some_digit])
array([ True])
```

### 1.3.2 Multi-class

Cuando y_train tiene más de una clases, el SGDClassifier __automáticamente trabaja en modo multi-class__, creando un clasificador para cada una de las clases, y retornando aquella clase en la que el threshold resulto ser más alto. __Esto es lo que se denomina Uno contra Todo, o OvR (One vs Rest)__.

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

## 1.4 Support Vector Machine (SVM) (binario y multi-class)

Con esta técnica el objetivo es encontrar un modelo que establece una separación más drastica entre las instancias a clasificar, que es lo más ortogonal posible - el vector que determina el gradiente de cambio entre los dos colectivos, sera el que defina la superficie de separación entre los colectivos. Con estos modelos se introduce también el concepto de ancho en la separación. Cuanto mayor es el ancho mejor es la capacidad para generalizar del modelo, pero más dificil será encontrar una solución, y quizás tengamos que admitir algunos errores en la clasificación. El parametro que se usa para balancear la capacidad de generalizar con el rigor de la clasificación es comumente referido como __C__. Cuanto más pequeño sea _C_ más genérico es el modelo.

El modelo se puede entrenar usando varias funciones de coste, entre ellas la _hinge_ que hemos visto antes, y una función lineal.

Este modelo no se basa en la estimación de probabilidades sino que busca la separación geométrica, con rectas, curvas, superficies,... que separe los diferentes grupos de datos, con la mayor separación posible entre ellos.

### 1.4.1 Datasets separables de forma lineal (LinearSVC)

El modelo se puede configurar con los siguientes parámetros:

- __loss__. función de error a optimizar. Por defecto es 'squared_hinge'. Otras opciones son:
	- 'hinge'. _y = max(0, 1 - t * y)_
	- 'squared_hinge'. Cuadrado de la función 'hinge'
- __penalty__. Regularización a utilizar. Por defecto se usa _l2_
	- 'l2'
	- 'l1'
	- 'elasticnet'
- __tol__. Tolerancia. Por debajo de este error, el algoritmo se detiene
- __C__. Por defecto 1.0. Parametro de regularización. Cuanto más pequeño es, más tolerantes somos a _margin violations_. Cuanto más pequeño es, más ancha es la banda. Tener una banda ancha indica que el modelo generalizara mejor, que si añadimos algun dato más, la probabilidad de que lo clasifiquemos bien es mayor. Cuanto menor sea la banda, "más apelotonados" estarán los nuevos datos, y será más dificil separar unos de otros.
- __multi_class__. Determina que estratégia seguir para casos multi-class. Por defecto 'ovr':
	- 'ovr'.
	- 'crammer_singer'. Raramente se usa
- __fit_intercept__. Indica si debe estimarse el termino constante, el bias, o no. Por defecto es True, se estima
- __random_state__. Semilla para el generador de números aleatorios. Por defecto es None. Es un valor entero. Solo se utiliza cuando dual es True. En caso de que dual sea False, la implementación del algoritmo es determinista
- __max_iter__. número máximo de iteraciones
- __dual__. Por defecto True. El algoritmo liblinear, que es el que se utiliza con este modelo, es _dual-based_. Cuando n_samples > n_features es preferible False. True es preferible cuando:
	- Datasets grandes y "sparse" (e.g., documents) con un C no muy grande
	- n_samples << n_features

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

#### Nota

Podemos usa otros modelos como alternativa a LinearSVC:
- SVC. Es más lento que LinearSVC, por esto no se recomienda
	```py
	SVC(kernel="linear", C=1)
	```
- SGDClassifier. Aplica Stochastic Gradient Descent. No converge tan rápido como LinearSVC, pero cuando el tamaño del dataset es muy grande y no cabe en la memoría, puede ser util. También nos servirá para casos en los que el entrenamiento tenga que hacerse online 
	```py
	SGDClassifier(loss="hinge",alpha=1/(m*C))
	```
	
### 1.4.2 Datasets NO separables de forma lineal 

#### 1.4.2.1 Usar Polynomial Regresion + LinearSVC

Usamos la misma táctica que ya vimos para reutilizar la regresión lineal en problemas polinómicos. Lo que haremos es crear features que representan las potencias de las features disponibles, de modo que calculemos un modelo lineal que use estas features sintéticas. Cuando Mayor sea el grado que elijamos mayor será el número de features que se calcularan. Esto hace que el método resulte prohibitivo cuando la potencia sea alta.

Veamos con un pipeline de ejemplo como sería. En esencia no cambia nada con respecto al caso de clasificación lineal.

```py
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

polynomial_svm_clf = Pipeline([("poly_features", PolynomialFeatures(degree=3)),("scaler", StandardScaler()),("svm_clf", LinearSVC(C=10, loss="hinge"))])

polynomial_svm_clf.fit(X, y)
```

#### 1.4.2.2 Usar Polynomial Kernel (SVC)

El método anterior deja de ser práctico a medida que vamos subiendo el grado de la potencia. El algoritmo que usamos con este modelo esta basado en _libsvm_. La complejidad del entrenamiento es _O(n*2)_ con el número de muestras. Por este motivo para datasets grandes es aconsejable usar el _LinearSVC_ o el _SGDClassifier_.

Los Kernels son una capacidad que tenemos disponible en el modelo __SVC__. El [Kernel](https://scikit-learn.org/stable/modules/svm.html#svm-kernels) es una función que nos permite definir la distancia entre dos puntos. __Aplicaremos un kernel sobre los datos, y los datos que resultan, una vez el kernel se ha aplicado sobre ellos, serán separables linealmente__. Seran estos datos transformados los que clasifiquemos. Esta es la idea.

__SVC__ sigue una estratégia __OvO__ cuando el problema de clasificación sea __multi-class__.

Los parámetros con los que podemos configurar __SVC__ son:

- __C__. Por defecto 1.0. Es la inversa de la intensidad de la regularización. Es un concepto similar al usado en vector machines, valores pequeños implican una intensidad en la regularización mayor
- __kernel__. Por defecto 'rbf'. Puede tomar los siguientes [valores](https://scikit-learn.org/stable/modules/svm.html#svm-kernels):
	- 'linear'
	- 'poly'
	- 'rbf'
	- 'sigmoid'
	- 'precomputed'
- __degree__. Por defecto 3. Grado de la función polinomial en el caso de que el kernel sea 'poly'
- __gamma__. Por defecto 'scale'. Es el coeficiente del kernel. Puede valer:
	- 'scale'. Usa _1 / (n_features * X.var())_ como valor para gamma 
	- 'auto'. Usa _1 / n_features_
- __coef__. Por defecto 0.0. Es el término independiente en la función del kernel. Solo es relevante cuando el kernel es 'poly' o 'sigmoid'
- __tol__. Por defecto el valor es 1e-4. Tolerancia usada para parar el adiestramiento
- __max_iter__. Máximo número de interacciones. El valor por defecto es 100
- __decision_function_shape__. Por defecto 'ovr'. Puede valer 'ovo' o 'ovr'. Este parámetro se ignora en casos de clasificación binaria
- __random_state__. Por defecto None. Se usa cuando el algoritmo es 'sag', 'saga' o 'liblinear', para que se barajen los datos
- __probability__. Por defecto False. Cuando True, SVC usa cross-validation para estimar las probabilidades de cada clase, de modo que el predictor tendrá también el método _predict_proba_ disponible. Por defecto SVC nos indicara un booleano - clasificación binaria - o la clase - multi-class -, no las probabilidades

Veamos un ejemplo:

```py
from sklearn.svm import SVC

poly_kernel_svm_clf = Pipeline([("scaler", StandardScaler()),("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))])

poly_kernel_svm_clf.fit(X, y)
```

## 1.5 Ensemble Learning

Podemos __usar varios clasificadores y combinar__ sus resultados en un ensemble. El resultado del ensemble será tanto mejor como más independientes entre si sean los clasificadores que usamos. Podemos fomentar la independencia entre los clasificadores de dos maneras:

- Usando __diferentes tipos__ de clasificadores. De esta forma será dificil que compartan datos intermedios
- Usando __diferentes juegos de datos__ para entrenar cada clasificador

La forma en que combinamos los resultados de los clasificadores intermedios pude hacerse de dos formas:

- __hard__. Tomamos como resultado de la clasificación la del voto mayoritarío
- __soft__. En aquellos casos en los que los clasificadores intermedios retornen una probabilidad - es decir, los predictores tengan un método _predict_proba_ -, podemos usar la probabilidad estimada en cada clasificador para calcular la probabilidad del conjunto

### 1.5.1 Diversidad usando diferentes modelos

Veamos un ejemplo. En este ejemplo usamos __hard__ voting, es decir la mayoría gana:

```py
#Ensemble
from sklearn.ensemble import VotingClassifier
#Clasificadores individuales
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#Crea cada uno de los clasificadores
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

#Crea el ensemble. El algoritmo de votación es hard. Se vota por mayoría
voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],voting='hard')

#Entrena
voting_clf.fit(X_train, y_train)
```

Podemos ver como la precisión de cada clasificador es menor que la del ensemble:

```py
from sklearn.metrics import accuracy_score

#Hacemos el entrenamiento de cada clasificador, y luego del ensemble
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


LogisticRegression 0.864
RandomForestClassifier 0.896
SVC 0.888
VotingClassifier 0.904
```

Si queremos usar __soft__ voting, que se calcule la probabilidad promedio, se necesitan dos cosas:
- Que todos los clasificadores incluyan el metodo *predict_proba()*
- Configurar _soft_ en lugar de _hard_ en el ensemble

```py
log_clf = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(gamma="scale", probability=True, random_state=42)

voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],voting='soft')
voting_clf.fit(X_train, y_train)
```

### 1.5.2 Diversidad con el mismo modelo

Para conseguir diversidad usando el mismo tipo de clasificador, lo que tendremos que hacer es entrenar cada clasificador con datos diferentes. De esta forma conseguimos la "independencia" que hará que el ensemble se comporte mejor.

Los datos se eligen al azar. Podemos controlar __si el mismo dato se usa en varios clasificadores__, **_bootstrap=True_**, o si un dato solo se utiliza en un controlador _bootstrap=False_.

En este ejemplo estamos diciendo que se usen 500 clasificadores, y que cada clasificador se entrene con un máximo de 100 datos - elegidos al azar:

```py
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,max_samples=100, bootstrap=True, n_jobs=-1)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
```

_Bootstrap=True_ produce generalmente mejores modelos, al incrementarse la diversidad entre los clasificadores.
#### 1.5.2.1 oob_score_

Cuando _Bootstrap=True_ no todos los datos en el dataset van a ser utilizados para adiestrar los clasificadores del ensemble. Podemos usar aquellos que no se hayan empleado para validar el modelo especificando _oob_score=True_:

```py
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,bootstrap=True, n_jobs=-1, oob_score=True)

bag_clf.fit(X_train, y_train)

bag_clf.oob_score_
0.90133333333333332

bag_clf.oob_decision_function_

array([[0.31746032, 0.68253968],
[0.34117647, 0.65882353],
[1. , 0. ],
...
[1. , 0. ],
[0.03108808, 0.96891192],
[0.57291667, 0.42708333]])
```

Podemos ver con *oob_decision_function_* la decisión que se tomo con cada registro.

### 1.5.3 Random Subspaces

De la misma forma que podemos conseguir diversidad alimentando diferentes juegos de datos a diferentes clasificadores, podemos trocear las features, de modo que no alimentemos a todos los clasificadores con las mismas features. Esto puede ser interesante en casos en los que haya muchas dimensiones.

Para configurar estos subspaces hay dos hiper-parámetros muy parecidos a los que hemos usado para elegir aleatoriamente los datos:
- max_features. Define cuantas features se encargará de gestionar cada clasificador 
- bootstrap_features. Define si la misma feature puede o no usarse en más de un clasificador - y si habrá features que pudieran no usarse en absoluto

### 1.5.4 Random Forests

Es un caso particular de ensemble, en el que el clasificador son Decission Trees, y se usa _bootstrap=True_.

```py
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
```

Viene a ser equivalente a:

```py
bag_clf = BaggingClassifier(DecisionTreeClassifier(splitter="random", max_leaf_nodes=16),n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1)
```

Digo viene porque _RandomForestClassifier_ introduce algunas optimizaciones sobre la forma general de crear ensembles.

### 1.5.4.1 Importancia de cada Feature

Podemos evaluar la importancia de cada feature en el decission tree. La importancia se mide evaluando el efecto que la feature tiene en la reducción del gini/entropía en el arbol. Esta información está accesible en **feature_importances_**:


```py
from sklearn.datasets import load_iris

iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(iris["data"], iris["target"])

for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
	print(name, score)

sepal length (cm) 0.112492250999
sepal width (cm) 0.0231192882825
petal length (cm) 0.441030464364
petal width (cm) 0.423357996355
```
### 1.5.6 Boosting

Con _Boosting_ lo que hacemos es hacer varias rondas de entrenamiento de forma secuencial, con diferentes clasificadores. La segunda ronda se apoya en los resultados de la primera ronda para mejorar la estimación, y así sucesivamente.
#### AdaBoostClassifier

Con _AdaBoosting_ lo que hacemos es hacer varias rondas de entrenamiento. En la primer ronda entrenamos los clasificadores, y miramos aquellas instancias que se han clasificado peor. A continuación haremos otra ronda de entrenamiento, pero alimentaremos estas instancias con un peso mayor, de modo que se mejore el resultado del aprendizaje. Esto se hará varias veces hasta que estemos satisfechos con el resultado. 

```py
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200,
algorithm="SAMME.R", learning_rate=0.5)

ada_clf.fit(X_train, y_train)
```

#### GradientBoostingRegressor

La idea es la misma que en el AdaBoosting, pero en lugar de ajustar pesos de instancias, lo que haremos es en la segunda ronda estimar el error, el residuo, obtenido en la primera, y así sucesivamente. Por ejemplo, hacemos un entrenamiento:

```py
from sklearn.tree import DecisionTreeRegressor

tree_reg1 = DecisionTreeRegressor(max_depth=2)
tree_reg1.fit(X, y)
```

El segundo predictor, *tree_reg2*, modelara el residuo del primero:

```py
y2 = y - tree_reg1.predict(X)

tree_reg2 = DecisionTreeRegressor(max_depth=2)
tree_reg2.fit(X, y2)
```

El tercero hará lo propio con el residuo del segundo:

```py
y3 = y2 - tree_reg2.predict(X)

tree_reg3 = DecisionTreeRegressor(max_depth=2)
tree_reg3.fit(X, y3)
```

Finalmente, cuando queramos hacer una estimación, tomaremos la suma de las estimaciones de los tres modelos:

```py
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
```

En sklearn existe una clase, *GradientBoostingRegressor*, que implementa exactamente lo que acabamos de hacer

```py
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(X, y)
```
