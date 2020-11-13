# Algoritmos

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






