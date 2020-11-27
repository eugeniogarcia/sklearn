# 1 Problemas con el Gradiente

El algoritmo de _backpropagation_ funciona a) navegando desde la capa de salida hacia la entrada para calcular el gradiente de la función de coste/error en cada capa, y b), una vez que el gradiente se ha calculado, navega desde la entrada hacia la salida para actualizar los parámetros de cada capa - usando el gradiente calculado.

Históricamente se configuraban las capas con pesos aleatorios, que seguían una normal de media 0 y std-dev 1, y usando una activación sigmoidal. Esto provoca dos problemas:

- __Vanishing gradient__. El gradiente en cada capa va haciendose más pequeño a medida que nos acercamos a la entrada. Esto tiene como resultado que los parámetros de esas capas no cambian, no se adiestran, y el algorimo no converge
- __Exploding gradient__. A medida que vamos subiendo desde la entrada calculando las salidas, nos sucede lo contraro, que se va ampliando el valor, de modo que las oscilaciones hacen que el algoritmo no converga

La derivada de la función sigmoidal en los extremos tiende a cero, lo que contribuye al vanishing, y al rededor de certo tiene un valor medio de 0.5, lo que contribuye al exploding.

Un factor que se observa y que explica estos fenómenos, es que la std-dev de los datos que alimentan una capa es menor de la  std-dev de las salidas. Lo que necesitamos para evitar los dos problemas que hemos descrito, es que la std-dev en las entradas y salidas se mantenga con un valor similar tanto cuando navegamos hacia delatante, como cuando vamos hacía atras. En este sentido juega un factor clave el número de entradas y neuronas de cada capa sea igual - *fan-in*, y *fan-out*.

## 1.1 Inicialización de pesos

Lo que observamos es que inicializando los pesos con media 0, pero con una std-dev diferente a 1, consiguimos aumentar la estabilidad del algoritmo. Empiricamente se ha comprobado que según la activación que se use, la std-dev a aplicar a los pesos que resulta más conveniente es - se indica el nombre que se le ha dado a cada uno de estos criterios de inicialización:

|Inicialización|Activación|std dev|
|-------|-------|-------|
|Glorot|None, Tanh, Logistic, Softmax|1 / fan-avg|
|He|ReLU y sus variantes|2 / fan-in|
|LeCun|SELU|1 / fan-in|

## 1.2 Activaciones

- __Sigmoidal__. Saturación de valores -> vanishing gradient
- __Relu__. Evita la saturación de valores positivos, pero sufre de dying RELU
- __Leaky Relu (LRelu)__. Evita el dying RELU. Mantiene una pendiente constante y pequeña para valores negativos. La pendiente es un parámetro de la LRelu
- __Randomized Relu (RRelu)__. Similar a la LRelu, pero el parámetro se elije al azar con cada epoch
- __Parametrized Relu (PRelu)__. Similar a la LRelu, pero el parámetro se "aprende"
- __Exponencial Linear Unit (ELU)__. Para valores negativos sigue una exponencial que a infinito tiende a el valor del parámetro. El parámetro 
- __Scaled ELU (SELU)__. Como la ELU, pero la salida tiene a preservar que su media sea 0, y la std-dev sea 1

### 1.2.3 Elección

En orden de preferencia, este sería el criterio de selección de una activación:
```
SELU > ELU > LRelu & sus variantes > tanh > logistic
```
Podemo configurar cada escenario en Keras como sigue:

- __he__ para inicializar los pesos, pero con una distribución uniforme:

```py
he_avg_init = keras.initializers.VarianceScaling(scale=2., mode='fan_avg', distribution='uniform')
keras.layers.Dense(10, activation="sigmoid", kernel_initializer=he_avg_init)
```

- __Relu con he__:

```py
keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal")
```

- __Leaky Relu con he__:

```py
leaky_relu = keras.layers.LeakyReLU(alpha=0.2)
layer = keras.layers.Dense(10, activation=leaky_relu,kernel_initializer="he_normal")
```

- __SELU con lekun__:

```py
layer = keras.layers.Dense(10, activation="selu",kernel_initializer="lecun_normal")
``` 

## 1.3 Batch Normalization

Usando he con Elu o Relu disminuye el problema del vanishing y exploding gradient, normalizar cada una de las capas para que las salidas esten normalizadas, ayudara a reducir el problema.

Con una capa de normalización introducimos cuatro parametros, dos _trainables_ el _beta_ y el _bias_, y dos que se calculan, _la std-dev_ y la _media.

La salida será 

```py
z = _beta_ * x + bias_
```

```py
x = x-media/sqrt(std-dev)
```

- _beta_ y _bias_ se aprenderán usando _backpropagation_.
- _media_ y _std-dev_ se calculan durante el aprendizaje usando el estadístico de la media y de la desviación estandard.


Podemos definir la batch normalization en el modelo como una capa más:

```py
model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28, 28]),

keras.layers.BatchNormalization(),

keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),

keras.layers.BatchNormalization(),

keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),

keras.layers.BatchNormalization(),

keras.layers.Dense(10, activation="softmax")
])
```

La capa de batchnormalization introduce cuatro parámetros, pero solo dos de ellos, _gamma_ y _beta_ son entrenables. *moving_mean* y *moving_variance* se calculan:

```py
[(var.name, var.trainable) for var in model.layers[1].variables]

[('batch_normalization_v2/gamma:0', True),
('batch_normalization_v2/beta:0', True),
('batch_normalization_v2/moving_mean:0', False),
('batch_normalization_v2/moving_variance:0', False)]
```

En muchas ocasiones se recomienda añadir la capa de normalización antes de la activación. Bastaría con no especificar el parámetro _activation_ en las capas, y especificar la activación como una capa más, *keras.layers.Activation*:

```py
model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28, 28]),

keras.layers.BatchNormalization(),

keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False),

keras.layers.BatchNormalization(),

keras.layers.Activation("elu"),
keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
keras.layers.Activation("elu"),

keras.layers.BatchNormalization(),
keras.layers.Dense(10, activation="softmax")
])
```

La capa Batchnormalization tiene también algunos hiperparametros. Destacar *momentum*, que nos permite aplicar un momento sobre los valores que se calculan en la capa.

### 1.3.1 Gradient Cliping

Una forma de evitar el problema del *exploding gradient* es "caparle". Hay dos formas de hacerlo:
- clipvalue. Lo que hacemos es capar las dimensiones del gradiente. Tiene el efecto de cambiar no solo el módulo del gradiente, más tambien su dirección

```py
optimizer = keras.optimizers.SGD(clipvalue=1.0)
model.compile(loss="mse", optimizer=optimizer)
```

- clipnorm. "Capa" la norma del gradiente al tiempo que mantiene su dirección

```py
optimizer = keras.optimizers.SGD(clipnorm=1.0)
model.compile(loss="mse", optimizer=optimizer)
```

## 1.4 Learning Transfer

Cargamos un modelo, *model_A*, y lo usamos para definir un modelo *model_B_on_A*

```py
model_A = keras.models.load_model("my_model_A.h5")
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))
```

Con este enfoque cualquier cambio que hagamos en *model_B_on_A* sobre las capas que hemos cargado con *model_A*, afecta tambien a *model_A*. Para evitar esto, podemos clonar *model_A*. Al clonar estamos copiando solo la estructura, así que si queremos copiar también los pesos lo tenemos que hacer explicitamente: 

```py
model_A_clone = keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())
```

Durante el aprendizaje es habitual que se mantenga, al menos al principio, las capas que hemos *transferido* del modelo:

```py
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False
```

Por ejemplo, podemos entrenar el modelo manteniendo las capas *transferidas* congeladas, durante 4 epochs en este ejemplo, y luego entrenar todas las capas, usando una *learning rate* más pequeña, **de modo que los pesos no se cambien tanto**:

```py
model_B_on_A.compile(loss="binary_crossentropy", optimizer="sgd",
metrics=["accuracy"])

history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4, validation_data=(X_valid_B, y_valid_B))

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True

optimizer = keras.optimizers.SGD(lr=1e-4) # the default lr is 1e-3
model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer,metrics=["accuracy"])
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16,
validation_data=(X_valid_B, y_valid_B))
```

# 2. Optimizers

Tipicamente el _Gradient Descent (GD)_ aprenderá los parámetros de la siguiente forma:

```
par=par - lr * grad J(par)|par
```

- __Momento__. Incluimos una inercia en el cambio de los parametros. La inercia se controla con el parámetro *beta*. Con *beta* igual a cero estamos en el GD estandard. Si el gradiente fuera constante, los parametros se cambiarían con una velocidad `1 / (1 - beta)`.

```
m = m * momentum - lr * grad J(par)|par
par = par + m
```

En Keras se configura como sigue. Basta con usar el parámetro *momentum*:

```py
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
```

- __Nesterov__. Es una variante del _momento_. La diferencia reside en que en lugar de calcular el gradiente en _par_, lo calculamos en un punto separado de _par_ en la dirección del momento:

```
m = m * momentum - lr * grad J(par + beta * momentum)|par
par = par + m
```

En Keras se configura como sigue:

```py
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
```

- __AdaGrad__. Lo que hacemos en este algoritmo es normalizar el gradiente - escalarlo:

```
s = s + ( grad J(par)|par )^2

par=par - lr * grad J(par)|par / sqrt (s + epsilon)
```

AdaGrad funciona bien en problemas cuadráticos, pero en el resto de casos _momento_ o _nesterov_ funcionan mejor.

- __RMSProp__. Evoluciona _AdaGrad_ de modo que la norma del gradiente, en lugar de hacerse con todos los datos, se hace solo con los más recientes:

```
s = rho * s + (1 - rho) * ( grad J(par)|par )^2

par=par - lr * grad J(par)|par / sqrt (s + epsilon)
```

En Keras se configura como sigue:

```py
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)
```

- __Adam & Nadam__. _Adam, Adaptative Moment Estimation_, comvina las ideas de _momentum_ y _RMSProp_. 

En Keras se configura como sigue:

```py
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
```

*beta_1* es el equivalente al _momentum_, y *beta_2* equivalente a _rho_.

# 3. Learning Rate Scheduling

Con estas técnicas lo que se pretende es la _learning rate_ se vaya adapatando, con valores más grandes al inicio, y más pequeños al final, de modo que se consiga que el algoritmo converja - valores pequeños -, pero de forma rápida - valores grandes.

Hay cuatro técnicas:

- __Power schedulling__. El learning arranca con un valor _lr0_, que cada _s_ pasos se divide. La secuencia sería: _lr0_, _lr0 / 2_, _lr0 / 3_, _lr0 / 4_, _lr0 / 5_, ...

Para configurar este método basta con especificar el parámetro *decay* en el optimizer:

```py
optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-4)
```

Donde _1 / decay_ es el número de pasos a partir de los cuales dividimos la _lr_ en otra unidad.

- __Exponential schedulling__. El learning arranca con un valor _lr0_, que cada _s_ pasos se divide. La secuencia sería: _lr0_, _lr0 * .1_, _lr0 * .1^2_, _lr0 * .1^3_, _lr0 * .1^4_, ...

Esta optimización se configura especificando un callback. Se define un método que retorna el *lr*, y se usa el callback `keras.callbacks.LearningRateScheduler`:

```py
#Definimos una función que toma un par de parámetros y nos devuelve la función de decay
def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn

#Función de decay del lr
exponential_decay_fn = exponential_decay(lr0=0.01, s=20)

#Lo aplicamos al modelo
lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
history = model.fit(X_train_scaled, y_train, [...], callbacks=[lr_scheduler])
```

Otra forma de lograr lo mismo es usando un __custom callback__:

```py
K = keras.backend

class ExponentialDecay(keras.callbacks.Callback):
    def __init__(self, s=40000):
        super().__init__()
        self.s = s

    def on_batch_begin(self, batch, logs=None):
        # Note: the `batch` argument is reset at each epoch
        lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, lr * 0.1**(1 / s))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
```

Ahora lo podemos usar:

```py
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(10, activation="softmax")
])

lr0 = 0.01
optimizer = keras.optimizers.Nadam(lr=lr0)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
n_epochs = 25

s = 20 * len(X_train) // 32 # number of steps in 20 epochs (batch size = 32)
exp_decay = ExponentialDecay(s)
history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                    validation_data=(X_valid_scaled, y_valid),
                    callbacks=[exp_decay])
```

Cuando se graba un modelo, los parámetros, incluidos la _lr_, se guardan con él, de modo que cuando se cargue los datos del modelo, se retomará en el mismo estado en el que estaba cuando se guardó. Sin embargo, cuando usamos _epoch_, aquí tenemos un problema, porque _epoch_ se reseterará a cero cada vez que llamamos al método _fit()_. Una opción sería pasar el parámetro __initial_epoch__ cuando llamemos a _fit()_.

- __Picewise constant schedulling__. Con este enfoque tendríamos algo así como _lr0_ los primeros 5 epochs, _lr1_ los siguientes 50, etc.

Como antes, definimos un método que devuelve la *lr*:

```py
def piecewise_constant_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001
```

O si queremos hacerlo más sofisticado y que sea parametrizable:

```py
def piecewise_constant(boundaries, values):
    boundaries = np.array([0] + boundaries)
    values = np.array(values)
    def piecewise_constant_fn(epoch):
        return values[np.argmax(boundaries > epoch) - 1]
    return piecewise_constant_fn

piecewise_constant_fn = piecewise_constant([5, 15], [0.01, 0.005, 0.001])
```

Podemos usarlo, como antes, utilizando el callback `keras.callbacks.LearningRateScheduler`:

```py
lr_scheduler = keras.callbacks.LearningRateScheduler(piecewise_constant_fn)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
n_epochs = 25
history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                    validation_data=(X_valid_scaled, y_valid),
                    callbacks=[lr_scheduler])
```

- __Performance schedulling__. Usaríamos un algoritmo similar a los que se usan para para el training de forma anticipada

Podemos usar el callback `keras.callbacks.ReduceLROnPlateau`:

```py
lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
```

# 4. Overfitting

Para evitar el overfittig tenemos diferentes herramientas:
- Regularización
- Dropout

## 4.1 Regularización

Con la regularización "estimulamos" a la DNN para que mantenga la norma de los pesos pequeña. Para ello las capas con regularización introducen un _loss_ que se añade al coste de la red, y que se utiliza durante el entrenamiento. En Keras podemos implementarlo como sigue:

```py
layer = keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.01))
```

```py
layer = keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l1(0.01))
```

```py
layer = keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l1_l2(0.01))
```

Para evitar tener que repetir el código multiples veces, podemos usar `partial`:

```py
from functools import partial

RegularizedDense = partial(keras.layers.Dense,activation="elu",kernel_initializer="he_normal",kernel_regularizer=keras.regularizers.l2(0.01))

model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28, 28]),
RegularizedDense(300),
RegularizedDense(100),
RegularizedDense(10, activation="softmax",
kernel_initializer="glorot_uniform")
])
```

Con `partial` podemos derivar una función a partir de otra, especificando una serie de parámetros de modo que la función resultante es una especialización de la primera.

## 4.2 Dropout

Otro método eficaz para evitar el overfitting es usar dropout. Con esta técnica lo que hacemos es de forma aleatoria "desconectar" neuronas de la entrada o de capas intermedias - nunca de la salida -, de modo que efectivamente cuando entrenamos la DNN realmente estamos __en cada step__ una red diferente.

En Keras se implementa como sigue, apenas indicando la probabilidad de dropout:

```py
model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28, 28]),

keras.layers.Dropout(rate=0.2),

keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),

keras.layers.Dropout(rate=0.2),

keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),

keras.layers.Dropout(rate=0.2),

keras.layers.Dense(10, activation="softmax")
])
```

Indicar que en inferencia no se usara el Dropout. Esto significa que la DNN tiene que compensar por la falta "de intensidad". Por ejemplo, si en training el dropout rate era 50%, significa que a las salidas se activan con un 50% de las entradas, de media. Si en inferencia usaramos el modelo sin más, estaría recibiendo el donde de "intensidad", por eso hay que compensar en tiempo de inferencia, bien multiplicando los pesos por *1-p*, o dividiendo las salidas por *1-p*, donde *p* es la tasa de dropout - 0.5 en este ejemplo.

Cuando el modelo haga overfitting, __incrementaremos el dropout rate__, cuando haga underfittimg, __reduciremos el dropout rate__.

### 4.2.1 Alphadropout

Alphadropout es una variante de Dropout en la que se mantienen la media y std-dev. Tipicamente se usara junto con la activación *SElu*

```py
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),

    keras.layers.AlphaDropout(rate=0.2),

    keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),

    keras.layers.AlphaDropout(rate=0.2),

    keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),

    keras.layers.AlphaDropout(rate=0.2),

    keras.layers.Dense(10, activation="softmax")
])
```

### 4.2.2 Monte-Carlo (MC) Dropout

Con esta técnica lo que vamos a hacer es aplicar dropout durante la inferencia, ejecutar la inferencia un número determinado de veces - 100 en el ejemplo que sigue -, y tomar el valor medio con resultado de la inferencia:

```py
with keras.backend.learning_phase_scope(1): # force training mode = dropout on
    y_probas = np.stack([model.predict(X_test_scaled) for sample in range(100)])
    y_proba = y_probas.mean(axis=0)
    y_std = y_probas.std(axis=0)
```

Es equivalente a:

```py
y_probas = np.stack([model(X_test_scaled, training=True)
                     for sample in range(100)])
y_proba = y_probas.mean(axis=0)
y_std = y_probas.std(axis=0)
```

Si tomamos una de las respuestas:

```py
np.round(model.predict(X_test_scaled[:1]), 2)

array([[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.01, 0. , 0.99]], dtype=float32)
```

Sin embargo al ver el conjunto de respuestas de la emulación Montecarlo, no está tan claro que el dato sea de clase 9, hay una cierta varianza:

```py
np.round(y_probas[:, :1], 2)

array([[[0. , 0. , 0. , 0. , 0. , 0.14, 0. , 0.17, 0. , 0.68]],
[[0. , 0. , 0. , 0. , 0. , 0.16, 0. , 0.2 , 0. , 0.64]],
[[0. , 0. , 0. , 0. , 0. , 0.02, 0. , 0.01, 0. , 0.97]],
[...]
```

La precisión del modelo es mayor cuando aplicamos Montecarlo.

### 4.2.3 Emular Training

Podemos indicar a Keras que trabaje en modo *training* durante la fase de *inferencia* usando `keras.backend.learning_phase_scope(1)`, como en este ejemplo:

```py
with keras.backend.learning_phase_scope(1): # force training mode = dropout on
    y_probas=model.predict(X_test_scaled)
```

Otra forma de aplicar Dropout durante la inferencia, especialmente util cuando hay otras capas en el DNN que tienen un comportamiento diferente en modo entrenamiento y en modo inferencia (batch normalization) es el de crear una capa custom:

```py
class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)
```

## 4.3 Max-Norm Regularization

Con este método lo que hacemos es un cliping del peso:

```py
keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal",
kernel_constraint=keras.constraints.max_norm(1.))
```