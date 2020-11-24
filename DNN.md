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

```py
optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-4)
```

Donde _1 / decay_ es el número de pasos a partir de los cuales dividimos la _lr_ en otra unidad.

- __Exponential schedulling__. El learning arranca con un valor _lr0_, que cada _s_ pasos se divide. La secuencia sería: _lr0_, _lr0 * .1_, _lr0 * .1^2_, _lr0 * .1^3_, _lr0 * .1^4_, ...

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
Cuando se graba un modelo, los parámetros, incluidos la _lr_, se guardan con él, de modo que cuando se cargue los datos del modelo, se retomará en el mismo estado en el que estaba cuando se guardó. Sin embargo, cuando usamos _epoch_, aquí tenemos un problema, porque _epoch_ se reseterará a cero cada vez que llamamos al método _fit()_. Una opción sería pasar el parámetro __initial_epoch__ cuando llamemos a _fit()_.

- __Picewise constant schedulling__. Con este enfoque tendríamos algo así como _lr0_ los primeros 5 epochs, _lr1_ los siguientes 50, etc.
- __Performance schedulling__. Usaríamos un algoritmo similar a los que se usan para para el training de forma anticipada