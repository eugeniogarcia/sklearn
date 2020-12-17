Tensorflow es una librería construida en c++, que es similar a NumPy pero que es compatible con GPUs. Admite procesamiento distribuido. Usa un compilador JIT para optimizar el uso de recursos de computo y memoria.

Sobre Tensorflow se construyen muchas features, incluida Keras:

![Tensor](.\imagenes\librerias_tf.png)

Tensorflow soporta varios Kernels (CPUs, GPUs, y TPUs). Se puede usar la API en varios lenguajes, Python, Go, etc. Tensorflow soporta Nvidia GPUs por medio de CUDA.

![Tensor](.\imagenes\tf.png)

## Basics de Tensorflow

### Operaciones más habituales

```py
t=tf.constant([[1., 2., 3.], [4., 5., 6.]])
t.shape
t.dtype
t[:, 1:]
f.constant(42)
t + 10
tf.square(t)
tf.sqrt(t)
tf.transpose(t)
```

### Keras

Keras esta implementada en Tensorflow, de modo que podemos usar código compatible con Keras - y por lo tanto transladable a otras implementaciones de Keras -, dentro del propio tensorflow:

```py
from tensorflow import keras

K = keras.backend
K.square(K.transpose(t)) + 10
```

### Numpy y Tensorflow

Pomdeos convertir un narray en un tensor fácilmente:

```py
a = np.array([2., 4., 5.])
tf.constant(a)
```

### Fuertemente tipado

No se pueden "mezclar" tensores de diferentes tipos. Esto daría un error:

```py
tf.constant(2.) + tf.constant(40)

tf.constant(2.) + tf.constant(40., dtype=tf.float64)
```

Podemos hacer cast:

```py
t2 = tf.constant(40., dtype=tf.float64)

tf.constant(2.0) + tf.cast(t2, tf.float32)
```

### Variables

Sobre la variables podemos aplicar las mismas operaciones que podíamos aplicar sobre las constantes, pero además, podemos cambiar su valor usando `assign()`, `assign_add()` o `assign_sub()`:

```py
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])

v.assign(2 * v) # => [[2., 4., 6.], [8., 10., 12.]]
v[0, 1].assign(42)
```

## Personalización de Modelos

### Funciones de Coste

Por ejemplo, definamos una función que implementa `huber`:

```py
def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)
```

Usarla es sencillo:

```py
model.compile(loss=huber_fn, optimizer="nadam")

model.fit(X_train, y_train, [...])
``` 

### Guardar y Cargar modelos

Al guardar el modelo, lo que estamos guardando es el nombre de la función de coste personalizada que hemos usado, _hubber_fn_ en este caso. Al recuperar el modelo, tenemos que indicar cual es la implementación que tenemos que asociar a huber_fn_:

```py
#Asociamos a la etiqueta huber_fn la implementación de la función a la que se corresponde
model = keras.models.load_model("my_model_with_a_custom_loss.h5", custom_objects={"huber_fn": huber_fn})
```

Si nuestra función fuera como sigue:

```py
def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss = threshold * tf.abs(error) - threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fn

model.compile(loss=create_huber(2.0), optimizer="nadam")
```

Al cargarla especificaríamos:

```py
model = keras.models.load_model("my_model_with_a_custom_loss_threshold_2.h5", custom_objects={"huber_fn": create_huber(2.0)})
```

Para evitar tener que "recordar" el argumento de la función, podríamos crear la función de coste como una hija de `keras.losses.Loss`, e implementar el método `get_config`:

```py
class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}
```

En `call` tenemos la implementación de la función de coste, en el constructor pasamos el argumento que usa la función de coste, y en `get_config` especificamos los parametros que tienen que ser guardados con la función.

Si tenemos:

```py
model.compile(loss=HuberLoss(2.), optimizer="nadam")
```

Cuando guardemos el modelo se guardará también el parámetro _2._. Al cargarlo bastara con hacer:

```py
model = keras.models.load_model("my_model_with_a_custom_loss_class.h5",
custom_objects={"HuberLoss": HuberLoss})
```

### Activacion, Inicialización de Pesos, Regularización de Pesos, Constrains de Pesos

Podemos customizar:
- Funciones de activación
- Funciones de inicialización de pesos
- Funciones de regulaización
- Restricciones de pesos

```py
def my_softplus(z): # return value is just tf.nn.softplus(z)
    return tf.math.log(tf.exp(z) + 1.0)

def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)

def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01 * weights))

def my_positive_weights(weights): # return value is just tf.nn.relu(weights)
    return tf.where(weights < 0., tf.zeros_like(weights), weights)
```

Podemos crear una capa con todas estas funciones personalizadas:

```py
layer = keras.layers.Dense(30, activation=my_softplus,
                kernel_initializer=my_glorot_initializer,
                kernel_regularizer=my_l1_regularizer,
                kernel_constraint=my_positive_weights)
```

Cuando necesitemos guardar parámetros junto con el modelo, tendremos que hacer una clase hija. Por ejemplo aquí creamos un regularizer:

```py
class MyL1Regularizer(keras.regularizers.Regularizer):
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor * weights))
    
    def get_config(self):
        return {"factor": self.factor}
```

### Metricas

Con las métricas sucede lo mismo que con los otros objetos que hemos visto, podemos definir una función, o si necesitamos mantener el estado, heredar de una clase base.


En este ejemplo usaremo huber, no como función de coste sino como métrica:

```py
class HuberMetric(keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs) # handles base args (e.g., dtype)
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

def update_state(self, y_true, y_pred, sample_weight=None):
    metric = self.huber_fn(y_true, y_pred)
    self.total.assign_add(tf.reduce_sum(metric))
    self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

def result(self):
    return self.total / self.count

def get_config(self):
    base_config = super().get_config()
    return {**base_config, "threshold": self.threshold}
```


- En el constructor guardamos los parámetros que serán necesarios para calcular la metrica. Se usa `self.add_weight`
- Se implementa `get_config` para definir que se guarda con el modelo. En este caso guardamos el threshold
- `update_state` se llama con cada batch. Keras ira guardando con cada epoch el valor medio de la metrica
- `result` recupera el valor de la métrica


Para usar esta métrica:

```py
model.compile(loss="mse", optimizer="nadam", metrics=[HuberMetric(2.0)])
```

### Capas custom

