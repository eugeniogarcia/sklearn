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



