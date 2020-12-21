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

Si necesitaramos crear una capa custom podemos hacerlo de dos formas. Si la capa no tiene pesos que tengan que aprenderse, no tiene estado, podemos definir una función y usarla con el wrapper `keras.layers.Lambda`:

```py
exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))
```

Si necesitamos pesos, habrá que heredar de `keras.layers.Layer`:

```py
class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, batch_input_shape):
        self.kernel = self.add_weight(
        name="kernel", shape=[batch_input_shape[-1], self.units], initializer="glorot_normal")
        self.bias = self.add_weight(name="bias", shape=[self.units], initializer="zeros")
        super().build(batch_input_shape) # must be at the end

    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)

    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units, "activation": keras.activations.serialize(self.activation)}
```

- El constructor llama a. constructor base pasando los `**kwargs`. Guardamos los hiperparámetros
- `build` se llama una sola vez, la primera vez que la capa sea utilizada. Creamos los pesos usando `add_weight`, y tenemos acceso a la shape del input. Al final se terminará llamando al método `build` base. Hay que hacerlo como último paso de la función
- `call` se llama con cada uso que se haga de la capa.
- `compute_output_shape` se utiliza cuando necesitemos saber la shape de la capa
- `get_config` crea el diccionario de datos que se guardara con la capa cuando se guarde el modelo 

Cuando queramos que la capa se comporte de forma diferente durante el training, usaremos el argumento `training` del método call:

```py
def call(self, X, training=None):
    if training:
        noise = tf.random.normal(tf.shape(X), stddev=self.stddev)
        return X + noise
    else:
        return X
```

### Modelos custom

Para crear un modelo custom heredamos de `keras.models.Model`:

```py
class ResidualRegressor(keras.models.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(30, activation="elu",kernel_initializer="he_normal")
        self.block1 = ResidualBlock(2, 30)
        self.block2 = ResidualBlock(2, 30)
        self.out = keras.layers.Dense(output_dim)

    def call(self, inputs):
        Z = self.hidden1(inputs)
        for _ in range(1 + 3):
            Z = self.block1(Z)
        Z = self.block2(Z)
        return self.out(Z)
```

La clase Model es una clase hija de `keras.layers.Layer`, e incluye métodos como `compile()`, `fit()`, `evaluate()` y `predict()`.

### Errores (losses) y métricas con datos "internos" del modelo

Si queremos incluir entre los losses algun dato relativo al modelo, usaremos el método `self.add_loss`, como podemos ver aquí en el método `call`:

```py
class ReconstructingRegressor(keras.models.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(30, activation="selu",kernel_initializer="lecun_normal") for _ in range(5)]
        self.out = keras.layers.Dense(output_dim)

    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        self.reconstruct = keras.layers.Dense(n_inputs)
        super().build(batch_input_shape)
    
    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        reconstruction = self.reconstruct(Z)
        recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
        self.add_loss(0.05 * recon_loss)
        return self.out(Z)
```

## Cálculo del Gradiente

Definimos una función:

```py
def f(w1, w2):
    return 3 * w1 ** 2 + 2 * w1 * w2
```

Calculamos el gradiente

```py
w1, w2 = tf.Variable(5.), tf.Variable(3.)

with tf.GradientTape() as tape:
    z = f(w1, w2)

gradients = tape.gradient(z, [w1, w2])
```

Tras calcular el gradiente, el contexto se elimina. Si queremos calcular el gradiente más de una vez tendremos que hacerlo persistente:

```py
with tf.GradientTape(persistent=True) as tape:
    z = f(w1, w2)

dz_dw1 = tape.gradient(z, w1) # => tensor 36.0
dz_dw2 = tape.gradient(z, w2) # => tensor 10.0, works fine now!
del tape
```

Para calcular el gradiente se tiene que derivar contra variables tensor flow. Si derivamos contra una constante obtendremos _None_ como resultado:

```py
c1, c2 = tf.constant(5.), tf.constant(3.)

with tf.GradientTape() as tape:
    z = f(c1, c2)

gradients = tape.gradient(z, [c1, c2]) # returns [None, None]
```

Si queremos derivar contra un determinado tensor, tenemos que pedir a GradientTape que lo monitorice; Hacemos esto usando watch:

```py
with tf.GradientTape() as tape:
    tape.watch(c1)
    tape.watch(c2)
    z = f(c1, c2)

gradients = tape.gradient(z, [c1, c2]) # returns [tensor 36., tensor 10.]
```

En el ejemplo anterior hemos pedido a GradientTape que trate _c1_ y _c2_ como variables, de modo que podamos ver cual es el efecto de cambiar las constantes en la función.

Podemos calcular __derivadas segundas__ anidando GradientTapes. La derivada primera la vamos a tener que usar varias veces, por este motivo tenemos que hacerla persistente:

```py
with tf.GradientTape(persistent=True) as hessian_tape:
    with tf.GradientTape() as jacobian_tape:
        z = f(w1, w2)
    jacobians = jacobian_tape.gradient(z, [w1, w2])
    
hessians = [hessian_tape.gradient(jacobian, [w1, w2]) for jacobian in jacobians]

del hessian_tape
```

__NOTA__: Tensorflow retornara _None_ cuando se use una función para la que no exista el gradiente, o se este derivando contra una variable independiente.

### No propagar el gradiente

Si no quisieramos trasladar el cálculo del gradiente a una parte de la función, podemos usar `tf.stop_gradient`:

```py
def f(w1, w2):
    return 3 * w1 ** 2 + tf.stop_gradient(2 * w1 * w2)

with tf.GradientTape() as tape:
    z = f(w1, w2) # same result as without stop_gradient()

gradients = tape.gradient(z, [w1, w2]) # => returns [tensor 30., None]
```

Vemos como la derivada parcial respecto a _w2_ retorna _None_. Esto es debido a que hemos indicado que no se apliquen las derivadas sobre la parte de la función que depende de _w2_.

### Usar una derivada predenterminada

GradientTape utiliza cálculo numérico para calcular las derivadas. Si la función tiene  asíntoticamente a un valor para determinados valores, el cálculo numerico puede resultar inestable y eventualmente dar _NaN_. Si sabemos cual es la derivada, podemos usarla. Basta con definir una función que retorne el valor normal y el gradiente, y que este anotada con `tf.custom_gradient`:

```py
#Anotación
@tf.custom_gradient
def my_better_softplus(z):
    exp = tf.exp(z)
    #Devolvemos una función que retorna dos valores, el valor normal, y la derivada
    def my_softplus_gradients(grad):
        return grad / (1 + 1 / exp)
        return tf.math.log(exp + 1), my_softplus_gradients
```

### Custom fit

Ver los comentarios sobre el código:

```py
n_epochs = 5
batch_size = 32
n_steps = len(X_train) // batch_size
optimizer = keras.optimizers.Nadam(lr=0.01)
loss_fn = keras.losses.mean_squared_error
mean_loss = keras.metrics.Mean()
metrics = [keras.metrics.MeanAbsoluteError()]
```

```py
#El primer loop itera los epochs
for epoch in range(1, n_epochs + 1):
    print("Epoch {}/{}".format(epoch, n_epochs))
    #Cada epoch tiene una serie de mini-batches
    for step in range(1, n_steps + 1):
        #Obtenemos los datos del mini-batch
        X_batch, y_batch = random_batch(X_train_scaled, y_train)

        #Calculamos el la función de error como la suma del error del modelo, y los errore
        #de cada capa del modelo
        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)

        #Calculamos el gradiente del error respecto a las variables
        gradients = tape.gradient(loss, model.trainable_variables)

        #Calcula los nuevos pesos usando el optimizador
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        #Actualizamos el error del model
        mean_loss(loss)
        #Actualizamos las métricas tras adiestrar la epoch
        for metric in metrics:
            metric(y_batch, y_pred)

        print_status_bar(step * batch_size, len(y_train), mean_loss, metrics)

    print_status_bar(len(y_train), len(y_train), mean_loss, metrics)

    for metric in [mean_loss] + metrics:
        metric.reset_states()
```

## Funciones y gráficos

