Hay algunos notebooks que no he resumido:
- Capitulo 13. Hace referencia al uso de Datasets, como encadenar operaciones en un Datasets, como procesar los datos que se han guardado en multiples archivos, como usar TFRecord; Se describe también el protobuf custom que se define en tensorflow - que es el vehiculo ideal para convertir a binario
- Capitulo 14. CNNs. 
    - Vemos como implementar un modelo con CNNs
    - Usar un modelo ya pre-construido, as-is. Cargar el modelo, y preprorcesar los datos para poderlos usar con el modelo
    - Usar un modelo ya pre-construido. Aprovechar las capas de un modelo pre-construido para sobre ellas, añadir nuestras capas custom
    - Como definir capas custom que enlaten convinaciones repetitivas - ejemplo, convolución + pooling. Uso de partial como una forma de dar valores por defecto
    - Como crear un modelo que clasifique una imagen, y __además__ la recuadre
    - Identificar multiples objetos en una imagen 
- Capítulos 15 y 16. RNNs
    - Crear modelos sequence -> vector y sequence -> sequence
    - Uso de RNN, LTSM, GRU
    - Uso de convoluciónes 1D como forma de simplificar el tamaño de la sequencia que debe ser procesado
    - Modelo para estimar el siguiente caracter de un texto
    - RNN con estado
    - Modelo para estimar el sentimiento transmitido por un comentario - de una película
    - Transferencia de Embeddings para mejorar el comportamiento del modelo de sentimiento
    - Encoder + Decoder para crear un modelo de traductor
- Cápitulo 17. Autoencoders y GAN
    - Autoencoders
        - Disminución de Dimensiones en datos
        - Para mejorar la calidad de modelos supervisados en caso de que no haya suficientes datos etiquetados
        - Entrenar un Autoencoder usando los mismos pesos en el Encoder y Decoder
        - Entrenar un Autoencoder capa a capa
        - Autoencoders con CNN
        - Autoencoders con RNN
        - Autoencoders como filtro de ruido en datos 
        - Sparse Autoencoders. Buscamos mejorar la precisión del Autoencoder forzando a que la salida del Encoder sea más sparse
        - Variational Autoencoders. La salida del Encoder en estos modelos es la media y la desviación estandar que pueden usarse para generar la entrada al Decoder siguiendo una distribución Normal. En este sentido podemos
            - Usar el encoder una vez y generar diferentes entradas para el Decoder
            - El modelo tiene un caríz de "generación", podemos "inventarnos" imagenes. Las GANs han desplazado a este tipo de modelos
    -GAN 
