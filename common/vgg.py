import tensorflow as tf

def VGG16_A(model_input, no_classes: int,
            layer_dropout_rate: float=None, input_dropout_rate: float=None,
            verbose: bool=False):
    if model_input is None:
        raise ValueError("VGG16_A - model input cannot be None")

    if no_classes < 1 is None:
        raise ValueError("VGG16_A - no_classes cannot be < 1")

    use_input_dropout = input_dropout_rate is not None and input_dropout_rate > 0.0
    use_layer_dropout = layer_dropout_rate is not None and input_dropout_rate > 0.0

    filter_sizes = [64, 128, 256, 256, 512, 512]
    kernel_sizes = [3, 3, 3, 3, 3, 3]
    fc_sizes     = [4096, 4096, 1000]

    if verbose:
        print("Using Input Dropout: {0}".format(use_input_dropout))
        print("Using Layer Dropout: {0}".format(use_layer_dropout))

    with tf.name_scope('input'):
        x = model_input
        if use_input_dropout:
            x = tf.keras.layers.Dropout(rate=input_dropout_rate)(x)

    for idx, filter_size in enumerate(filter_sizes):
        with tf.name_scope('layer{0}'.format(idx)):
            kernel_size = (kernel_sizes[idx], kernel_sizes[idx])

            if verbose:
                print("Adding VGG: {0}/ {1}".format(filter_size, kernel_size))
            x = vgg_layer(filter_size=filter_size, kernel_size=kernel_size, input=x)

            if use_layer_dropout:
                x = tf.keras.layers.Dropout(rate=layer_dropout_rate)(x)

    with tf.name_scope("output"):
        x = tf.keras.layers.Flatten()(x)
        for fc_size in fc_sizes:
            x = tf.keras.layers.Dense(fc_size)(x)
            if use_layer_dropout:
                x = tf.keras.layers.Dropout(rate=layer_dropout_rate)(x)

        model_output = tf.keras.layers.Dense(no_classes, activation='softmax')(x)
        model_output = model_output

    return model_output


def vgg_layer(filter_size: int, input, padding: str='same', activation: str='relu', kernel_size: (int, int)=(3, 3),
              use_batch_normalization=True,
              pool_size: (int, int) =(2, 2),
              no_conv_blocks: int = 2):
    x = input
    for _ in range(no_conv_blocks):
        x = tf.keras.layers.Conv2D(filters=filter_size, kernel_size=kernel_size, padding=padding)(x)

    if use_batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(activation=activation)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=pool_size, padding='same')(x)
    return x