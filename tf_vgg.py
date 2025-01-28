import tensorflow as tf
from keras import layers, models

def create_vgg_ll():
    base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet',
                                             input_shape=(200, 200, 3))
    
    for layer in base_model.layers[:-1]:
        layer.trainable = False

    model_vgg16 = models.Sequential()
    model_vgg16.add(base_model)
    model_vgg16.add(layers.Flatten())
    model_vgg16.add(layers.Dense(5, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01), name='race_output'))
    model_vgg16.summary()

    model_vgg16.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy', 'mae'])

    return model_vgg16