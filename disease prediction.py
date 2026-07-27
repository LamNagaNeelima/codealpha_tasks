import tensorflow as tf
from tensorflow.keras import layers, models

input_img = tf.keras.Input(shape=(32, 128, 1), name='image')

x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2))(x)

new_shape = ((32//4), (128//4) * 64)
x = layers.Reshape(target_shape=new_shape)(x)

x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

outputs = layers.Dense(27, activation='softmax')(x)

model = models.Model(inputs=input_img, outputs=outputs)

model.summary()