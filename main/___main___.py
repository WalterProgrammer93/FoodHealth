import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# Carga del conjunto de datos de imágenes de alimentos
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    directory='./data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Definición del modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compilación y entrenamiento del modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=len(train_generator), epochs=10)

# Carga de la imagen del alimento a identificar
image = tf.keras.preprocessing.image.load_img('./image.jpg', target_size=(224, 224))
image = tf.keras.preprocessing.image.img_to_array(image)
image = np.expand_dims(image, axis=0)

# Predicción de la categoría del alimento
prediction = model.predict(image)
predicted_class = np.argmax(prediction[0])
print('Categoría del alimento:', train_generator.class_indices[predicted_class])

# Obtención de información nutricional
nutritional_info = tf.get_nutritional_info(predicted_class)
print('Información nutricional:', nutritional_info)