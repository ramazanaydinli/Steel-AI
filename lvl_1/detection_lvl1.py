
import tensorflow as tf
import numpy as np


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        'C:\\Users\\METE\Desktop\\LevelUp\\generated_images',
        target_size=(300, 300),
        batch_size=128,
        class_mode='binary')

history = model.fit(
      train_generator,
      steps_per_epoch=8,
      epochs=5,
      verbose=1)

prediction_path = 'C:\\Users\\METE\\Desktop\\LevelUp\\test_images\\ipn_trial.png'

img = tf.keras.utils.load_img(prediction_path, target_size=(300, 300))
x = tf.keras.utils.img_to_array(img)
x /= 255
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images)
class_names = train_generator.class_indices
score = classes[0]
if score < 0.5:
    print("This image most likely belongs to section images .")
else:
    print("This image most likely belongs to white images .")