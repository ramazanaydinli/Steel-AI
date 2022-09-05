
import tensorflow as tf
import numpy as np


train_path = 'C:\\Users\\METE\Desktop\\LevelUp\\generated_images\\training'
validation_path = 'C:\\Users\\METE\Desktop\\LevelUp\\generated_images\\validation'
prediction_path = 'C:\\Users\\METE\\Desktop\\LevelUp\\test_images\\ipn_trial.png'

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(400, 400, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                                horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(400, 400),
        batch_size=128,
        class_mode='categorical')
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                               horizontal_flip=True)

validation_generator = test_datagen.flow_from_directory(
    validation_path,
    target_size=(400, 400),
    batch_size=32,
    class_mode='categorical'
)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        """
        Stops training if the specified parameter reaches target value
        :param epoch: epoch number
        :param logs: log file
        :return: 0
        """
        if logs.get('loss') < 0.05:
            print("\n Accuracy is reached %95 or more, training stopped!")
            self.model.stop_training = True


callbacks = myCallback()

history = model.fit(train_generator, steps_per_epoch=8, epochs=20, validation_data=validation_generator,
                    validation_steps=8, verbose=1, callbacks=[callbacks])


img = tf.keras.utils.load_img(prediction_path, target_size=(400, 400))
x = tf.keras.utils.img_to_array(img)
x /= 255
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images)
class_names = train_generator.class_indices
score = classes[0]
print(class_names)
print(score)
