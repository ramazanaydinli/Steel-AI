
import tensorflow as tf
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras import layers
from keras import Model


train_path = 'C:\\Users\\METE\Desktop\\LevelUp\\generated_images\\training'
validation_path = 'C:\\Users\\METE\Desktop\\LevelUp\\generated_images\\validation'
prediction_path = 'C:\\Users\\METE\\Desktop\\LevelUp\\test_images\\ipn_trial.png'
weights_file= 'C:\\Users\\METE\\Desktop\\LevelUp\\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(
    input_shape=(400, 400, 3),
    include_top=False,
    weights=None
)
pre_trained_model.load_weights(weights_file)
for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
last_output=last_layer.output
x= layers.Flatten()(last_output)
x=layers.Dense(1024, activation='relu')(x)
x=layers.Dense(5, activation='softmax')(x)
model = Model(pre_trained_model.input, x)
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                                horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        train_path,
        color_mode='rgb',
        target_size=(400, 400),
        batch_size=128,
        class_mode='categorical')
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                               horizontal_flip=True)

validation_generator = test_datagen.flow_from_directory(
    validation_path,
    color_mode='rgb',
    target_size=(400, 400),
    batch_size=32,
    class_mode='categorical')

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


img = tf.keras.utils.load_img(prediction_path, color_mode='rgb', target_size=(400, 400))
x = tf.keras.utils.img_to_array(img)
x /= 255
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images)
class_names = train_generator.class_indices
score = classes[0]
print(class_names)
print(score)
