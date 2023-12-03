from google.colab import drive
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

from google.colab import files
uploaded = files.upload()

# applying rescaling in test and validation datasets
train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

drive.mount('/content/gdrive')

# setting target size of the images to 200 to help model learn features
train_dataset = train.flow_from_directory('/content/gdrive/MyDrive/Dataset/training',
                                          target_size=(200, 200),
                                          batch_size=32,
                                          class_mode='categorical')

validation_dataset = train.flow_from_directory('/content/gdrive/MyDrive/Dataset/validation',
                                               target_size=(200, 200),
                                               batch_size=32,
                                               class_mode='categorical')

# define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                           input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPool2D(2, 2),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# compile the model with categorical cross-entropy loss, RMSprop optimizer with learning rate of 0.001 and accuracy as a metric
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              metrics=['accuracy'])

# fit the model on the training dataset with 10 epochs and validation dataset
history = model.fit(train_dataset, epochs=10,
                    validation_data=validation_dataset)

# plot the training and validation accuracy over epochs
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# predict the class of test images
dir_path = ('/content/gdrive/MyDrive/Dataset/alien_test')

for i in os.listdir(dir_path):
    img = image.load_img(dir_path + '/' + i, target_size=(200, 200))
    plt.imshow(img)
    plt.show()

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    classes = model.predict(images)
    # print(i)
    print("The Image Belong to Following Class: If value is 1 i.e the Image Belong to That particualr Class: ", classes)
