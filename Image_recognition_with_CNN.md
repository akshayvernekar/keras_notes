## Training a simple CNN image recogniser using Keras

### Imports:
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
```

### Preparing image generators:
Now lets prepare image generators for feeding training , testing and validation data . 

The images for different classes are organised by their class names in each of training , validataion and test folders.

```python
train_path = os.path.join('train_images_path')
test_path = os.path.join('test_images_path')
val_path = os.path.join('val_images_path')

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size=(224,224), classes=['cat','dog'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path, target_size=(224,224), classes=['cat','dog'], batch_size=10, shuffle= False)
val_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=val_path, target_size=(224,224), classes=['cat','dog'], batch_size=10)

```

### Utility function to check sample images in ImageGenerators

```python
def plotImages(images_arr):
  fig, axes = plt.subplots(1, 10, figsize=(20,20))
  axes = axes.flatten()
  for img ,ax in zip( images_arr, axes):
    ax.axis('off')
    ax.imshow(img)
  plt.tight_layout()
  plt.show()

imgs, labels = next(train_batches)

plotImages(imgs)
```

### Creating the model

```python
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same',activation='relu',input_shape=(224,224,3)))
model.add(MaxPool2D(pool_size=(2,2),strides=2))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=2))
model.add(Flatten())
model.add(Dense(activation='softmax',units=2))
model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss ='categorical_crossentropy',metrics=['accuracy'])

model.fit(x=train_batches,validation_data=val_batches,steps_per_epoch=len(train_batches), validation_steps=len(val_batches), epochs=10, verbose=2)
```