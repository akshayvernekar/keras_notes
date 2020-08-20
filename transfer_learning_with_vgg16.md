## Transfer Learning With VGG16

We will learn how to train a pretraied model such as VGG16 to recognise the images of classes we desire ('Dog', 'Cat') to get improved accuracy.

### Imports
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

### Configure TF to use GPU

```python
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
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

### Downloading VGG16 model:
```python
vgg16_model = tf.keras.applications.vgg16.VGG16()
```

Lets create our own model now using Sequential

```python
model = Sequential()

# adding layers from VGG16 into our model
# We are adding all layers except the last layer. 
for layer in vgg16_model.layers[:-1]:
  model.add(layer)

# Lets mark these new added layers as non trainable so that the weights in these layers are not updated when training with our custom dataset
for layer in model.layers:
  layer.trainable = False

# lets add a Dense layer for detecting our classes
model.add(Dense(units=2, activation='softmax'))

model.summary()

```

### Lets prepare input data:

```python
import zipfile
with zipfile.ZipFile('tf_dog_cat.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

train_path = os.path.join('tf_dog_cat/train')
val_path = os.path.join('tf_dog_cat/val')
test_path = os.path.join('tf_dog_cat/test')

train_bactches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, batch_size = 10, target_size=(224,224), classes= ['cat','dog'])
test_bactches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path, batch_size = 10, target_size=(224,224),shuffle= False, classes= ['cat','dog'])
val_bactches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=val_path, batch_size = 10, target_size=(224,224), classes= ['cat','dog'])
```

### Training the model:
```python
model.compile(optimizer=Adam(learning_rate=0.0001), loss= 'categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_bactches, validation_data=val_bactches, epochs=5, steps_per_epoch=len(train_bactches), validation_steps=len(val_bactches), verbose=2)
```

### Infering test data:
```python
predictions = model.predict(x=test_bactches)
predict_indexes = np.argmax(predictions, axis = -1)

cm = confusion_matrix(y_pred=predic_indxes, y_true=test_bactches.classes )

plot_confusion_matrix(cm, classes=['cat','dog'])

```