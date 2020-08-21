## Transfer Learning With MobileNet

### Imports and image batch creation:
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt
%matplotlib inline

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
print("num GPUs Available:", len(gpu_devices))
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

# import zipfile
# with zipfile.ZipFile("play.zip", 'r') as zip_ref:
#     zip_ref.extractall(".")

train_path = os.path.join("play/train")
test_path = os.path.join("play/test")
val_path = os.path.join("play/val")

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=train_path, target_size=(224,224), batch_size=10, classes=['0','1','2','3','4','5','6','7','8','9'])
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=test_path, target_size=(224,224), batch_size=10, classes=['0','1','2','3','4','5','6','7','8','9'], shuffle= False)
val_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=val_path, target_size=(224,224), batch_size=10, classes=['0','1','2','3','4','5','6','7','8','9'])



```

### Downloading Mobilenet model:
```python
mobile = tf.keras.applications.mobilenet.MobileNet()
mobile.summary()

```

### Adding mobile net layers to a new model for transfer learning:

```python
base_model = mobile.layers[-6].output
output = Dense(units=10, activation="softmax")(base_model)
model = Model(inputs=mobile.input, outputs=output)

model.summary()

# Lets make all layers except for last 23 layers as untrainable
for layer in model.layers[:-23]:
  layer.trainable = False
```

### Compiling and training the model:

```python
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(
    x=train_batches,
    steps_per_epoch = len(train_batches),
    validation_data = val_batches,
    validation_steps=len(val_batches),
    verbose=2,
    epochs=30
)

predictions = model.predict(x=test_batches)

predicted_indices = np.argmax(predictions,axis=-1)
```

### Plotting confusion matrix:

```python
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm = confusion_matrix(y_true=test_batches.classes, y_pred=predicted_indexes)
plot_confusion_matrix(cm, classes=class_labels)
```

### Pre process function for single image

```python
def preprocess_image(image_path):
  file_path = os.path.join(image_path)
  img = image.load_img(file_path, target_size=(224,224))
  img_array = image.img_to_array(img)
  imag_array_expanded_dims = np.expand_dims(img_array, axis = 0)
  return tf.keras.applications.mobilenet.preprocess_input(imag_array_expanded_dims)

test_image = preprocess_image('nine2.jpg')

prediction = model.predict(x=test_image)

print(np.argmax(prediction))
```