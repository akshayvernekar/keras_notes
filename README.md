## Tensorflow Keras Quick Start

### Imports

```python
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
```

### Building a model
To build model we will use Keras's **Sequential API**

We will build a simple model composed of Dense layers

```python
model = models.Sequential()
model.add(Dense(units=16, input_shape=(1,), activation='relu'))
model.add(Dense(units=32, input_shape=(1,), activation='relu'))
model.add(Dense(units=2, input_shape=(1,), activation='softmax'))
```
To check the summary of the model

```python
model.summary()
```
To compile the model

```python
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
learning_rate : tells us how fast the weight are to be adjusted. Choosing lower learning rate will make the model take longer duration to train. Choosing a higher value might lead to model missing the local minima. 

loss : Loss function to use

### Getting predictions from the model

Assume we have test samples and corresponding labels in **test_samples** and **test_labels**

```python
predictions = model.predict(test_samples, batch_size= 10)
```
Getting the index of prediction with max probablity

```python
index_predictions = np.argmax(predictions, -1)
```

### Plotting confusion matirx
We will use sklearn's confusion matrix function to plot

```python
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
```
Plot Confusion Matric function

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
```

Lets plot confusion matrix

```python
cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)

cm_plot_labels = ['no_side_effects','side_effects']

plot_confusion_matrix(cm, classes=cm_plot_labels, title='Confusion matrix')
```