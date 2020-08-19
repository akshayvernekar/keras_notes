## Tensorflow Keras Quick Start

### Imports

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


### Building a model
To build model we will use Keras's **Sequential API**

We will build a simple model composed of Dense layers

	model = models.Sequential()
	model.add(Dense(units=16, input_shape=(1,), activation='relu'))
	model.add(Dense(units=32, input_shape=(1,), activation='relu'))
	model.add(Dense(units=2, input_shape=(1,), activation='softmax'))

To check the summary of the model

	model.summary()

To compile the model

	model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

learning_rate : tells us how fast the weight are to be adjusted. Choosing lower learning rate will make the model take longer duration to train. Choosing a higher value might lead to model missing the local minima. 

loss : Loss function to use
