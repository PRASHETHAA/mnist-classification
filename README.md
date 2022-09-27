# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value. 112
![1](https://user-images.githubusercontent.com/75237886/192455871-21feb6e0-d640-49d8-b162-d418f2089a99.png)

## Neural Network Model

![2](https://user-images.githubusercontent.com/75237886/192456803-3acfacee-5a13-458f-9e70-30d87591f4e3.png)


## DESIGN STEPS
## STEP-1:
Import tensorflow and preprocessing libraries

## STEP 2:
Download and load the dataset

## STEP 3:
Scale the dataset between it's min and max values

## STEP 4:
Using one hot encode, encode the categorical values

## STEP-5:
Split the data into train and test

## STEP-6:
Build the convolutional neural network model

## STEP-7:
Train the model with the training data

## STEP-8:
Plot the performance plot

## STEP-9:
Evaluate the model with the testing data

## STEP-10:
Fit the model and predict the single input
## PROGRAM
```python3
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
(x_train,y_train),(x_test,y_test)=mnist.load_data()
plt.imshow(x_train[0],cmap='gray')
x_train_scaled=x_train/255
x_test_scaled=x_test/255
print(x_train_scaled.min())
x_train_scaled.max()
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
x_train_scaled = x_train_scaled.reshape(-1,28,28,1)
x_test_scaled = x_test_scaled.reshape(-1,28,28,1)
model=Sequential([layers.Input(shape=(28,28,1)),
                  Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding='valid',activation='relu'),
                  MaxPool2D(pool_size=(2,2)),
                  Conv2D(filters=64,kernel_size=(5,5),strides=(1,1),padding='same',activation='relu'),
                  MaxPool2D(pool_size=(2,2)),
                  layers.Flatten(),
                  Dense(8,activation='relu'),
                  Dense(10,activation='softmax')
                  ])
model.summary()
model.compile(optimizer='Adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

model.fit(x_train_scaled ,y_train_onehot, epochs=15,
          batch_size=256, 
          validation_data=(x_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(x_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))

img = image.load_img('img.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
## ACCURACY VS VAL_ACCURACY
![3](https://user-images.githubusercontent.com/75237886/192456872-832609de-aa43-42b0-a392-ab9533434413.png)

## TRAINING_LOSS VS VAL_LOSS
![4](https://user-images.githubusercontent.com/75237886/192456943-d2072b47-6e3f-4d6e-9120-df33bfcf93db.png)
## Classification Report
![5](https://user-images.githubusercontent.com/75237886/192457411-de4616a7-2a6b-499d-af84-7ca8fc1ae296.png)

### Confusion Matrix
![6](https://user-images.githubusercontent.com/75237886/192457706-58039d56-a934-4c53-9ece-cf637388985f.png)

### New Sample Data Prediction
![7](https://user-images.githubusercontent.com/75237886/192457787-343e765a-740b-40bb-bf8c-a5ed08ef2641.png)

## RESULT
A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
