#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import keras
from PIL import Image, ImageOps
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import optimizers
from scipy import io as spio

emnist = spio.loadmat("C:/Users/leechristo/Documents/CODE/TicTacToe/storage/emnist-letters.mat")

#%%

# load data, convert to float and normalize
x_train = emnist['dataset'][0][0][0][0][0][0]
x_test = emnist['dataset'][0][0][1][0][0][0]
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train /= 255
x_test /= 255

# load label data and store in alt vars for visualization
y_train = emnist['dataset'][0][0][0][0][0][1]
y_test = emnist['dataset'][0][0][1][0][0][1]

train_labels = y_train
test_labels = y_test

#%%

# reshape data and onehot encode labels
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_train = keras.utils.to_categorical(y_train, 27)
y_test = keras.utils.to_categorical(y_test, 27)

#%%

# index all the x's and o's
xo_train_index = []
xo_test_index = []

for label in range(train_labels.shape[0]):
    if train_labels[label][0] == 24 or train_labels[label][0] == 15:
        xo_train_index.append(label)

for label in range(test_labels.shape[0]):
    if test_labels[label][0] == 24 or test_labels[label][0] == 15:
        xo_test_index.append(label)

#%%

# separate x and o data
xo_train = np.ndarray(shape=(len(xo_train_index), 28, 28, 1))
xo_test = np.ndarray(shape=(len(xo_test_index), 28, 28, 1))

for element in xo_train_index:
      xo_train[xo_train_index.index(element)] = x_train[element]

for element in xo_test_index:
      xo_test[xo_test_index.index(element)] = x_test[element]

#%%

# separate x and o labels; reindex x=1, o=0
xo_train_labels = np.ndarray(shape=(len(xo_train_index), 1))
xo_test_labels = np.ndarray(shape=(len(xo_test_index), 1))

for element in xo_train_index:
      if train_labels[element][0] == 24:
            xo_train_labels[xo_train_index.index(element)][0] = 1
      elif train_labels[element][0] == 15:
            xo_train_labels[xo_train_index.index(element)][0] = 0

for element in xo_test_index:
      if test_labels[element][0] == 24:
            xo_test_labels[xo_test_index.index(element)][0] = 1
      elif test_labels[element][0] == 15:
            xo_test_labels[xo_test_index.index(element)][0] = 0

#%%

# define model architecture and hyperparameters
def gen_model():
      input_shape = (28,28,1)

      model = Sequential()
      model.add(Conv2D(32,(5,5),input_shape=input_shape))
      model.add(MaxPooling2D())
      model.add(Flatten())
      model.add(Dense(32,activation='relu'))
      model.add(Dropout(0.3))
      model.add(Dense(8,activation='relu'))
      model.add(Dropout(0.2))
      model.add(Dense(2,activation='softmax'))

      model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

      model.fit(x=xo_train, y=xo_train_labels, batch_size=16, epochs=10)

      return model

#%%

# train model and evaluate performance
model = gen_model()
model.evaluate(xo_test,xo_test_labels)

#%%

# test model using built-in sample image
classes = {0:'O',1:'X'}
samplenum = 444
plt.imshow(xo_test[samplenum].reshape(28,28),cmap='Greys')
pred = model.predict(xo_test[samplenum].reshape(1, 28, 28, 1))
print("Prediction: "+classes[pred.argmax()])
print("Confidence: "+str(pred[0][pred.argmax()]*100)+"%")

#%%

# test model using BYOI
image = Image.open('7.png').convert("L")
# image = ImageOps.invert(image)
image = image.resize((28,28)) 
image = img_to_array(image)
image.shape
image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
plt.imshow(image.reshape(28,28),cmap='Greys')
print(type(image))
pred = model.predict(image)
print("Prediction: "+classes[pred.argmax()])
print("Confidence: "+str(pred[0][pred.argmax()]*100)+"%")

#%%
from keras.models import model_from_yaml
import os

model_json = model.to_json()
with open("model.json", "w") as json_file:
      json_file.write(model_json)
model.save_weights("model.h5")

#%%

# MLP for Pima Indians Dataset serialize to YAML and HDF5
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_yaml
import numpy
import os

from keras.models import model_from_json
json_file = open("model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")


#%%
# test model using BYOI
image = Image.open('7.png').convert("L")
# image = ImageOps.invert(image)
image = image.resize((28,28)) 
image = img_to_array(image)
image.shape
image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
plt.imshow(image.reshape(28,28),cmap='Greys')
print(type(image))
pred = loaded_model.predict(image)
print("Prediction: "+classes[pred.argmax()])
print("Confidence: "+str(pred[0][pred.argmax()]*100)+"%")


#%%
import cv2
import model
model = model.XOModel(os.path.join('storage', 'model.yaml'), 
                                   os.path.join('storage', 'model.h5'))


#%%
import camera as cam
import board as asdf
mycam = cam.Camera(crop=(75,100,1450,1000))

board = asdf.Board(mycam.frame, calibrate=True, draw=True)
while 1:
      mycam.get_frame()
      
      board.update_image(mycam.frame)

      board.update(draw=True)
      cv2.imshow('frame', board.frame)
      cv2.imshow('mask', board.xo_mask)
      key = cv2.waitKey(0)
      if key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
      elif key == ord('c'):
            board.calibrate()

#%%
