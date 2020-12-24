# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 12:28:14 2020

@author: hasc
"""

from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from keras.layers import Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
 
import tensorflow as tf
import random as rn
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
             
                 

###   %matplotlib inline

train_gen = ImageDataGenerator(rescale = 1./255,  vertical_flip=True,     
                                  width_shift_range=0.1,  
                                  height_shift_range=0.1, 
                                  #rotation_range=5,
                                  #shear_range=0.7,
                                  #zoom_range=[0.9, 2.2],
                                  fill_mode='nearest')
train_generator = train_gen.flow_from_directory('./gdrive/My Drive/Colab Notebooks/train',target_size=(64,64), batch_size=15, class_mode = 'categorical')

test_gen = ImageDataGenerator(rescale = 1./255)
test_generator = test_gen.flow_from_directory('./gdrive/My Drive/Colab Notebooks/test',target_size=(64,64), batch_size=12, class_mode = 'categorical')

"""
x_batch, y_batch = next(train_gen)
print(x_batch.shape)
print(y_batch.shape)

for i in range(len(x_batch)):
  plt.imshow(x_batch[i].astype('int32'))
  plt.title(y_batch[i])
  plt.show()
"""

SEED = 3

np.random.seed(SEED)
rn.seed(SEED)
tf.random.set_seed(SEED)

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (64,64,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation = "softmax"))

#red_lr= ReduceLROnPlateau(monitor='val_loss',patience=10,verbose=1,factor = 0.9)

early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 18)
model.compile(optimizer=Adam(lr=0.0002),loss='categorical_crossentropy',metrics=['accuracy'])
#model.summary()
history = model.fit_generator(train_generator,steps_per_epoch = 30,
                              epochs = 100, validation_data = test_generator,
                              validation_steps= 10, callbacks=[early_stopping_callback] )

print("-- Evaluate --")
scores = model.evaluate_generator(test_generator, steps = 7)
print("%s: %.4f%%" %(model.metrics_names[1], scores[1]*100))

print("-- Predict --")
output = model.predict_generator(test_generator, steps= 7)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()


