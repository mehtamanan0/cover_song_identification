import keras
import os
import numpy as np
from numpy import genfromtxt
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.externals import joblib

batch_size = 128
epochs = 12
num_classes = 2

#check for train file
if os.path.isfile('../data/models/train.pkl'):
    X, y = joblib.load('../data/models/train.pkl')

#generate training data
else:
    p_lis = [os.path.join('../data/csm/pair/', x) for x in os.listdir('../data/csm/pair/')]
    n_lis = [os.path.join('../data/csm/npair/', x) for x in os.listdir('../data/csm/npair/')]

    X = [genfromtxt(x, delimiter=',') for x in p_lis] + [genfromtxt(x, delimiter=',') for x in n_lis]
    y = [1] * len(p_lis) + [0] * len(n_lis)

    #dump model 
    joblib.dump((X,y), '../data/models/train.pkl')

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

x_train = np.stack(x_train, axis=0)
x_test = np.stack(x_test, axis=0)

#reshape
x_train = x_train.reshape(x_train.shape[0],180,180,1)
x_test = x_test.reshape(x_test.shape[0],180,180,1)

#standardise
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(180, 180, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

#compile
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#model fit
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print("saving model...")
joblib.dump(model, '../data/models/model.pkl')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
