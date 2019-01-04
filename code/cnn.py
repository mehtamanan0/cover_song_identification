import keras
import os
import numpy as np
from numpy import genfromtxt
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from sklearn.externals import joblib

batch_size = 64
epochs = 1
num_classes = 2

#check for train file
if os.path.isfile('../data/models/train.pkl'):
    print("Loading data from pickle")
    X, y = joblib.load('../data/models/train.pkl')

#generate training data
else:
    print("Generating data")
    p_lis = [os.path.join('../data/csm/pair/', x) for x in os.listdir('../data/csm/pair/')]
    n_lis = [os.path.join('../data/csm/npair/', x) for x in os.listdir('../data/csm/npair/')][:5000]

    X = [genfromtxt(x, delimiter=',') for x in p_lis] + [genfromtxt(x, delimiter=',') for x in n_lis]
    y = [1] * len(p_lis) + [0] * len(n_lis)

    #dump model 
    joblib.dump((X,y), '../data/models/train_{}.pkl'.format(len(n_lis)))

print("splitting")
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

x_train = np.stack(x_train, axis=0)
x_test = np.stack(x_test, axis=0)

#reshape
x_train = x_train.reshape(x_train.shape[0],180,180,1)
x_test = x_test.reshape(x_test.shape[0],180,180,1)

#standardise
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#model architecture
print(x_train.shape)
#block 1
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(180,180,1)))
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
print ("block 1 {}".format(model.output_shape))
#block 2
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
print ("block 2 {}".format(model.output_shape))
#block 3
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
print ("block 3 {}".format(model.output_shape))
#block 4
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
print ("block 4 {}".format(model.output_shape))
#block 5
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
print ("block 5 {}".format(model.output_shape))
#final layers
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
print ("block final {}".format(model.output_shape))
model.add(Dense(2, activation='softmax'))
print ("block final {}".format(model.output_shape))
#compile
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

#model fit
print("Training")
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print("saving model...")
joblib.dump(model, '../data/models/model_{}.pkl'.format(x_train.shape[0]))
print('Test loss:', score[0])
print('Test accuracy:', score[1])
