import os
from inspect import getsourcefile
from os.path import abspath
import cv2
import random
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

path = os.path.dirname(abspath(getsourcefile(lambda:0)))

chars = os.listdir(os.path.join(path, 'data'))
y = []
X = []
print('Getting samples')
for i, char in enumerate(chars):
    print(chr(int(char)))
    samples = os.listdir(os.path.join(path, 'data', char))
    for sample in samples:
        x = cv2.imread(os.path.join(path, 'data', char, sample), cv2.IMREAD_GRAYSCALE)
        #x = x.reshape(1, 900).astype('float32')
        x = x.reshape(30, 30, 1)
        X.append(x)
        y.append(i)


print('train test split')
#X = np.vstack(X)
c = list(zip(X, y))
random.shuffle(c)
X, y = zip(*c)

X = np.reshape(X, (len(X), 30, 30, 1))
y = keras.utils.to_categorical(y, len(chars))
test_train_split = round(len(X) * .66)
X_train, y_train = X[:test_train_split], y[:test_train_split]
X_test, y_test = X[test_train_split:], y[test_train_split:]

X_train = X_train / 255
X_test = X_test / 255

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(30, 30, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(len(chars), activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

batch_size = 128
epochs = 15 

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("Baseline Error: %.2f%%" % (100-score[1]*100))