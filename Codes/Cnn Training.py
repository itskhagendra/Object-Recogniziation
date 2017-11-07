import numpy as np
import matplotlib
import matplotlib as plt
from keras import *
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation
from helper import *

matplotlib.style.use('ggplot')
class_names = get_class_names()
print(class_names)
num_classes = len(class_names)
print(num_classes)
IMAGE_SIZE = 32
CHANNELS = 3
images_train, labels_train, class_train = get_train_data()
images_test, labels_test, class_test = get_test_data()


def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0, 25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    # model.summary()

    return model


model = cnn_model()
checkpoint = ModelCheckpoint('best_model_simple.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1.0e-4), metrics=['accuracy'])
model_details = model.fit(images_train, class_train, batch_size=128, epochs=350,
                          validation_data=(images_test, class_test), callbacks=[checkpoint], verbose=1)
scores = model.evaluate(images_test, class_test, verbose=0)
print("Accuracy:", scores[1] * 100)
plot_model(model_details)
class_pred = model.predict(images_test, batch_size=32)
labels_pred = np.argmax(class_pred, axis=1)
correct = (labels_pred == labels_test)
print("Correct Predictions:%d" % sum(correct))
