import matplotlib
from keras.callbacks import ModelCheckpoint
from keras.layers import GlobalAveragePooling2D, Conv2D, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import Adam

from helper import get_class_names, get_train_data, get_test_data

matplotlib.style.use('ggplot')
class_names = get_class_names()
num_classes = len(class_names)
IMAGE_SIZE = 32
CHANNELS = 3
NUM_EPOCH = 350
LEARN_RATE = 1.0e-4
images_train, labels_train, class_train = get_train_data()
images_test, labels_test, class_test = get_test_data()


def pure_cnn_model():
    model = Sequential()

    model.add(Conv2D(96, (3, 3), activation='relu', padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)))
    model.add(Dropout(0.2))

    model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(96, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(192, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(192, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10, (1, 1), padding='valid'))

    model.add(GlobalAveragePooling2D())

    model.add(Activation('softmax'))

    model.summary()

    return model
model = pure_cnn_model()
checkpoint = ModelCheckpoint('Improved_Cnn.h5', monitor='val_loss', verbose=0, save_best_only= True,mode='auto')
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=LEARN_RATE),metrics = ['accuracy'])
model_details = model.fit(images_train, class_train,batch_size=128,epochs=NUM_EPOCH,validation_data=(images_test, class_test),callbacks=[checkpoint],verbose=1)
scores = model.evaluate(images_test, class_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")