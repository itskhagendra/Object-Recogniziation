import matplotlib
from keras.callbacks import ModelCheckpoint
from keras.layers import GlobalAveragePooling2D, Conv2D, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from helper import get_class_names, get_train_data, get_test_data
from helper import predict_classes

matplotlib.style.use('ggplot')
class_names = get_class_names()
num_classes = len(class_names)
# Hight and width of the images
IMAGE_SIZE = 32
# 3 channels, Red, Green and Blue
CHANNELS = 3
# Number of epochs
NUM_EPOCH = 100
# learning rate
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


datagen = ImageDataGenerator(
    featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,
    samplewise_std_normalization=False, zca_whitening=False, rotation_range=45, width_shift_range=0.2,
    height_shift_range=0.2, horizontal_flip=True, vertical_flip=False)
print('Data Augmentation Performed Successfully')
datagen.fit(images_train)

augmented_model = pure_cnn_model()

augmented_checkpoint = ModelCheckpoint('augmented_best_model.h5', monitor='val_loss', verbose=0, save_best_only=True,
                                       mode='auto')

augmented_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LEARN_RATE), metrics=['accuracy'])

augmented_model_details = augmented_model.fit_generator(datagen.flow(images_train, class_train, batch_size=32),
                                                        steps_per_epoch=len(images_train) / 32, epochs=NUM_EPOCH,
                                                        validation_data=(images_test, class_test),
                                                        callbacks=[augmented_checkpoint], verbose=1)

correct, labels_pred = predict_classes(augmented_model, images_test, labels_test)

num_images = len(correct)
print("Accuracy: %.2f%%" % ((sum(correct) * 100) / num_images))

model_json = augmented_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
augmented_model.save_weights("model.h5")
print("Saved model to disk")
