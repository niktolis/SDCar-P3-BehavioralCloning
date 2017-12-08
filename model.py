import csv
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

samples = []

with open('./mydataTrack1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#Remove Header
#samples.pop(0)

train_samples, validation_samples = train_test_split(samples, test_size=0.25)

print(len(train_samples), len(validation_samples))

def generator(samples, batch_size=128, train=False):
    num_samples = len(samples)
    if train:
        img_per_row = 3
        correction = 0.2
    else:
        img_per_row = 1
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(img_per_row):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = os.path.join('./mydataTrack1/IMG', filename)
                    image = mpimg.imread(current_path)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                    images.append(image)
                    if i == 0:
                        measurement = float(batch_sample[3])
                    elif i == 1:
                        measurement = float(batch_sample[3]) + correction
                    elif i == 2:
                        measurement = float(batch_sample[3]) - correction
                    measurements.append(measurement)
                    if train:
                        augmented_images, augmented_measurements = [], []
                        for image, measurement in zip(images, measurements):
                            augmented_images.append(image)
                            augmented_measurements.append(measurement)
                            augmented_images.append(cv2.flip(image,1))
                            augmented_measurements.append(measurement*-1.0)

            if train:
                X_train = np.array(augmented_images)
                y_train = np.array(augmented_measurements)
            else:
                X_train = np.array(images)
                y_train = np.array(measurements)

            yield shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=49, train=True)
validation_generator = generator(validation_samples, batch_size=49)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D
from keras.layers import Cropping2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from test import get_activations, compare_images

model = Sequential()
model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: K.tf.image.resize_images(x, size=(66,200))))
model.add(Lambda(lambda x: x / 127.5 - 1.0))
model.add(Convolution2D(24,5,5,subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1))


checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_loss',
                             verbose=0, save_best_only=True, mode='auto')

model.compile(loss='mse', optimizer='adam')
history_obj = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6,
 validation_data=validation_generator, nb_val_samples=len(validation_samples),
 callbacks=[checkpoint], nb_epoch=7, verbose=1)





print(history_obj.history.keys())

plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('Model MSE Loss')
plt.ylabel('MSE Loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation_set'], loc='upper right')
plt.show()


model.save('model.h5')
exit()
