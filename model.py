import csv
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

samples = []

# Open the csv file and retrieve file names
with open('./mydata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# This line was needed for the default mydata
# included in the project in order to remove the header.
# On the data created with the simulation later it was needed as the
# header is not appended by default.
#samples.pop(0)

# Split the train and validation samples to 75% and 25%
train_samples, validation_samples = train_test_split(samples, test_size=0.25)

# Check the split
print(len(train_samples), len(validation_samples))

def generator(samples, batch_size=128, train=False):
    """
    Generator made to continuously feed on keras layers a min amount
    of preprocessed samples. If train is True the generator takes care of
    data augmentation also.
    """
    num_samples = len(samples)
    # Check if the generator works on training dataset.
    if train:
        # take into account also right and left image and applied correction.
        img_per_row = 3
        correction = 0.2
    else:
        # On validation set we need only center image to assess the data.
        img_per_row = 1
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(img_per_row):
                    # Read the images from the folder
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = os.path.join('./mydata/IMG', filename)
                    image = mpimg.imread(current_path)
                    # Apply change of colorspace as preprocessing. The same
                    # is applied on drive.py before feeding the image to the
                    # model.
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                    images.append(image)
                    # In case of training we get all images (i = 0,1,2)
                    # In case of validation we get only center image (i = 0)
                    if i == 0:
                        measurement = float(batch_sample[3])
                    elif i == 1:
                        measurement = float(batch_sample[3]) + correction
                    elif i == 2:
                        measurement = float(batch_sample[3]) - correction
                    measurements.append(measurement)

                    # If we are in training dataset we also augment the dataset
                    # by flipping the images and applying negative measurement.
                    # That way during collecting the data we avoided driving
                    # backwards as this is simulated by such data augmentation
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

# Prepare the train and validation generator to feed the model.
train_generator = generator(train_samples, batch_size=49, train=True)
validation_generator = generator(validation_samples, batch_size=49)

# Import keras modules and helper functions.
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D
from keras.layers import Cropping2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from test import get_activations, compare_images



# Similar to NVIDIA model
model = Sequential()
# Cropping and resizing the images to reduce processing
model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: K.tf.image.resize_images(x, size=(66,200))))
# Normalizing the image.
model.add(Lambda(lambda x: x / 127.5 - 1.0))
# Apply convolutions with RELU activations.
model.add(Convolution2D(24,5,5,subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
# Add dropout layer to avoid overfitting.
model.add(Dropout(0.4))
# Dense Layers with RELU activations
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1))

# Create checkpoint of the best model regarding validation loss
checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_loss',
                             verbose=0, save_best_only=True, mode='auto')

# Loss function is MSE and optimizer is ADAM
model.compile(loss='mse', optimizer='adam')


# Create the fit generator. Length of train samples is *6 because we have for each
# center image 2 additional (1 right and 1 left) and all of them are flipped
# as a data augmentation technique. So for each sample read we actually evaluate
# 6 samples.
history_obj = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6,
 validation_data=validation_generator, nb_val_samples=len(validation_samples),
 callbacks=[checkpoint], nb_epoch=7, verbose=1)

# Print training and validation loss
print(history_obj.history.keys())

# Plot the graph of training and validation loss across the epochs.
plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('Model MSE Loss')
plt.ylabel('MSE Loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation_set'], loc='upper right')
plt.show()

# Save the model and exit.
model.save('model.h5')
exit()
