"""
Python source code - replace this with a description of the code and write the code below this text.
"""
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

#
# Ronald Karunia
#
import csv
import cv2
import numpy as np
from glob import glob
import sklearn

samples = []
images = []
angles = []
csvpath = 'data/driving_log.csv'
# Read the driving data and images
with open (csvpath) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
count = 0
for line in samples:
    angle = float(line[3])
    keep = 1
    # Discard 75% of data with low steering angle
    if (-0.1 <= angle <= 0.0):
        if (count < 4):
            keep = 0
            count += 1
        else:
            keep = 1
            count = 0
    if (keep == 1): 
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('\\')[-1]
            filepath = 'data/IMG/' + filename
            angle = float(line[3])
            # Left and right steering angle correction
            if (i == 1):
                angle = angle + 0.2
            elif (i == 2):
                angle = angle - 0.2
            angles.append(angle)
            image = cv2.imread(filepath)
            image_flipped = cv2.flip(image,1)
            images.append(image)
            images.append(image_flipped)
            angles.append(angle*-1.0)

X_train = np.array(images)
y_train = np.array(angles)
row, col, ch = 160, 320, 3 

# Build the learning layers
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(48,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

# Train the network
# Save the model on every epoch
checkpointPath = "wi-{epoch:02d}-{val_loss:.2f}.h5"
checkpoint = ModelCheckpoint(checkpointPath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
cb_list = [checkpoint]
model.compile(loss='mse', optimizer='adam')
history_obj = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10, callbacks=cb_list)
# Save the loss history for plotting later
loss_history = history_obj.history['loss']
val_loss_history = history_obj.history['val_loss']
np_loss_history = np.array(loss_history)
np_val_loss_history = np.array(val_loss_history)
np.savetxt("loss_history.txt", np_loss_history, delimiter=",")
np.savetxt("val_loss_history.txt", np_val_loss_history, delimiter=",")
# Save the model from the last epoch
model.save('model.NVDA.h5')
