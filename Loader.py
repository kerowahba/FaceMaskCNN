import os
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from mtcnn import MTCNN

# Maximize the usage of the CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

trainImages = os.path.join("./MedicalMask/trainImages")
testImages = os.path.join("./MedicalMask/testImages")

train = pd.read_csv(os.path.join("./train.csv"))
submission = pd.read_csv(os.path.join("./submission.csv"))

trainFolder = os.listdir(trainImages)
trainFolder.sort()

testFolder = os.listdir(testImages)
testFolder.sort()

train_images = trainFolder[1:]
test_images = testFolder[1:]

print("The Dataset has %d faces, that includes the 4 required classes "
      "and many more others that will be disregarded, distributed among %d images \n"
      % ((len(train) + len(submission)), len(trainFolder)))

print("The Training Dataset has %d faces, distributed among %d images\n"
      % (len(train), len(train_images)))

print("The Test Dataset has %d faces, distributed among %d images\n"
      % (len(submission), len(test_images)))

# NOTE: test_images ranges from [0, 600]
# NOTE: starts from Image name 0001 --> 0633
img = plot.imread(os.path.join(testImages, test_images[3]))
plot.imshow(img)
plot.show()

# NOTE: train_images ranges from [0, 1479]
# NOTE: starts from Image name 1800 --> 3400
img = plot.imread(os.path.join(trainImages, train_images[10]))
plot.imshow(img)
plot.show()

# Filtering the images in the train dataset to have only the images with the 4 categories
options = ['face_no_mask', 'mask_colorful', 'mask_surgical', 'K/N95_mask']
train = train[train['classname'].isin(options)]
train.sort_values('name', inplace=True)

print("----- After eliminating the non-mask related images -----:")
print("The Training Dataset has %d faces, that includes ONLY the 4 required classes, "
      "distributed among %d images\n"
      % (len(train), len(train_images)))

# Data Pre-processing: importing the mask boxes coordinates from the train.csv file
bbox = []
for i in range(len(train)):
    arr = []
    for j in train.iloc[i][["x1", 'x2', 'y1', 'y2']]:
        arr.append(j)
    bbox.append(arr)
train["bbox"] = bbox


# Function that takes the image ID , and return the boxes coordinates.
def get_boxes(id):
    boxes = []
    for i in train[train["name"] == str(id)]["bbox"]:
        boxes.append(i)
    return boxes


# Printing image with position 10 among the training images set.
image = train_images[10]
img = plot.imread(os.path.join(trainImages, image))

# Extracting the axis from the plotted image and append them on current picture.
_, axis = plot.subplots(1)
axis.imshow(img)
boxes = get_boxes(image)
for box in boxes:
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r',
                             facecolor='none')
    axis.add_patch(rect)
plot.show()

# Plotting the chart with the numbers of each class.
plot.bar(['face_no_mask', 'mask_colorful', 'mask_surgical', 'K/N95_mask'], train.classname.value_counts())
plot.show()

img_size = 50
data = []
path = './MedicalMask/images/'


# Converting the images from the training set into Matrix to be able to run the model on and the filters also later on
# NOTES: the index 0 in tempData indicated the image itself,
# 1,2,3,4 => the coordinates of the box,
# 5 => the output image that will be appended to the data that will be used in the Model
def create_data():
    for i in range(len(train)):
        tempData = []
        for j in train.iloc[i]:
            tempData.append(j)
        img_array = cv2.imread(os.path.join(trainImages, tempData[0]), cv2.IMREAD_GRAYSCALE)
        crop_image = img_array[tempData[2]:tempData[4], tempData[1]:tempData[3]]
        new_img_array = cv2.resize(crop_image, (img_size, img_size))
        data.append([new_img_array, tempData[5]])


create_data()
print("A Post-Processed Dataset has been created of image size (50x50) and will be used in the Model creation, \n"
      "This training dataset includes %d faces" % len(data))

# An Example of the cropped image that will be used to learn the model.
plot.imshow(data[100][0])
plot.show()

# Pre-Processing by doing the Transform Step
x = []
y = []
for features, labels in data:
    x.append(features)
    y.append(labels)

from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
y = lbl.fit_transform(y)

# Data Augmentation and Normalization using Keras from TensorFlow
x = np.array(x).reshape(-1, 50, 50, 1)
x = tf.keras.utils.normalize(x, axis=1)
from keras.utils.np_utils import to_categorical

y = to_categorical(y)

# --------------- MODEL CREATION ----------------

numEpochs = 1
print("\n\nThe Model is being created now, and will run %d Epoch/es." % numEpochs)
model = Sequential()
model.add(Conv2D(100, (3, 3), input_shape=x.shape[1:], activation='relu', strides=2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(4, activation='softmax'))

opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-5)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs=numEpochs, batch_size=5)

# Testing for Mask Detection in image 3 in the Test dataset
detector = MTCNN()
img = plot.imread(os.path.join(testImages, test_images[3]))
face = detector.detect_faces(img)
for face in face:
    bounding_box = face['box']
    x = cv2.rectangle(img,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0, 155, 255),
                      10)
    plot.imshow(x)
    plot.show()
