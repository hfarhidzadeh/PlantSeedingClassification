# Referred https://www.pyimagesearch.com/2017/NUM_CLASSES/11/image-classification-with-keras-and-deep-learning/
# https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#import tensorflow as tf
##from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
import sys
import pandas as pd
sys.path.append('C://Program Files//Python36//Lib//site-packages')
import cv2
from keras.models import model_from_json

from keras.applications.vgg16 import VGG16

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os
import sys

from keras.utils import to_categorical
import matplotlib

# kaggle/python docker image: https://github.com/kaggle/docker-python
# Input data files are available in the "../input/" directory.
from subprocess import check_output
#list the files in the input directory
#print(check_output(["ls", "../input"]).decode("utf8"))
#print(check_output(["pwd",""]).decode("utf8")) # returns /kaggle/working
#classes = check_output(["ls", "../input/train"]).decode("utf8") # returns 12 directories
#print((classes))


from keras.utils import multi_gpu_model
#import tensorflow as tf
from keras.applications import Xception

def classes_to_int(label):
    # label = classes.index(dir)
    label = label.strip()
    if label == "Black-grass":  return 0
    if label == "Charlock":  return 1
    if label == "Cleavers":  return 2
    if label == "Common Chickweed":  return 3
    if label == "Common wheat":  return 4
    if label == "Fat Hen":  return 5
    if label == "Loose Silky-bent": return 6
    if label == "Maize":  return 7
    if label == "Scentless Mayweed": return 8
    if label == "Shepherds Purse": return 9
    if label == "Small-flowered Cranesbill": return 10
    if label == "Sugar beet": return 11
    print("Invalid Label", label)
    return 12

def int_to_classes(i):
    if i == 0: return "Black-grass"
    elif i == 1: return "Charlock"
    elif i == 2: return "Cleavers"
    elif i == 3: return "Common Chickweed"
    elif i == 4: return "Common wheat"
    elif i == 5: return "Fat Hen"
    elif i == 6: return "Loose Silky-bent"
    elif i == 7: return "Maize"
    elif i == 8: return "Scentless Mayweed"
    elif i == 9: return "Shepherds Purse"
    elif i == 10: return "Small-flowered Cranesbill"
    elif i == 11: return "Sugar beet"
    print("Invalid class ", i)
    return "Invalid Class"

def readTrainData(trainDir):
    data = []
    labels = []
    # loop over the input images
    dirs = os.listdir(trainDir) 
    for dir in dirs:
        absDirPath = os.path.join(os.path.sep,trainDir, dir)
        images = os.listdir(absDirPath)
        for imageFileName in images:
            # load the image, pre-process it, and store it in the data list
            imageFullPath = os.path.join(trainDir, dir, imageFileName)
            #print(imageFullPath)
            img = load_img(imageFullPath)
            arr = img_to_array(img)  # Numpy array with shape (233,233,3)
            arr = cv2.resize(arr, (HEIGHT,WIDTH)) #Numpy array with shape (HEIGHT, WIDTH,3)
            #print(arr.shape) 
            data.append(arr)
            label = classes_to_int(dir)
            labels.append(label)
    return data, labels

#The Plant Seedlings Dataset contains images of approximately 960 unique plants belonging to
# 12 species at several growth stages.
# It comprises annotated RGB images with a physical resolution of roughly 10 pixels per mm.
NUM_CLASSES = 12
# we need images of same size so we convert them into the size
WIDTH = 128
HEIGHT = 128
DEPTH = 3
inputShape = (WIDTH, HEIGHT, DEPTH)
# initialize number of epochs to train for, initial learning rate and batch size
EPOCHS = 20
INIT_LR = 1e-3
BS = 32



def createModel():
    model = Sequential()
    # first set of CONV => RELU => POOL layers
    # The CONV  layer will learn 20 convolution filters, each of which are 5×5.
    
    model.add(Conv2D(25, (5, 5), padding="same", input_shape=inputShape))
    
    # We then apply a ReLU activation function followed by 2×2 max-pooling in both 
    # the x and y direction with a stride of two. 
    #To visualize this operation, consider a sliding window that “slides” across 
    #the activation volume, taking the max operation over each region, while taking 
    #a step of two pixels in both the horizontal and vertical direction.
    
    #model.add(Activation("relu"))
    model.add(LeakyReLU(alpha=.001))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # second set of CONV => RELU => POOL layers
    #This time we are learning 50 convolutional filters rather than the 20 convolutional
    #filters as in the previous layer set. It’s common to see the number of CONV 
    #filters learned increase the deeper we go in the network architecture.
    
    model.add(Conv2D(50, (5, 5), padding="same"))
    #model.add(Activation("relu"))
    model.add(LeakyReLU(alpha=.001))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    
    # first (and only) set of FC => RELU layers
    # Flattening out the volume into a set of fully-connected layers
    # Take the output of the preceding MaxPooling2D layer and flatten it into a single vector.
    # This operation allows us to apply our dense/fully-connected layers.
    # Fully-connected layer contains 500 nodes which is passed through another 
    # nonlinear ReLU activation.
    
    model.add(Flatten())
    model.add(Dense(300))
    #model.add(Activation("relu"))
    model.add(LeakyReLU(alpha=.001))
    
    # softmax classifier
    # Another fully-connected layer, but this one is special — the number of nodes is equal 
    # to the number of classes  (i.e., the classes we want to recognize).
    # This Dense layer is then fed into our softmax classifier
    # which will yield the probability for each class.
    model.add(Dense(output_dim=12))
    model.add(Activation("softmax"))
    # returns our fully constructed deep learning + Keras image classifier 
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    # use binary_crossentropy if there are two classes
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

random.seed(10)
allLabels =  os.listdir("E://Kaggle//train//train")  # list of subdirectories and files
print("Loading images...")
sys.stdout.flush()
X, Y = readTrainData("E://Kaggle//train//train")
# scale the raw pixel intensities to the range [0, 1]
X = np.array(X, dtype="float") / 255.0
Y = np.array(Y)
# convert the labels from integers to vectors
Y =  to_categorical(Y, num_classes=12)


print("Parttition data into 75:25...")
sys.stdout.flush()
# partition the data into training and testing splits using 75% training and 25% for validation
(trainX, valX, trainY, valY) = train_test_split(X,Y,test_size=0.25, random_state=10)

#construct the image generator for data augmentation
print("Generating images...")
sys.stdout.flush()
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, \
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,\
    horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("compiling model...")
sys.stdout.flush()
#model = createModel()

''' with tf.device('/cpu:0'):
    model = Xception(weights=None,
                     input_shape=(HEIGHT, WIDTH, 3),
                     classes=NUM_CLASSES)

parallel_model = multi_gpu_model(model, gpus=1)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop') '''
# train the network
print("training network...")
sys.stdout.flush()

base_model = VGG16(weights='imagenet', include_top=False, input_shape = inputShape)

x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')



H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), \
# #H = parallel_model.fit(aug.flow(trainX, trainY, batch_size=BS), \
    validation_data=(valX, valY), \
    steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, verbose=1)


#model = DenseNet121(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=12)

# save the model to disk
print("Saving model to disk")
sys.stdout.flush()
model.save('E://Kaggle//my_model.h5')

''' # set the matplotlib backend so figures can be saved in the background
# plot the training loss and accuracy
print("Generating plots...")
sys.stdout.flush()
matplotlib.use("Agg")
matplotlib.pyplot.style.use("ggplot")
matplotlib.pyplot.figure()
N = EPOCHS
matplotlib.pyplot.plot(np.arange(0, N), H.history["loss"], label="train_loss")
matplotlib.pyplot.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
matplotlib.pyplot.plot(np.arange(0, N), H.history["acc"], label="train_acc")
matplotlib.pyplot.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
matplotlib.pyplot.title("Training Loss and Accuracy on  crop classification")
matplotlib.pyplot.xlabel("Epoch #")
matplotlib.pyplot.ylabel("Loss/Accuracy")
matplotlib.pyplot.legend(loc="lower left")
matplotlib.pyplot.savefig("plot.png") '''

def readTestData(testDir):
    data = []
    filenames = []
    # loop over the input images
    images = os.listdir(testDir)
    for imageFileName in images:
        # load the image, pre-process it, and store it in the data list
        imageFullPath = os.path.join(testDir, imageFileName)
        #print(imageFullPath)
        img = load_img(imageFullPath)
        arr = img_to_array(img)  # Numpy array with shape (...,..,3)
        arr = cv2.resize(arr, (HEIGHT,WIDTH)) 
        data.append(arr)
        filenames.append(imageFileName)
    return data, filenames


# read test data and find its classification
testX, filenames = readTestData("E://Kaggle//test//test")
# scale the raw pixel intensities to the range [0, 1]
testX = np.array(testX, dtype="float") / 255.0

from keras.models import load_model
mymodel = load_model('E://Kaggle//my_model.h5')

# mymodel = model_from_json(loaded_model_json)
# # load weights into new model
# mymodel.load_weights(mymodel)
# mymodel.compile('sgd','mse')


yFit = mymodel.predict(testX, batch_size=10, verbose=1) 


import csv  
with open('E://Kaggle//output.csv', 'w', newline='') as csvfile:
    fieldnames = ['file', 'species']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for index, file in enumerate(filenames):
        classesProbs = yFit[index]
        maxIdx = 0
        maxProb = 0
        for idx in range(0,11):
            if(classesProbs[idx] > maxProb):
                maxIdx = idx
                maxProb = classesProbs[idx]
        writer.writerow({'file': file, 'species': int_to_classes(maxIdx)})
print("Writing complete")