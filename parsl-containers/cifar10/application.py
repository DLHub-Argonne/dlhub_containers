import os
import keras
import numpy as np
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

####### CONSTANTS #######
normalize=True
batch_size=50
model_path = "./model/cifar10vgg.h5"
class_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def run(data):

    if isinstance(data, list):
        data = np.array(data)
    model = build_model_architecture()
    model.load_weights(model_path)

    if normalize:
        data = normalize_production(data)
    res = model.predict(data,batch_size)

    output = []
    for i in range(len(res)): # Prediction for each image
        classes_sorted = np.argsort(res[i])
        im_pred = [{class_labels[j]:float(res[i][j])} for j in classes_sorted[::-1]]
        output.append(im_pred)

    return output

def test_run():
    data_path = "./data/x_test.npy"
    label_path = "./data/y_test.npy"
    data = np.load(data_path)
    labels = np.load(label_path)

    data = data[:2].tolist()

    output = run(data)
    #assert len(output) == len(labels) # 1 prediction for each image

    for i in range(2):
        print("Image {}".format(i))
        print("Predicted class: {}, True class: {}".format(output[i][0], class_labels[labels[i][0]]))
        print("Prediction: {}\n".format(output[i]))
    acc = np.mean([list(output[i][0].keys())[0] == class_labels[labels[i][0]] for i in range(len(output))])
    print("Predicted on {} images with an accuracy of {}".format(len(output), acc))
    #acc = np.mean([output[i] == labels[i] for i in range(len(output))])
    #print("Predicted on {} images with an accuracy of {} ".format(len(output), acc))

    return output

def normalize_production(x):
    #this function is used to normalize instances in production according to saved training set statistics
    # Input: X - a training set
    # Output X - a normalized training set according to normalization constants.

    #these values produced during first training and are general for the standard cifar10 training set normalization
    mean = 120.707
    std = 64.15
    return (x-mean)/(std+1e-7)

def build_model_architecture():
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
    num_classes = 10
    weight_decay = 0.0005
    x_shape = [32,32,3]

    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model

if __name__ == '__main__':
    test_run()
