import keras
import numpy as np
from keras.datasets import mnist
from keras import backend as K

####### CONSTANTS #######
model_path = "./model/mnist_cnn.hdf5"
batch_size = 128
num_classes = 10
# input image dimensions
img_rows, img_cols = 28, 28

def run(data):
    model = keras.models.load_model(model_path)

    if K.image_data_format() == 'channels_first':
        data= data.reshape(data.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        data = data.reshape(data.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    data = data.astype('float32')
    data /= 255

    pred = model.predict(data)

    return pred

def test_run():

    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # y_test = keras.utils.to_categorical(y_test, num_classes)
    x_test = np.load("./data/mnist_test.npy")
    y_test = np.load("./data/mnist_test_labels.npy")
    y_test = keras.utils.to_categorical(y_test, num_classes)

    output = run(x_test)

    pred_classes = [np.argmax(x) for x in output]
    y_classes = [np.argmax(y) for y in y_test]

    acc = np.mean([pred_classes[i] == y_classes[i] for i in range(len(pred_classes))])
    print("Prediction ran on {} samples with an accuracy of {}".format(len(pred_classes), acc))

    return output


if __name__ == '__main__':
    test_run()
