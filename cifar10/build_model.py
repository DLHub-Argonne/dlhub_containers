""""Tools used to construct the basic model

Modified from https://github.com/geifmany/cifar-vgg to save the model architecture along with
the weights, among other streamlining changes (e.g., using ImageDataGenerator to do normalization
rather than custom logic)

"""
from keras.layers import Conv2D, Activation, BatchNormalization, Dropout, \
    Dense, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical
from keras.models import Sequential
from keras.optimizers import SGD
from keras import regularizers
import pickle as pkl


def build_model_architecture(num_classes=10, weight_decay=0.0005, x_shape=(32, 32, 3)):
    """Build the network of vgg for 10 classes with massive dropout and
    weight decay as described in the paper.

    Args:
        num_classes (int): Number of output classes
        weight_decay (float): Weight decay rate
        x_shape (tuple of int): Shape of input images
    Returns:
        (Model) keras model
    """

    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=x_shape, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


if __name__ == "__main__":
    from keras.datasets import cifar10
    import os

    # Training parameters
    batch_size = 128
    maxepochs = 250

    learning_rate = 0.1
    lr_decay = 1e-6
    lr_drop = 20

    # Make the learning rate scheduler
    def lr_scheduler(epoch):
        return learning_rate * (0.5 ** (epoch // lr_drop))
    reduce_lr = LearningRateScheduler(lr_scheduler)
    
    # Load in the training data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Convert X's to floats and y's to (None, 10) arrays
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_test = to_categorical(y_test)
    y_train = to_categorical(y_train)

    # Create the optimizer
    sgd = SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)

    # Build and compile the model
    model = build_model_architecture()
    model.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    # Create a tool for data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(X_train)

    # Standardize the test set
    X_test = datagen.standardize(X_test)

    # Train it
    model.fit_generator(datagen.flow(X_train, y_train),
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        epochs=maxepochs,
                        validation_data=(X_test, y_test),
                        callbacks=[reduce_lr],
                        verbose=1)

    # Evaluate performance
    score = model.evaluate(X_test, y_test)
    print('Model accuracy:', score[1])

    # Save the model and the normalizer
    model.save(os.path.join('models', 'cifar10vgg.h5'))
    with open('models/img_normalizer.pkl', 'wb') as fp:
        pkl.dump(datagen, fp)
