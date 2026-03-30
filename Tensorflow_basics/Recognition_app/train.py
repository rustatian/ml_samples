import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf


def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def build_model(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(25, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(num_classes),
    ])
    return model


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=2000, minibatch_size=32, print_cost=True):

    n_x = X_train.shape[0]
    n_y = Y_train.shape[0]

    # Transpose to (samples, features) for Keras
    X_train_t = X_train.T.astype(np.float32)
    X_test_t = X_test.T.astype(np.float32)
    Y_train_t = Y_train.T.astype(np.int32).flatten()
    Y_test_t = Y_test.T.astype(np.int32).flatten()

    net = build_model(n_x, n_y)
    net.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    history = net.fit(
        X_train_t, Y_train_t,
        epochs=num_epochs,
        batch_size=minibatch_size,
        validation_data=(X_test_t, Y_test_t),
        verbose=0,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs:
                    print("Cost after epoch %i: %f" % (epoch, logs['loss']))
                    if print_cost and epoch % 100 == 0 else None
            ),
        ],
    )

    net.save_weights("model/recogn.weights.h5")

    # Plot the cost
    costs = history.history['loss'][::5]
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    print("Parameters have been trained!")

    _, train_acc = net.evaluate(X_train_t, Y_train_t, verbose=0)
    _, test_acc = net.evaluate(X_test_t, Y_test_t, verbose=0)
    print("Train Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)

    return net


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

# Normalize image vectors
X_train = X_train_flatten / 255.
X_test = X_test_flatten / 255.

# Convert training and test labels to one hot matrices
Y_train = Y_train_orig
Y_test = Y_test_orig

parameters = model(X_train, Y_train, X_test, Y_test)
