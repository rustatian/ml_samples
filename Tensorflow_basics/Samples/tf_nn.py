import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


def get_normalized_data():
    print("Reading in and transforming data...")
    df = pd.read_csv('train.csv')
    data = df.to_numpy().astype(np.float32)
    np.random.shuffle(data)
    X = data[:, 1:]
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std, std == 0, 1)
    X = (X - mu) / std
    Y = data[:, 0].astype(np.int32)
    return X, Y


def main():
    X, Y = get_normalized_data()

    Xtrain, Ytrain = X[:-1000], Y[:-1000]
    Xtest, Ytest = X[-1000:], Y[-1000:]

    M = 300
    K = 10

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(M, activation='relu', input_shape=(Xtrain.shape[1],)),
        tf.keras.layers.Dense(K),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.00004),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    history = model.fit(
        Xtrain, Ytrain,
        epochs=20,
        batch_size=500,
        validation_data=(Xtest, Ytest),
        verbose=1,
    )

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()


if __name__ == '__main__':
    main()
