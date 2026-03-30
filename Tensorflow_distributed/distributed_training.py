"""Distributed training example using tf.distribute.MirroredStrategy.

MirroredStrategy replicates the model across available GPUs and synchronizes
gradients at each step. This replaces the old parameter server / worker pattern.
"""

import tensorflow as tf


def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10),
    ])


def main():
    # MirroredStrategy distributes across all visible GPUs
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # Load MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

    # Build and compile model inside strategy scope
    with strategy.scope():
        model = build_model()
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

    # Batch size scales with number of replicas
    batch_size = 64 * strategy.num_replicas_in_sync

    model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
    )

    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print("Test accuracy: {:.4f}".format(test_acc))


if __name__ == '__main__':
    main()
