import numpy as np
import cv2
import tensorflow as tf


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(25, activation='relu', input_shape=(12288,)),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(6),
    ])
    model.load_weights("model/recogn.weights.h5")
    return model


model = build_model()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    my_image = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    image = my_image.reshape((1, 64 * 64 * 3)).astype(np.float32)

    prediction = np.argmax(model.predict(image, verbose=0), axis=1)
    print("Algorithm predicts: y = " + str(np.squeeze(prediction)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
