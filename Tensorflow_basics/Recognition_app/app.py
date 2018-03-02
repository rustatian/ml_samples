import tensorflow as tf
import numpy as np
import cv2


tf.reset_default_graph()

W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer(), dtype=tf.float32)
W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer(), dtype=tf.float32)
W3 = tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer(), dtype=tf.float32)
X = tf.placeholder("float", [12288, 1])

saver = tf.train.Saver()
cap = cv2.VideoCapture(0)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, "model/recogn.ckpt")
    while True:
        ret, frame = cap.read()

        my_image = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA).reshape((1, 64 * 64 * 3)).T

        image = my_image.astype(np.float32)

        Z1 = tf.add(tf.matmul(W1, image), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        pred = tf.argmax(Z3)

        prediction = sess.run(pred, feed_dict={X: image})
        print("Algorithm predicts: y = " + str(np.squeeze(prediction)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
