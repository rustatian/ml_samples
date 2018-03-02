import tensorflow as tf

cluster = tf.train.ClusterSpec({
    "ps": [
        "ps0.localhost:2223"
    ]})

server = tf.train.Server(cluster, job_name="ps", task_index=0)
server.join()
