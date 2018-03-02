import os
import os.path as osp
from keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imshow, imresize
import tensorflow as tf


def init():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('model.h5')
    print("Model loaded from disk")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    keras_graph = tf.get_default_graph()

    return loaded_model, keras_graph


input_fld = '/'
weight_file = 'model.h5'
num_output = 1
write_graph_def_ascii_flag = True
prefix_output_node_names_of_final_network = 'output_node'
output_graph_name = 'model.pb'

output_fld = '<output/folder/name>'
if not os.path.isdir(output_fld):
    os.mkdir(output_fld)

K.set_learning_phase(0)
net_model, graph = init()

pred = [None]*num_output
pred_node_names = [None]*num_output
for i in range(num_output):
    pred_node_names[i] = prefix_output_node_names_of_final_network+str(i)
    pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
print('output nodes names are: ', pred_node_names)

sess = K.get_session()
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
graph_io.write_graph(constant_graph, output_fld, output_graph_name, as_text=False)
print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))

