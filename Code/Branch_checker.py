'''
FINAL
'''
import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle

session = tf.Session()
new_saver = tf.train.import_meta_graph("/home/josmilrom/Libraries/gml/Sessions/model.meta")
new_saver.restore(session,tf.train.latest_checkpoint('/home/josmilrom/Libraries/gml/Sessions'))
graph_inputs = tf.get_default_graph().get_tensor_by_name("single_input:0")
pretreated_data_tensor = tf.get_default_graph().get_tensor_by_name("pretreated_data:0")
single_vel_logits_tensor = tf.get_default_graph().get_tensor_by_name("single_vel_logits:0")
graph_outputs = tf.get_default_graph().get_tensor_by_name("vel_posttreated:0")

inputs_trans = np.transpose([[0.5],[0.2],[0.6],[0.9],[0.4],[0.1],[0.8],[0.5],[0.4]])
pretreated_data = session.run(pretreated_data_tensor, feed_dict={graph_inputs:inputs_trans})
single_vel_logits = session.run(single_vel_logits_tensor, feed_dict={graph_inputs:inputs_trans})
selected_velocity = session.run(graph_outputs, feed_dict={graph_inputs:inputs_trans})
# print(pretreated_data)
# print(single_vel_logits)
print(selected_velocity)

'''
Entrenamiento
'''

with open("/home/josmilrom/catkin_ws/src/pydag/Data_Storage/Simulations/gauss/orca3/type1_Nuav1_Nobs1/dataset_1/preprocessed.pickle", 'rb') as f:
            save = pickle.load(f)
            prepro_dict = save['prepro_dict']
whole_dataset = inputs_trans
zero_mean_dataset = whole_dataset - np.tile(prepro_dict["mean"][:whole_dataset.shape[1]],[whole_dataset.shape[0],1])
clean_dataset = zero_mean_dataset / np.tile(prepro_dict["max"][:whole_dataset.shape[1]],[whole_dataset.shape[0],1])

batch = np.tile(clean_dataset,[100,1])

# print(batch[0])

tf_train_dataset = tf.get_default_graph().get_tensor_by_name("tf_train_dataset:0")
logits = tf.get_default_graph().get_tensor_by_name("logits:0")
feed_dict = {tf_train_dataset : batch}
predictions = session.run(logits, feed_dict=feed_dict)

# print(predictions[0])

clean_dataset_back = predictions[0] * prepro_dict["max"][whole_dataset.shape[1]:]
zero_mean_dataset_back = clean_dataset_back + prepro_dict["mean"][whole_dataset.shape[1]:]

print(zero_mean_dataset_back)

