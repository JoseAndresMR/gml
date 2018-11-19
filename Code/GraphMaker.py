# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

class GraphMaker(object):
    def __init__(self, project = "gauss",world_type = 1,N_uav = 1,N_obs = 1,from_ROS = False):
        self.GettingWorldDefinition(project,world_type,N_uav,N_obs,from_ROS)

        self.first_folder_path = "/home/{4}/catkin_ws/src/pydag/Data_Storage/Simulations/{0}/{5}/type{1}_Nuav{2}_Nobs{3}".format(self.project,self.world_type,self.N_uav,self.N_obs,self.home_path,self.solver_algorithm)
        self.second_folder_path = self.first_folder_path + "/dataset_{}".format(self.n_dataset)

        self.LoadDatasetFromCSV()

        # batch_size = 128

        # self.FullyConnectedGAUSS(batch_size,4,[5,5,5,5])

    def FullyConnectedGAUSS(self,batch_size,hidden_layers,hidden_nodes):                
        graph = tf.Graph()
        with graph.as_default():
            # Input data. For the training data, we use a placeholder that will be fed
            # at run time with a training minibatch.
            num_inputs = self.train_dataset.shape[1]
            num_outputs = self.train_labels.shape[1]
            tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, num_inputs))
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_outputs))
            tf_valid_dataset = tf.constant(self.valid_dataset.astype(np.float32))
            tf_test_dataset = tf.constant(self.test_dataset.astype(np.float32))

            # Variables.
            weights = [tf.Variable(
                tf.truncated_normal([num_inputs, hidden_nodes[0]]))]
            biases = [tf.Variable(tf.zeros([hidden_nodes[0]]))]

            for n_layer in range(1,hidden_layers):
                weights.append(tf.Variable(tf.truncated_normal([hidden_nodes[n_layer-1], hidden_nodes[n_layer]])))
                biases.append(tf.Variable(tf.zeros([hidden_nodes[n_layer]])))

            weights.append(tf.Variable(tf.truncated_normal([hidden_nodes[-1], num_outputs])))
            biases.append(tf.Variable(tf.zeros([num_outputs])))

            # Training computation.
            def model(data):
                layer_output = tf.matmul(data, weights[0]) + biases[0]
                for n_layer in range(1,hidden_layers):
                    layer_output = tf.matmul(tf.nn.relu(layer_output), weights[n_layer]) + biases[n_layer]
                layer_output =  tf.matmul(layer_output, weights[-1]) + biases[-1]

                return layer_output
            
            logits = model(tf_train_dataset)
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

            # Optimizer.
            optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

            # Predictions for the training, validation, and test data.
            train_prediction = tf.nn.softmax(logits)
            valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
            test_prediction = tf.nn.softmax(model(tf_test_dataset))

            saver = tf.train.Saver()

        with tf.Session(graph=graph) as sess:
            tf.global_variables_initializer().run()
            # Save the variables to disk.
            writer = tf.summary.FileWriter("/tmp/log/...", sess.graph)
            save_path = saver.save(sess,"/home/josmilrom/Libraries/gml/model.ckpt")
            print("Model saved in path: %s" % save_path)

        return graph

    def LoadDatasetFromCSV(self):
        with open(self.second_folder_path + "/preprocessed.pickle", 'rb') as f:
            save = pickle.load(f)
            self.train_dataset = save['train_dataset']
            self.train_labels = save['train_labels']
            self.valid_dataset = save['valid_dataset']
            self.valid_labels = save['valid_labels']
            self.test_dataset = save['test_dataset']
            self.test_labels = save['test_labels']
            del save  # hint to help gc free up memory
            print('Training set', self.train_dataset.shape, self.train_labels.shape)
            print('Validation set', self.valid_dataset.shape, self.valid_labels.shape)
            print('Test set', self.test_dataset.shape, self.test_labels.shape)


    def GettingWorldDefinition(self, project = "gauss",world_type = 1,N_uav = 1,N_obs = 1,from_ROS = False):
        if from_ROS == True:
            self.world_definition = rospy.get_param('world_definition')
            self.project = self.world_definition['project']
            self.world_type = self.world_definition['type']
            self.N_uav = self.world_definition['N_uav']
            self.N_obs = self.world_definition['N_obs']
            self.obs_tube = self.world_definition['obs_tube']
            self.uav_models = self.world_definition['uav_models']
            self.n_dataset = self.world_definition['n_dataset']
            self.solver_algorithm = self.world_definition['solver_algorithm']
            self.obs_pose_list = self.world_definition['obs_pose_list']
            self.home_path = self.world_definition['home_path']
            self.depth_camera_use = self.world_definition['depth_camera_use']

        if from_ROS == False:
            self.project = project
            self.world_type = world_type
            self.N_uav = N_uav
            self.N_obs = N_obs
            self.obs_tube = []
            self.uav_models = ["typhoon_h480","typhoon_h480","typhoon_h480"]
            self.n_dataset = 1
            self.solver_algorithm = 'orca3'
            self.home_path = 'josmilrom'
            self.depth_camera_use = False

# FullyConnectedGAUSS = GraphMaker()