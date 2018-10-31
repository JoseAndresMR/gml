# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
# from tensorflow.python.saved_model import tag_constants
from six.moves import cPickle as pickle

class GraphMakerTrainer(object):
    def __init__(self, role = "path",world_type = 1,N_uav = 1,N_obs = 1,from_ROS = False):
        self.GettingWorldDefinition(role,world_type,N_uav,N_obs,from_ROS)

        self.gml_folder_path = "/home/{4}/Libraries/gml/Sessions/{0}/type{1}_Nuav{2}_Nobs{3}".format(self.role,self.world_type,self.N_uav,self.N_obs,self.home_path)

        self.LoadDatasetFromCSV()

    def FullyConnected(self,batch_size,fc_hidden_nodes,num_steps,fc_learning_rate,conv_hidden_nodes,conv_learning_rate,fvi):                
        self.fc_hidden_nodes = fc_hidden_nodes
        self.fc_learning_rate = fc_learning_rate
        self.fvi = fvi
        graph = tf.Graph()
        with graph.as_default():
            # Input data. For the training data, we use a placeholder that will be fed
            # at run time with a training minibatch.
            num_inputs = self.train_dataset.shape[1]
            num_outputs = self.train_labels.shape[1]
            tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, num_inputs), name="tf_train_dataset")
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_outputs), name="tf_train_labels")
            tf_valid_dataset = tf.constant(self.valid_dataset.astype(np.float32))
            tf_test_dataset = tf.constant(self.test_dataset.astype(np.float32))

            prepro_mean = tf.constant(self.prepro_dict["mean"].astype(np.float32))
            prepro_max = tf.constant(self.prepro_dict["max"].astype(np.float32))

            single_input = tf.placeholder(tf.float32,shape=(1, num_inputs), name="single_input")

            self.fc_weights = []
            self.fc_biases = []

            def fully_connected(data, name = "fully_connected"):
                with tf.name_scope(name):
                    
                    fc_layer_outputs = []

                    nodes = [num_inputs]
                    for i in range(len(self.fc_hidden_nodes)):
                        nodes.append(self.fc_hidden_nodes[i])
                    nodes.append(num_outputs)

                    for n_layer in range(0,len(nodes)-1):
                        if len(self.fc_weights) < len(nodes):

                            self.fc_weights.append(tf.Variable(tf.truncated_normal([nodes[n_layer], nodes[n_layer+1]]),name = "cd_w_{}".format(n_layer+1)))
                            self.fc_biases.append(tf.Variable(tf.zeros([nodes[n_layer+1]]),name = "cd_b_{}".format(n_layer+1)))

                        if fc_layer_outputs == []:
                            fc_layer_outputs = [tf.nn.tanh(tf.matmul(data, self.fc_weights[n_layer]) + self.fc_biases[n_layer])]
                        else:
                            fc_layer_outputs.append(tf.nn.tanh(tf.matmul(fc_layer_outputs[n_layer-1], self.fc_weights[n_layer]) + self.fc_biases[n_layer]))

                        tf.summary.histogram("fc_weights",self.fc_weights[n_layer])
                        tf.summary.histogram("fc_biases",self.fc_biases[n_layer])
                        # tf.summary.histogram("activations",fc_layer_outputs[n_layer-1])

                    return fc_layer_outputs, self.fc_weights, self.fc_biases

            def convolutional(data, name = "convolutional"):

                conv_layer_outputs = []

                conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
                hidden = tf.nn.relu(conv + layer1_biases)
                pool = tf.nn.max_pool(hidden,[1, 2, 2, 1],[1, 2, 2, 1], padding='SAME')
                conv = tf.nn.conv2d(pool, layer2_weights, [1, 1, 1, 1], padding='SAME')
                hidden = tf.nn.relu(conv + layer2_biases)
                pool = tf.nn.max_pool(hidden,[1, 2, 2, 1],[1, 2, 2, 1], padding='SAME')
                reshape = flatten(pool)
                hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
                return tf.matmul(hidden, layer4_weights) + layer4_biases

            fc_layer_outputs, fc_weights, fc_biases = fully_connected(tf_train_dataset)

            logits = tf.add(fc_layer_outputs[-1], 0, name = "logits")
                        
            tf.summary.histogram("logits",logits)
            
            def get_single_vel(data, name = "get_single_vel"):
                with tf.name_scope(name):
                    pretreated_data = tf.div(data - prepro_mean[:num_inputs], prepro_max[:num_inputs],name = 'pretreated_data') 
                    # single_vel_logits = fully_connected(pretreated_data)
                    single_logits, fc_weights, fc_biases = fully_connected(pretreated_data)
                    # single_vel_logits = tf.add(logits[-1], 0, name = "single_vel_logits")
                    posttreated = tf.add(tf.multiply(single_logits[-1],prepro_max[num_inputs:]), prepro_mean[num_inputs:])
                    
                    return posttreated

            single_logits = tf.add(get_single_vel(single_input), 0, name = "vel_posttreated")

            # Regularization
            with tf.name_scope("regularization"):
                beta = 5e-4 #tf.Variable([5e-6])
                loss_regu = 0
                for n_layer in range(1,len(fc_hidden_nodes)):
                    loss_regu += beta*tf.nn.l2_loss(fc_weights[n_layer])

            # Loss
            with tf.name_scope("loss"):
                loss = tf.losses.mean_squared_error(labels=tf_train_labels, predictions =logits) + loss_regu
                tf.summary.scalar("loss",loss)
            # Optimizer
            with tf.name_scope("train"):
                optimizer = tf.train.GradientDescentOptimizer(0.006).minimize(loss)

            # Predictions for the training, validation, and test data.
            with tf.name_scope("predictions"):
                train_prediction = logits
                
                valid_fc_layer_outputs, valid_fc_weights, valid_fc_biases = fully_connected(tf_valid_dataset)
                valid_prediction = valid_fc_layer_outputs[-1]

                test_fc_layer_outputs, test_fc_weights, test_fc_biases = fully_connected(tf_test_dataset)
                test_prediction = test_fc_layer_outputs[-1]
                # print(train_prediction.shape,tf_train_labels.shape)
                # train_accuracy = self.accuracy(train_prediction,tf_train_labels)
                # tf.summary.histogram("train_accuracy",train_accuracy)

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print("Initialized")
            saver = tf.train.Saver()

            merged_summary = tf.summary.merge_all()
            hparam_str = self.make_hparam_string(self.fc_learning_rate,self.fc_hidden_nodes,self.fvi)
            print(hparam_str)
            writer = tf.summary.FileWriter("/home/josmilrom/Libraries/gml/Sessions/{0}/type{1}_Nuav{2}_Nobs{3}/log/".format(self.role,self.world_type,self.N_uav,self.N_obs,self.home_path) + hparam_str, session.graph)
            save_path = saver.save(session,"/home/josmilrom/Libraries/gml/Sessions/{0}/type{1}_Nuav{2}_Nobs{3}/model".format(self.role,self.world_type,self.N_uav,self.N_obs,self.home_path))
            print("Model saved in path: %s" % save_path)

            for step in range(num_steps):
                # Pick an offset within the training data, which has been randomized.
                # Note: we could use better randomization across epochs.
                offset = (step * batch_size) % (self.train_labels.shape[0] - batch_size)
                # Generate a minibatch.
                batch_data = self.train_dataset[offset:(offset + batch_size), :]
                batch_labels = self.train_labels[offset:(offset + batch_size), :]
                # print(batch_data)
                # print(batch_labels)
                # Prepare a dictionary telling the session where to feed the minibatch.
                # The key of the dictionary is the placeholder node of the graph to be fed,
                # and the value is the numpy array to feed to it.
                feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
                if step % 5 == 0:
                    s = session.run(merged_summary, feed_dict = feed_dict)
                    writer.add_summary(s,step)
                _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
                if (step % 10000 == 0):
                    print("Minibatch loss at step %d: %f" % (step, l))
                    print("Minibatch accuracy: %.1f%%" % self.accuracy(predictions, batch_labels))
                    print("Validation accuracy: %.1f%%" % self.accuracy(
                        valid_prediction.eval(), self.valid_labels))
            print("Test accuracy: %.1f%%" % self.accuracy(test_prediction.eval(), self.test_labels))

        return

    def accuracy(self, predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                / predictions.shape[0])

    def make_hparam_string(self,learning_rate, hidden_nodes,fvi):
        fchn = ""
        for i in hidden_nodes:
            fchn = fchn + "_{}".format(str(i))
            
        return "lr_{0}_fchn_{1}_fvi_{2}".format(learning_rate, fchn, fvi)

    def LoadDatasetFromCSV(self):
        with open(self.gml_folder_path + "/preprocessed.pickle", 'rb') as f:
            save = pickle.load(f)
            decimals = 6
            self.train_dataset = np.around(save['train_dataset'],decimals=decimals)
            self.train_labels = np.around(save['train_labels'],decimals=decimals)
            self.valid_dataset = np.around(save['valid_dataset'],decimals=decimals)
            self.valid_labels = np.around(save['valid_labels'],decimals=decimals)
            self.test_dataset = np.around(save['test_dataset'],decimals=decimals)
            self.test_labels = np.around(save['test_labels'],decimals=decimals)
            self.prepro_dict = save['prepro_dict']
            
            del save  # hint to help gc free up memory
            print('Training set', self.train_dataset.shape, self.train_labels.shape)
            print('Validation set', self.valid_dataset.shape, self.valid_labels.shape)
            print('Test set', self.test_dataset.shape, self.test_labels.shape)

            

    def GettingWorldDefinition(self, role = "gauss",world_type = 1,N_uav = 1,N_obs = 1,from_ROS = False):
        if from_ROS == True:
            self.world_definition = rospy.get_param('world_definition')
            self.role = self.world_definition['role']
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
            self.role = role
            self.world_type = world_type
            self.N_uav = N_uav
            self.N_obs = N_obs
            self.obs_tube = []
            self.uav_models = ["typhoon_h480","typhoon_h480","typhoon_h480"]
            self.n_dataset = 1
            self.solver_algorithm = 'orca3'
            self.home_path = 'josmilrom'
            self.depth_camera_use = False