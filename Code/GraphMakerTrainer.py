# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
# from tensorflow.python.saved_model import tag_constants
from six.moves import cPickle as pickle

class GraphMakerTrainer(object):
    def __init__(self,learning_dataset_def,from_ROS = False):
        print("creating GraphMakerTrainer")
        
        self.GettingWorldDefinition(learning_dataset_def,from_ROS)

        self.conv_flag = False

        gml_folder_path = "/home/{0}/Libraries/gml".format("joseandresmr")
        self.session_path = gml_folder_path + "/Sessions/{0}/{1}/{2}".format(self.learning_dataset_def["teacher_role"],self.learning_dataset_def["teacher_algorithm"],self.learning_dataset_def["N_neighbors_aware"])


        self.LoadDatasetFromCSV()

    def TrainNewOne(self,batch_size,num_steps,fc_hidden_layers,conv_hidden_layers= [],learning_rate= 0,fvi= 0):                
        self.fc_hidden_layers = fc_hidden_layers
        self.conv_hidden_layers = conv_hidden_layers
        self.learning_rate = learning_rate
        self.fvi = fvi
        graph = tf.Graph()
        with graph.as_default():
            # Input data. For the training data, we use a placeholder that will be fed
            # at run time with a training minibatch.

            mu = 0
            sigma = 0.1

            if self.conv_flag == True:
                image_size = [480, 640]  # [480,640*4]
                num_pixels = image_size[0]*image_size[1]
                num_channels = 1
            else:
                num_pixels = 0

            # Divide inputs
            fc_num_inputs = self.train_dataset.shape[1] - num_pixels
            conv_num_inputs = num_pixels

            num_inputs = fc_num_inputs + conv_num_inputs

            # Placeholders
            fc_num_outputs = self.train_labels.shape[1]
            tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, num_inputs), name="tf_train_dataset")
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, fc_num_outputs), name="tf_train_labels")
            tf_valid_dataset = tf.constant(self.valid_dataset.astype(np.float32))
            tf_test_dataset = tf.constant(self.test_dataset.astype(np.float32))
            tf_single_input = tf.placeholder(tf.float32,shape=(1, num_inputs), name="single_input")

            fc_train_dataset = tf_train_dataset[:,:fc_num_inputs]
            fc_train_labels = tf_train_labels[:,:fc_num_inputs]
            fc_valid_dataset = tf_valid_dataset[:,:fc_num_inputs]
            fc_test_dataset = tf_test_dataset[:,:fc_num_inputs]

            prepro_mean = tf.constant(self.prepro_dict["mean"].astype(np.float32))
            prepro_max = tf.constant(self.prepro_dict["max"].astype(np.float32))

            # Start FC params
            self.fc_weights = []
            self.fc_biases = []

            def fully_connected(data, name = "fully_connected"):
                with tf.name_scope(name):
                    
                    fc_layer_outputs = []
                    fc_extended_num_inputs = int(data.shape[1])

                    nodes = [fc_extended_num_inputs]

                    for i in range(len(self.fc_hidden_layers)):
                        nodes.append(self.fc_hidden_layers[i])
                    nodes.append(fc_num_outputs)

                    for n_layer in range(0,len(nodes)-1):
                        if len(self.fc_weights) < len(nodes):

                            self.fc_weights.append(tf.Variable(tf.truncated_normal([nodes[n_layer], nodes[n_layer+1]], mean = mu, stddev=sigma),name = "cd_w_{}".format(n_layer+1)))
                            self.fc_biases.append(tf.Variable(tf.zeros([nodes[n_layer+1]]),name = "cd_b_{}".format(n_layer+1)))

                        if fc_layer_outputs == []:
                            fc_layer_outputs = [tf.nn.tanh(tf.matmul(data, self.fc_weights[n_layer]) + self.fc_biases[n_layer])]
                        else:
                            fc_layer_outputs.append(tf.nn.tanh(tf.matmul(fc_layer_outputs[n_layer-1], self.fc_weights[n_layer]) + self.fc_biases[n_layer]))

                        tf.summary.histogram("fc_weights",self.fc_weights[n_layer])
                        tf.summary.histogram("fc_biases",self.fc_biases[n_layer])
                        # tf.summary.histogram("activations",fc_layer_outputs[n_layer-1])

                    return fc_layer_outputs, self.fc_weights, self.fc_biases

            # Start CONV params

            if self.conv_flag == True:

                conv_train_dataset = tf_train_dataset[:, fc_num_inputs:]
                print(conv_train_dataset.shape)
                conv_train_dataset = tf.manip.reshape(conv_train_dataset, (batch_size, image_size[0], image_size[1], num_channels))
                conv_valid_dataset = tf_valid_dataset[:, fc_num_inputs:]
                conv_test_dataset = tf_valid_dataset[:, fc_num_inputs:]

                self.conv_weights = []
                self.conv_biases = []

                def convolutional(data, name = "convolutional"):
                    with tf.name_scope(name):
                        conv_layer_outputs = []

                        # layers = [conv_num_inputs]
                        # for layer in self.conv_hidden_layers:
                        #     nlayersappend(layer)
                        # layers.append(conv_num_outputs)
                        print("flag0")
                        for n_layer in range(len(self.conv_hidden_layers)):
                            print("flag1")
                            if len(self.conv_weights) < len(self.conv_hidden_layers):
                                print("flag2")
                                if n_layer == 0:
                                    print("flag3")
                                    self.conv_weights = [tf.Variable(tf.truncated_normal(
                                        [self.conv_hidden_layers[n_layer]["patch_size"], self.conv_hidden_layers[n_layer]["patch_size"],
                                        num_channels, self.conv_hidden_layers[n_layer]["depth"]], mean = mu, stddev=sigma),name = "conv_w_{}".format(n_layer+1))]
                                    self.conv_biases = [tf.Variable(tf.zeros([self.conv_hidden_layers[n_layer]["depth"]]),name = "conv_b_{}".format(n_layer+1))]

                                else:
                                    print("flag4")
                                    self.conv_weights.append(tf.Variable(tf.truncated_normal(
                                        [self.conv_hidden_layers[n_layer]["patch_size"], self.conv_hidden_layers[n_layer]["patch_size"], 
                                        self.conv_hidden_layers[n_layer-1]["depth"], self.conv_hidden_layers[n_layer]["depth"]], mean = mu, stddev=sigma),name = "conv_w_{}".format(n_layer+1)))
                                    self.conv_biases.append(tf.Variable(tf.constant(1.0, shape=[self.conv_hidden_layers[n_layer]["depth"]]),name = "conv_b_{}".format(n_layer+1)))

                            if conv_layer_outputs == []:
                                print("flag5")
                                conv = tf.nn.conv2d(data, self.conv_weights[n_layer], [1, 1, 1, 1], padding=self.conv_hidden_layers[n_layer]["padding"])
                                print(conv.shape)
                                hidden = tf.nn.relu(conv + self.conv_biases[n_layer])
                                print(hidden.shape)
                                conv_layer_outputs.append(tf.nn.max_pool(hidden,[1, 2, 2, 1],[1, 2, 2, 1], padding=self.conv_hidden_layers[n_layer]["padding"]))
                                print(conv_layer_outputs[-1].shape)
                            else:
                                print("flag6")
                                conv = tf.nn.conv2d(conv_layer_outputs[-1], self.conv_weights[n_layer], [1, 1, 1, 1], padding=self.conv_hidden_layers[n_layer]["padding"])
                                print(conv.shape)
                                hidden = tf.nn.relu(conv + self.conv_biases[n_layer])
                                print(hidden.shape)
                                conv_layer_outputs.append(tf.nn.max_pool(hidden,[1, 2, 2, 1],[1, 2, 2, 1], padding=self.conv_hidden_layers[n_layer]["padding"]))
                                print(conv_layer_outputs[-1].shape)

                        conv_layer_outputs.append(tf.contrib.layers.flatten(conv_layer_outputs[-1]))

                        return conv_layer_outputs, self.conv_weights, self.conv_biases


            # Complete model

            def complete_model(fc_train_dataset, conv_train_dataset = [], name="complete_model"):
                with tf.name_scope(name):

                    if self.conv_flag == True:
                        conv_layer_outputs, conv_weights, conv_biases = convolutional(conv_train_dataset)
                        print("out of FC",conv_layer_outputs[-1].shape)
                        print(fc_train_dataset.shape)
                        fc_train_dataset_expanded = tf.concat([fc_train_dataset,conv_layer_outputs[-1]],axis=1)

                        fc_layer_outputs, fc_weights, fc_biases = fully_connected(fc_train_dataset_expanded)

                    else:
                        fc_layer_outputs, fc_weights, fc_biases = fully_connected(fc_train_dataset)

                    return fc_layer_outputs, fc_weights, fc_biases

            if self.conv_flag == True:
                fc_layer_outputs, fc_weights, fc_biases = complete_model(fc_train_dataset, conv_train_dataset)
            
            else:

                fc_layer_outputs, fc_weights, fc_biases = complete_model(fc_train_dataset)

            logits = tf.add(fc_layer_outputs[-1], 0, name = "logits")
                        
            tf.summary.histogram("logits",logits)
            
            def get_single_vel(tf_single_input, name="get_single_vel"):
                with tf.name_scope(name):

                    tf_single_input_pretreated = tf.div(tf_single_input - prepro_mean[:num_inputs], prepro_max[:num_inputs], name='single_input_pretreated')

                    if self.conv_flag == True:

                        fc_single_input_pretreated = tf_single_input_pretreated[:, :fc_num_inputs]
                        conv_single_input_pretreated = tf_single_input_pretreated[:, fc_num_inputs:]
                        conv_single_input_pretreated = tf.manip.reshape(conv_single_input_pretreated, (1, image_size[0], image_size[1], num_channels))

                        single_logits, fc_weights, fc_biases = complete_model(fc_single_input_pretreated,conv_single_input_pretreated)

                    else:

                        fc_single_input_pretreated = tf_single_input_pretreated

                        single_logits, fc_weights, fc_biases = complete_model(fc_single_input_pretreated)

                    posttreated = tf.add(tf.multiply(single_logits[-1],prepro_max[num_inputs:]), prepro_mean[num_inputs:])
                    
                    return posttreated

            single_logits = tf.add(get_single_vel(tf_single_input), 0, name = "vel_posttreated")

            # Regularization
            with tf.name_scope("regularization"):
                beta = 5e-4 #tf.Variable([5e-6])
                loss_regu = 0
                for n_layer in range(1,len(fc_hidden_layers)):
                    loss_regu += beta*tf.nn.l2_loss(fc_weights[n_layer])

            # Loss
            with tf.name_scope("loss"):
                loss = tf.losses.mean_squared_error(labels=tf_train_labels, predictions =logits) + loss_regu
                tf.summary.scalar("loss",loss)
            # Optimizer
            with tf.name_scope("train"):
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

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
            hparam_str = self.make_hparam_string(self.learning_rate,self.fc_hidden_layers,self.fvi)
            print(hparam_str) 
            writer = tf.summary.FileWriter(self.session_path + "/log/",graph)
            save_path = saver.save(session,self.session_path + "/model")
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
                print(batch_data.shape,"jier")
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
        with open(self.session_path + "/preprocessed.pickle", 'rb') as f:
            save = pickle.load(f)
            decimals = 2
            self.train_dataset = np.around(save['train_dataset'],decimals=decimals)
            self.train_labels = np.around(save['train_labels'],decimals=decimals)
            self.valid_dataset = np.around(save['valid_dataset'],decimals=decimals)
            self.valid_labels = np.around(save['valid_labels'],decimals=decimals)
            self.test_dataset = np.around(save['test_dataset'],decimals=decimals)
            self.test_labels = np.around(save['test_labels'],decimals=decimals)
            self.prepro_dict = save['prepro_dict']
            
            del save  # hint to help gc free up memory
            print('FC Training set', self.train_dataset.shape, self.train_labels.shape)
            print('FC Validation set', self.valid_dataset.shape, self.valid_labels.shape)
            print('FC Test set', self.test_dataset.shape, self.test_labels.shape)

    def GettingWorldDefinition(self,learning_dataset_def,from_ROS = False):
        if from_ROS == True:
            self.world_definition = rospy.get_param('world_definition')
            self.mission = self.world_definition['mission']
            self.world_type = self.world_definition['type']
            self.N_agents = self.world_definition['N_agents']
            self.N_obs = self.world_definition['N_obs']
            self.obs_tube = self.world_definition['obs_tube']
            self.agent_models = self.world_definition['agent_models']
            self.n_dataset = self.world_definition['n_dataset']
            self.solver_algorithm = self.world_definition['solver_algorithm']
            self.obs_pose_list = self.world_definition['obs_pose_list']
            self.home_path = self.world_definition['home_path']
            self.image_depth_use = self.world_definition['image_depth_use']

        if from_ROS == False:
            self.learning_dataset_def = learning_dataset_def

