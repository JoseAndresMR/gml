from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

class Trainer(object):
    def __init__(self, project = "gauss",world_type = 1,N_uav = 1,N_obs = 1,from_ROS = False):
        self.GettingWorldDefinition(project,world_type,N_uav,N_obs,from_ROS)

        self.gml_folder_path = "/home/{4}/Libraries/gml/Sessions/{0}/{5}/type{1}_Nuav{2}_Nobs{3}".format(self.role,self.world_type,self.N_uav,self.N_obs,self.home_path,self.solver_algorithm)

    def LoadGraph(self):
        pass

    def ReceiveGraph(self,graph):
        self.graph = graph

    def Train(self,batch_size,num_steps):

        self.LoadDatasetFromCSV()

        tf.reset_default_graph()
        v1 = tf.get_variable("v1", shape=[3]) #SOME VARIABLES TO SKIP ERROR
        saver = tf.train.Saver()


        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            print("Initialized")
            for step in range(num_steps):
                saver.restore(sess, "/home/josmilrom/Libraries/gml/model.ckpt")

                # Pick an offset within the training data, which has been randomized.
                # Note: we could use better randomization across epochs.
                offset = (step * batch_size) % (self.train_labels.shape[0] - batch_size)
                # Generate a minibatch.
                batch_data = self.train_dataset[offset:(offset + batch_size), :]
                batch_labels = self.train_labels[offset:(offset + batch_size), :]
                # Prepare a dictionary telling the session where to feed the minibatch.
                # The key of the dictionary is the placeholder node of the graph to be fed,
                # and the value is the numpy array to feed to it.
                feed_dict = {"tf_train_dataset" : batch_data, "tf_train_labels" : batch_labels}
                _, l, predictions = sess.run(
                    [optimizer, loss, train_prediction], feed_dict=feed_dict)
                if (step % 500 == 0):
                    print("Minibatch loss at step %d: %f" % (step, l))
                    print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                    print("Validation accuracy: %.1f%%" % accuracy(
                        valid_prediction.eval(), valid_labels))
            print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

    def LoadDatasetFromCSV(self):
        with open(self.gml_folder_path + "/preprocessed.pickle", 'rb') as f:
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

# trained_graph = GraphTrainer()