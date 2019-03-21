# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import tarfile
import time
import hashlib
import csv
# from IPython.display import display, Image
# from sklearn.linear_model import LogisticRegression
# from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from ast import literal_eval

class PreProcesser(object):
    def __init__(self,learning_dataset_def,from_ROS = False):
        self.GettingWorldDefinition(learning_dataset_def,from_ROS)

        gml_folder_path = "/home/{0}/Libraries/gml".format("joseandresmr")
        self.session_path = gml_folder_path + "/Sessions/{0}/{1}/{2}".format(self.learning_dataset_def["teacher_role"],self.learning_dataset_def["teacher_algorithm"],self.learning_dataset_def["N_neighbors_aware"])

        conv_flag = False

        if conv_flag == True:
            image_size = [480, 640]  # [480,640*4]
            num_pixels = image_size[0]*image_size[1]
            num_channels = 1
        else:
            num_pixels = 0

        num_outputs = 3


        with open(self.session_path + "/raw.pickle", 'rb') as f:
            whole_dataset = np.asarray(pickle.load(f))

        prepro_dict = {"mean": np.mean(whole_dataset,axis=0)}
        zero_mean_dataset = whole_dataset - np.tile(prepro_dict["mean"],[whole_dataset.shape[0],1])
        prepro_dict["max"] = np.max(zero_mean_dataset,axis=0)
        clean_dataset = zero_mean_dataset / np.tile(prepro_dict["max"],[whole_dataset.shape[0],1])
        np.random.shuffle(clean_dataset)

        ### HASH
        dataset_hashes = [hashlib.sha1(x).digest() for x in clean_dataset]
        ###

        train_percentage = 0.7
        valid_percentage = 0.2
        test_percentage = 1 - train_percentage - valid_percentage

        dataset_len = clean_dataset.shape[0]

        valid_index = int(np.floor(dataset_len*train_percentage))
        test_index = int(np.floor(dataset_len*(train_percentage+test_percentage)))

        train_dataset = clean_dataset[0:valid_index,:-num_outputs]
        train_labels = clean_dataset[0:valid_index,-num_outputs:]
        valid_dataset = clean_dataset[valid_index:test_index,:-num_outputs]
        valid_labels = clean_dataset[valid_index:test_index,-num_outputs:]
        test_dataset = clean_dataset[test_index:,:-num_outputs]
        test_labels = clean_dataset[test_index:,-num_outputs:]

        try:
            f = open(self.session_path + "/preprocessed.pickle", 'wb')
            save = {
                'train_dataset': train_dataset,
                'train_labels': train_labels,
                'valid_dataset': valid_dataset,
                'valid_labels': valid_labels,
                'test_dataset': test_dataset,
                'test_labels': test_labels,
                'prepro_dict' : prepro_dict
                }

            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', self.session_path + "/preprocessed.pickle", ':', e)
            raise

        print("PREPROCESSMENT done")

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

# proprocesser = PreProcesser()
