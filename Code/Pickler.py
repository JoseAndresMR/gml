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
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from ast import literal_eval

import csv

class Pickler(object):
    def __init__(self,mission = "follow_paths_sbys", role = "path", n_dataset = 1, world_type = 1,N_uav = 1,N_obs = 1,from_ROS = False,fut_vel_inst = 0):
        self.GettingWorldDefinition(mission,role,n_dataset,world_type,N_uav,N_obs,from_ROS)

        self.mission_to_role_dicc = {"follow_paths_sbys":["path","path","path"],
                                     "queue_of_followers_ad":["path","uav_ad","uav_ad"],
                                     "queue_of_followers_ap":["path","uav_ap","uav_ap"]}

        self.pd_first_folder_path = "/home/{4}/catkin_ws/src/pydag/Data_Storage/Simulations/{0}/orca3/type{1}_Nuav{2}_Nobs{3}".format(self.mission,self.world_type,self.N_uav,self.N_obs,self.home_path,self.teacher_algorithm)
        self.pd_second_folder_path = self.pd_first_folder_path + "/dataset_{}".format(self.n_dataset)
        
        self.final_pickle = []
        for simulation in sorted(os.listdir(self.pd_second_folder_path)):
            if simulation.split("_")[0] == "simulation":
                third_folder_path = os.path.join(self.pd_second_folder_path,simulation)
                if os.path.isfile(os.path.join(third_folder_path, "world_definition.csv")):
                    simulation_definition = self.LoadSimulationDefinition(third_folder_path)
                    if literal_eval(simulation_definition["simulation_succeed"]) == True:
                        for n_uav in range(1,self.N_uav+1):
                            if self.role == literal_eval(simulation_definition["roles_list"])[n_uav-1]:
                                single_pickle = self.LoadUAVData(third_folder_path,n_uav,simulation_definition,simulation,fut_vel_inst)
                                if self.final_pickle == [] and single_pickle != []: 
                                    self.final_pickle = single_pickle
                                elif single_pickle != []:
                                    self.final_pickle = np.concatenate((self.final_pickle,single_pickle),axis=0)
                else:
                    print("No Simulation definition for {}".format(simulation))
        self.MakePickle()
        print("PICKLING done")

    def LoadSimulationDefinition(self,third_folder_path):

        with open(third_folder_path + "/world_definition.csv", mode='r') as infile:
            reader = csv.reader(infile)
            simulation_definition = dict((rows[0],rows[1]) for rows in reader)

        return simulation_definition

    def LoadUAVData(self,third_folder_path,uav_id,simulation_definition,simulation,fut_vel_inst):
        single_pickle = []
        if os.path.isfile(os.path.join(third_folder_path, "uav_{}.csv".format(uav_id))):

            df = pd.read_csv(third_folder_path + "/uav_{}.csv".format(uav_id),',')

            if self.role.split("_")[1] == "depth":
                with open(third_folder_path + "/depth_camera_{}.pickle".format(uav_id), 'rb') as f:
                    depth_camera = np.asarray(pickle.load(f))

            if "uavs_list" in df.keys():
                instants = len(df["uavs_list"])
                instants = instants - fut_vel_inst

            else:
                return single_pickle

            if self.role == "path":
                input_dicc = ['own_vel','goal_pose_rel','others_pos_rel','others_vel','obs_pos_rel']
                output_dicc = ["sel_vel"]
            elif self.role == "uav_ad":
                input_dicc = ['own_vel','goal_pose_rel','goal_vel','distance','others_pos_rel','others_vel','obs_pos_rel']
                output_dicc = ["sel_vel"]
            elif self.role == "uav_ap":
                input_dicc = ['own_vel','goal_pose_rel','goal_vel','others_pos_rel','others_vel','obs_pos_rel']
                output_dicc = ["sel_vel"]

            elif self.role == "path_depth":
                input_dicc = ['own_vel', 'own_ori','goal_pose_rel','image_depth']
                output_dicc = ["sel_vel"]

            for n_input in input_dicc:
                own_pos = np.asarray([literal_eval(df["uavs_list"][i])[uav_id-1]["Pose"][0] for i in range(instants)])
                own_vel = np.asarray([literal_eval(df["uavs_list"][i])[uav_id-1]["Twist"][0] for i in range(instants)])
                goal_pos = np.asarray([literal_eval(df["goal"][i])[0]["Pose"][0] for i in range(instants)])
                goal_vel = np.asarray([literal_eval(df["goal"][i])[0]["Twist"][0] for i in range(instants)])
            
                if n_input == "own_vel":
                    single_pickle = own_vel
                elif n_input == "own_ori":
                    own_ori = np.asarray([literal_eval(df["uavs_list"][i])[uav_id-1]["Pose"][1] for i in range(instants)])
                    single_pickle = own_ori
                elif n_input == "goal_pose_rel":
                    single_pickle = np.concatenate((single_pickle,goal_pos-own_pos),axis=1)
                elif n_input == "goal_vel":
                    single_pickle = np.concatenate((single_pickle,goal_vel),axis=1)

                elif n_input == "distance":
                    distance = np.asarray([[literal_eval(df["goal"][i])[0]["distance"]] for i in range(instants)])
                    single_pickle = np.concatenate((single_pickle,distance),axis=1)

                elif n_input == "image_depth":
                    single_pickle = np.concatenate((single_pickle,depth_camera),axis=1)

                elif n_input == "others_pos_rel":
                    for n_uav in range(self.N_uav):
                        if n_uav != uav_id-1:
                            # print(n_input)
                            other_pos = np.asarray([literal_eval(df["uavs_list"][i])[n_uav]["Pose"][0] for i in range(instants)])
                            other_pos_rel = other_pos - own_pos
                            single_pickle = np.concatenate((single_pickle,other_pos_rel),axis=1)
                    
                elif n_input == "others_vel":
                    for n_uav in range(self.N_uav):
                        if n_uav != uav_id-1:
                            # print(n_input)
                            other_vel = np.asarray([literal_eval(df["uavs_list"][i])[n_uav]["Twist"][0] for i in range(instants)])
                            single_pickle = np.concatenate((single_pickle,other_vel),axis=1)
                    
                elif n_input == "obs_pos_rel":
                    for n_obs in range(self.N_obs):
                        # print(n_input)
                        obs_pos = np.tile(literal_eval(simulation_definition['obs_pose_list'][1:-1])[n_obs],[instants,1])
                        obs_pos_rel = obs_pos - own_pos
                        single_pickle = np.concatenate((single_pickle,obs_pos_rel),axis=1)

            for n_output in output_dicc:
                if n_output == "sel_vel":
                    # print(n_output)
                    sel_vel = np.asarray([literal_eval(df["selected_velocity"][i+fut_vel_inst])[0] for i in range(instants)])
                    single_pickle = np.concatenate((single_pickle,sel_vel),axis=1)
            # print(single_pickle[0][:])

            
        else:
            print("No UAV {} storage in {}".format(uav_id,simulation))

        return single_pickle

    def MakePickle(self):
        gml_folder_path = "/home/{4}/Libraries/gml/Sessions/{0}/type{1}_Nuav{2}_Nobs{3}".format(self.role,self.world_type,self.N_uav,self.N_obs,self.home_path)

        if not os.path.exists(gml_folder_path):
            os.mkdir(gml_folder_path)
        try:
            with open(gml_folder_path + "/raw.pickle", 'wb') as f:
                pickle.dump(self.final_pickle, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', gml_folder_path + "/raw.pickle", ':', e)
  
    def GettingWorldDefinition(self,mission = "follow_paths_sbys", role = "path", n_dataset = 1,world_type = 1,N_uav = 1,N_obs = 1,from_ROS = False):
        if from_ROS == True:
            self.world_definition = rospy.get_param('world_definition')
            self.mission = self.world_definition['mission']
            self.world_type = self.world_definition['type']
            self.N_uav = self.world_definition['N_uav']
            self.N_obs = self.world_definition['N_obs']
            self.obs_tube = self.world_definition['obs_tube']
            self.uav_models = self.world_definition['uav_models']
            self.n_dataset = self.world_definition['n_dataset']
            self.solver_algorithm = self.world_definition['solver_algorithm']
            self.obs_pose_list = self.world_definition['obs_pose_list']
            self.home_path = self.world_definition['home_path']
            self.image_depth_use = self.world_definition['image_depth_use']

        if from_ROS == False:
            self.mission = mission
            self.world_type = world_type
            self.N_uav = N_uav
            self.N_obs = N_obs
            self.role = role
            self.obs_tube = []
            self.uav_models = ["typhoon_h480","typhoon_h480","typhoon_h480"]
            self.n_dataset = n_dataset
            self.teacher_algorithm = 'orca3'
            self.home_path = 'josmilrom'
            self.image_depth_use = False




# pickler = Pickler()