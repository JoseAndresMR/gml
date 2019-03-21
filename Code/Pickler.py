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
import rospkg

# from IPython.display import display, Image
# from sklearn.linear_model import LogisticRegression
# from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from ast import literal_eval

import csv

class Pickler(object):
    def __init__(self,learning_dataset_def,from_ROS = False,fut_vel_inst = 0):
        # Extraction of hyperarameters depending on the souerce whether ROS params or by arguments
        self.GettingWorldDefinition(learning_dataset_def,from_ROS)

        # Build, from hyperparams, the path where to retrieve the dataset from
        self.magna_home_path = rospkg.RosPack().get_path('magna')[:-5]
        self.pd_first_folder_path = self.magna_home_path + "/Data_Storage/Simulations/{0}/{1}/{2}/{3}".format(
                                                        learning_dataset_def["folders"]["world"],
                                                        learning_dataset_def["folders"]["subworld"],
                                                        learning_dataset_def["folders"]["mission"],
                                                        learning_dataset_def["folders"]["submission"])
        self.pd_second_folder_path = self.pd_first_folder_path + "/dataset_{}".format(learning_dataset_def["folders"]["n_dataset"])
        
        self.final_pickle = []      # Create empty pickle

        # Check every simulation contained into the search path
        for simulation in sorted(os.listdir(self.pd_second_folder_path)):
            if simulation.split("_")[0] == "simulation":
                simulation_path = os.path.join(self.pd_second_folder_path,simulation)

                # Check if there is a world definition inside
                if os.path.isfile(os.path.join(simulation_path, "world_definition.csv")):

                    # Introspect that world definition and check the succeed field
                    simulation_definition = self.LoadSimulationDefinition(simulation_path)
                    if literal_eval(simulation_definition["simulation_succeed"]) == True:

                        # Look for Agents with the desired role
                        for n_agent in range(1,int(simulation_definition["N_agents"])+1):

                            if os.path.isfile(os.path.join(simulation_path, "agent_{}.csv".format(n_agent))):

                                df = pd.read_csv(simulation_path + "/agent_{}.csv".format(n_agent),',')
                                df_role_and_algorithm = df[df.role==self.learning_dataset_def["teacher_role"]]
                                # df_role_and_algorithm = df[df.algorithms_list==[self.learning_dataset_def["teacher_agorithm"]]]

                                # Load data, parsing it into a little pickle
                                single_pickle = self.LoadAgentData(df_role_and_algorithm,self.learning_dataset_def["teacher_role"],fut_vel_inst)
                                
                                # If it is not empty, add it to the pickle with the rest of simulations
                                if self.final_pickle == [] and single_pickle != []: 
                                    self.final_pickle = single_pickle
                                elif single_pickle != []:
                                    self.final_pickle = np.concatenate((self.final_pickle,single_pickle),axis=0)
                else:
                    print("No Simulation definition for {}".format(simulation))

        # Create the picke file
        self.MakePickle()
        print("PICKLING done")


    def LoadSimulationDefinition(self,simulation_path):

        with open(simulation_path + "/world_definition.csv", mode='r') as infile:
            reader = csv.reader(infile)
            simulation_definition = dict((rows[0],rows[1]) for rows in reader)

        return simulation_definition

    def LoadAgentData(self,df,role,fut_vel_inst):
        single_pickle = []

        # if self.role.split("_")[-1] == "depth":
        #     with open(simulation_path + "/depth_camera_{}.pickle".format(agent_id), 'rb') as f:
        #         depth_camera = np.asarray(pickle.load(f))[-480*640:-1]
        #         # print(depth_camera.size)

        if "main_agent" in df.keys():
            instants = len(df["main_agent"])
            instants = instants - fut_vel_inst

        else:
            return single_pickle

        if role == "path":
            input_dicc = ['own_vel','goal_pose_rel','neighbors_pos_rel','neighbors_vel','obs_pos_rel']
            output_dicc = ["sel_vel"]
        elif role == "agent_ad":
            input_dicc = ['own_vel','goal_pose_rel','goal_vel','distance','neighbors_pos_rel','neighbors_vel','obs_pos_rel']
            output_dicc = ["sel_vel"]
        elif role == "agent_ap":
            input_dicc = ['own_vel','goal_pose_rel','goal_vel','neighbors_pos_rel','neighbors_vel','obs_pos_rel']
            output_dicc = ["sel_vel"]

        elif role == "path_depth":
            input_dicc = ['own_vel', 'own_ori','goal_pose_rel','image_depth']
            output_dicc = ["sel_vel"]
        
        for n_input in input_dicc:
            # ME QUEDO SOLO CON LAS PARTES LINEALES, NADA DE ANGULARES!!
            role = np.asarray([df["role"][i] for i in range(instants)])
            own_pos = np.asarray([literal_eval(df["main_agent"][i])["Pose"][0] for i in range(instants)])
            own_vel = np.asarray([literal_eval(df["main_agent"][i])["Twist"][0] for i in range(instants)])
            goal_pos = np.asarray([literal_eval(df["goal"][i])[0]["Pose"][0] for i in range(instants)])     # Ponerlo sin []
            goal_vel = np.asarray([literal_eval(df["goal"][i])[0]["Twist"][0] for i in range(instants)])

            if n_input == "own_vel":
                # print("own_vel",own_vel[0].size)
                single_pickle = own_vel
            elif n_input == "own_ori":
                own_ori = own_pos
                single_pickle = own_ori
            elif n_input == "goal_pose_rel":
                # print("goal",goal_pos[0].size)
                # print("pickle",single_pickle)
                single_pickle = np.concatenate((single_pickle,goal_pos),axis=1)
            elif n_input == "goal_vel":
                single_pickle = np.concatenate((single_pickle,goal_vel),axis=1)

            elif n_input == "distance":
                distance = np.asarray([[literal_eval(df["goal"][i])[0]["distance"]] for i in range(instants)])
                single_pickle = np.concatenate((single_pickle,distance),axis=1)

            elif n_input == "image_depth":
                single_pickle = np.concatenate((single_pickle,depth_camera),axis=1)

            elif n_input == "neighbors_pos_rel":
                for n_agent in range(self.learning_dataset_def["N_neighbors_aware"]):
                    neighbor_pos_rel = np.asarray([literal_eval(df["neigh"][i])[n_agent]["Pose"][0] for i in range(instants)])
                    single_pickle = np.concatenate((single_pickle,neighbor_pos_rel),axis=1)
                
            elif n_input == "neighbors_vel":
                for n_agent in range(self.learning_dataset_def["N_neighbors_aware"]):
                    neighbor_vel = np.asarray([literal_eval(df["neigh"][i])[n_agent]["Twist"][0] for i in range(instants)])
                    single_pickle = np.concatenate((single_pickle,neighbor_vel),axis=1)
                

        for n_output in output_dicc:
            if n_output == "sel_vel":
                # print(n_output)
                sel_vel = np.asarray([literal_eval(df["selected_velocity"][i+fut_vel_inst])[0] for i in range(instants)])
                single_pickle = np.concatenate((single_pickle,sel_vel),axis=1)
        # print(single_pickle[0][:])


        return single_pickle

    def MakePickle(self):
        gml_folder_path = "/home/{0}/Libraries/gml".format("joseandresmr")
        session_path = gml_folder_path + "/Sessions/{0}/{1}/{2}".format(self.learning_dataset_def["teacher_role"],self.learning_dataset_def["teacher_algorithm"],self.learning_dataset_def["N_neighbors_aware"])
        
        if not os.path.exists(session_path):
            os.mkdir(session_path)
        try:
            with open(session_path + "/raw.pickle", 'wb') as f:
                pickle.dump(self.final_pickle, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', session_path + "/raw.pickle", ':', e)
  
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



# pickler = Pickler()