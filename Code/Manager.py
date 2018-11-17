from Pickler import *
from PreProcesser import *
from GraphMaker import *
from Trainer import *
from GraphMakerTrainer import *

class Manager(object):
    def __init__(self):

        mission = "follow_paths_sbys"
        # mission = "queue_of_followers_ad"
        # mission = "queue_of_followers_ap"
        
        role = "path"
        role = "uav_ad"
        role = "uav_ap"
        role = "path_depth"
        
        n_dataset = 1
        world_type = 3
        N_uav = 1
        N_obs = 133
        future_vel_inst = 0

        from_ROS = False

        future_vel_inst_list = [0]

        for fvi in future_vel_inst_list:

            # pickler = Pickler(mission,role,n_dataset,world_type,N_uav,N_obs,from_ROS,fvi)
            # preprocesser = PreProcesser(role,world_type,N_uav,N_obs)

            graphmakertrainer = GraphMakerTrainer(role,world_type,N_uav,N_obs)
            batch_size = 88

            # FULLY CONNECTED HYPERPARAMS
            fc_hidden_layers_list = [[20,20,20]]

            # COVNET HYPERPARAMS 
            conv_hidden_layers_list = [[{"patch_size" : 5,
                                      "depth" : 16,
                                      "padding" : "SAME"},
                                      {"patch_size" : 5,
                                      "depth" : 16,
                                      "padding" : "SAME"},
                                      ]]

            learning_rates_list = [0.001]
            num_steps = 1

            for learning_rate in learning_rates_list:
                for fc_hidden_layers in fc_hidden_layers_list:
                    for conv_hidden_layers in conv_hidden_layers_list:
                        graph = graphmakertrainer.TrainNewOne(batch_size,num_steps,fc_hidden_layers,
                                                                conv_hidden_layers,learning_rate,fvi)

manager = Manager()