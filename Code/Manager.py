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

            pickler = Pickler(mission,role,n_dataset,world_type,N_uav,N_obs,from_ROS,fvi)
            preprocesser = PreProcesser(role,world_type,N_uav,N_obs)

            graphmakertrainer = GraphMakerTrainer(role,world_type,N_uav,N_obs)
            batch_size = 100
            hidden_nodes_list = [[20,20,20],[20,30,20]]
            learning_rates_list = [0.001,0.0001,0.00001]
            num_steps = 100000
            for learning_rate in learning_rates_list:
                for hidden_nodes in hidden_nodes_list:
                    graph = graphmakertrainer.FullyConnected(batch_size,hidden_nodes,num_steps, learning_rate,fvi)

manager = Manager()