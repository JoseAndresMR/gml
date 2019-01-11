from Pickler import *
from PreProcesser import *
from GraphMaker import *
from Trainer import *
from GraphMakerTrainer import *

class Manager(object):
    def __init__(self):

        ### Hyperparameters selection

        n_dataset = 1                           # Where to retrieve supervised dataset

        mission = "pruebas"   
        # mission = "queue_of_followers_ad"
        # mission = "queue_of_followers_ap"     # What mission of the dataset choose for supervised dataset
        
        world_type = 1
        N_uav = 2
        N_obs = 2                            # Type of missions for supervised dataset        
        
        role = "path"
        # role = "uav_ad"
        # role = "uav_ap"
        # role = "path_depth"                     # What roles of the mission choose for supervised dataset
  
        from_ROS = False                        # Retrieve simulation hyperparameters from ROS params
        future_vel_inst_list = [0]              # Instants in advace to feed the neural network

        batch_size = 5                          # Number of instants from wich to learn every time
        learning_rates_list = [0.001]           # Gradient influence on weights tunning
        num_steps = 1                           # Number of baches used while training

        # Fully Connected parameters
        fc_hidden_layers_list = [[20,20,20]]

        # Convolutional parameters
        conv_hidden_layers_list = [[{"patch_size" : 5,
                                    "depth" : 16,
                                    "padding" : "SAME"},
                                    {"patch_size" : 5,
                                    "depth" : 16,
                                    "padding" : "SAME"},
                                    ]]

        # Iterations for every future velocity instant in the list
        for fvi in future_vel_inst_list:

            # A pickling and a preprocessment are assessed for the whole data providing required parameters
            pickler = Pickler(mission,role,n_dataset,world_type,N_uav,N_obs,from_ROS,fvi)
            preprocesser = PreProcesser(role,world_type,N_uav,N_obs)

            # Creating the object for training
            graphmakertrainer = GraphMakerTrainer(role,world_type,N_uav,N_obs)

            # Iteration between the different parameters that definde the different trainings
            for learning_rate in learning_rates_list:
                for fc_hidden_layers in fc_hidden_layers_list:
                    for conv_hidden_layers in conv_hidden_layers_list:

                        # Making a training defined with the actual chosen params
                        graph = graphmakertrainer.TrainNewOne(batch_size,num_steps,fc_hidden_layers,
                                                                conv_hidden_layers,learning_rate,fvi)

manager = Manager()