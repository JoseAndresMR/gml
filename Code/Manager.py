import rospkg

from Pickler import *
from PreProcesser import *
from GraphMaker import *
from Trainer import *
from GraphMakerTrainer import *

class Manager(object):
    def __init__(self):

        ### Hyperparameters selection

        learning_dataset_def = {}

        learning_dataset_folders = {}
        learning_dataset_folders["world"] = "Gym"
        learning_dataset_folders["subworld"] = "Empty_4_Cylinders"
        learning_dataset_folders["mission"] = "Gym"
        learning_dataset_folders["submission"] = "2UAVs_2paths"
        learning_dataset_folders["n_dataset"] = 1
        learning_dataset_def["folders"] = learning_dataset_folders

        learning_dataset_def["N_neighbors_aware"] = 1
        learning_dataset_def["teacher_role"] = "path"
        learning_dataset_def["teacher_algorithm"] = "orca3"

        steps_to_perform = {}
        steps_to_perform["Pickle"] = True
        steps_to_perform["Preprocess"] = True
        steps_to_perform["Train"] = True

        from_ROS = False                        # Retrieve simulation hyperparameters from ROS params
        future_vel_inst_list = [0]              # Instants in advace to feed the neural network

        batch_size = 500                          # Number of instants from wich to learn every time
        learning_rates_list = [0.001]           # Gradient influence on weights tunning
        num_steps = 50000                           # Number of baches used while training

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
            if steps_to_perform["Pickle"]:
                pickler = Pickler(learning_dataset_def,from_ROS,fvi)

            if steps_to_perform["Preprocess"]:
                preprocesser = PreProcesser(learning_dataset_def)

            # Creating the object for training
            if steps_to_perform["Train"]:
                graphmakertrainer = GraphMakerTrainer(learning_dataset_def)

                # Iteration between the different parameters that definde the different trainings
                for learning_rate in learning_rates_list:
                    for fc_hidden_layers in fc_hidden_layers_list:
                        for conv_hidden_layers in conv_hidden_layers_list:

                            # Making a training defined with the actual chosen params
                            graph = graphmakertrainer.TrainNewOne(batch_size,num_steps,fc_hidden_layers,
                                                                    conv_hidden_layers,learning_rate,fvi)

manager = Manager()