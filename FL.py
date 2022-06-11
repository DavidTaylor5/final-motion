from gc import callbacks
import os
import math

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl
import tensorflow as tf
import numpy as np

import binaryCNN
import dataPreprocess

from typing import Optional, Tuple

import matplotlib as plt

# I need to make sure that my federated learning settings are reproducible
# I need to create a 'seed' for both python random and tf random
#####################################################################
RANDOM_SEED = 47568
#seed(47568
# #tf.random.set_random_seed(seed_value))
os.environ['PYTHONHASHSEED']=str(47568)
#random.seed(47568)
tf.random.set_seed(47568)

np.random.seed(RANDOM_SEED)

NUM_CLIENTS = 24

class FlwerClient(fl.client.NumPyClient):
    def __init__(self, model, cid) -> None:
        super().__init__()
        self.model = model
        self.cid = cid
        self.acc = []
        self.loss = []


    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):

        #print("before mydata? Error?")

        myData = read_part_data_file(self.cid) #this doesn't exist in ray client memeory? read from file!

        #print("I have gotten my data!", myData[0].shape, myData[2].shape)

        self.model.set_weights(parameters)

        #print("I have set my weights!")
        #my_early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5) #added early stopping

        self.model.fit(myData[0], myData[1], epochs=35, verbose=0) #I could potentially attach a callback function here to make early stopping?

        #print("I have fitted my model!")
        #callbacks=[my_early_stop]

        return self.model.get_weights(), len(myData[0]), {}
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)

        myData = read_part_data_file(self.cid) #this doesn't exist in ray client memeory? read from file!

        loss, acc = self.model.evaluate(myData[2], myData[3], verbose=2)

        return loss, len(myData[2]), {"accuracy":acc}



# necessary data below, window data for each participant
partData = dataPreprocess.getIndividualDatasets(24)
dataPreprocess.normalizeParticipants(partData)
pooledData = dataPreprocess.getCentralDataset(partData)

part_windows = binaryCNN.participantWindows(partData, 50)

binaryCNN.participant_list_to_binary(part_windows)
pooled_windows = binaryCNN.poolWindows(part_windows, 50) #pooling all participants
pooled_men_test = binaryCNN.pool_by_attribute(binaryCNN.male_indexes, part_windows)
pooled_women_test = binaryCNN.pool_by_attribute(binaryCNN.female_indexes, part_windows)

#I need the correct data!

print(len(part_windows))


def part_windows_to_file(part_windows):

    foldername = './part_windows_folder'

    for i in range (0, len(part_windows)):


        #I create 4 files for each participant
        fileName = '/participantTRAINX' + str(i) + '.npy'
        with open(foldername+fileName, 'wb') as f:
            np.save(f, part_windows[i][0])

        fileName = '/participantTRAINy' + str(i) + '.npy'
        with open(foldername+fileName, 'wb') as f:
            np.save(f, part_windows[i][1])

        fileName = '/participantTESTX' + str(i) + '.npy'
        with open(foldername+fileName, 'wb') as f:
            np.save(f, part_windows[i][2])

        fileName = '/participantTesty' + str(i) + '.npy'
        with open(foldername+fileName, 'wb') as f:
            np.save(f, part_windows[i][3])

def read_part_data_file(participantID): #I STARTED OUT WITH 139 GB of storate -> will this continue to disappear with ray spilled IO objects? After 10 rounds... -> Why is there no change in first 7 rounds?
    #I open 4 files and grab their numpy contents
    foldername = './part_windows_folder'

    with open(foldername + "/participantTRAINX" + str(participantID) + '.npy', 'rb' ) as f:
        train_X = np.load(f)

    with open(foldername + "/participantTRAINy" + str(participantID) + '.npy', 'rb' ) as f:
        train_y = np.load(f)

    with open(foldername + "/participantTESTX" + str(participantID) + '.npy', 'rb' ) as f:
        test_X = np.load(f)

    with open(foldername + "/participantTESTy" + str(participantID) + '.npy', 'rb' ) as f:
        test_y = np.load(f)

    return [train_X, train_y, test_X, test_y]
    




def client_fn(cid: str) -> fl.client.Client:
    # Load model
    model = binaryCNN.sensor_activity_binary(n_timesteps=50, n_features=12, n_outputs=2) #as specified by David M

    #(x_train, y_train, x_val, y_val) = part_windows[int(cid)] #rather than giving workers their data to hold, tell them to reference dataset? Reference to part_windows

    # Create and return client
    return FlwerClient(model, cid)

# experiemental evaluate_config for clients
def evaluate_config(rnd: int): #EXPERIMENTAL
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps":val_steps}



def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        model.set_weights(weights)  # Update model with the latest parameters

        binaryCNN.check_fairness(model, pooled_men_test, pooled_women_test)
        score = model.evaluate(pooled_windows[2], pooled_windows[3], verbose=0) #checking score with pooled test set

        #I need to append my server level model's loss in a file then I can disply it as a graph!
        with open("FederatedLoss/FL.txt", "a") as f:
            f.write(str(score[0]) + "\n")
        #print('Test loss:', score[0]) 



        print('-> Pooled Test accuracy:', score[1])



    return evaluate

#I might need to make keras less verbose
# # Load and compile model for server-side parameter evaluation
# model = tf.keras.applications.EfficientNetB0(
#     input_shape=(32, 32, 3), weights=None, classes=10
# )
# model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])


def main() -> None:

    #send a participant's data into a file!
    print("sending training data to files ...")
    part_windows_to_file(part_windows)
    print("done sending training data to bianry files!")
    # example of one participant getting their data back from files!
    #ready_work = read_part_data_file(0)


    a_model = binaryCNN.sensor_activity_binary(n_timesteps=50, n_features=12, n_outputs=2)


    # Start Flower simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        client_resources={"num_cpus":4}, #trying 4
        num_rounds=90,
        strategy=fl.server.strategy.FedAvg(
            #fraction_fit=0.1,
            min_fit_clients=24, #testing, only fitting 4 participants per round #fitting with 10 clients a round
            min_available_clients=NUM_CLIENTS,

            eval_fn=get_eval_fn(a_model)
            # fraction_eval=0.2,
            # min_eval_clients=24,
            # on_evaluate_config_fn=evaluate_config
        )
    )


if __name__ == "__main__":
    main()


"""
Results from fixing the spilling issues that were plauging the binaryFL
DI: 1.0266653205067617
EOP: 0.025154077517853923
Avg EP diff: 0.013814562882596566
SPD: 0.029326856497608467
-> Pooled Test accuracy: 0.9606661796569824

"""