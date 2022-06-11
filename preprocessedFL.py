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
        self.model.set_weights(parameters)

        #my_early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5) #added early stopping

        data_back = read_pre_data_file(self.cid)

        self.model.fit(data_back[0], data_back[1], epochs=35, verbose=0, sample_weight=data_back[4]) #I could potentially attach a callback function here to make early stopping?
        #callbacks=[my_early_stop]

        return self.model.get_weights(), len(data_back[0]), {}
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)

        data_back = read_pre_data_file(self.cid)

        loss, acc = self.model.evaluate(data_back[2], data_back[3], verbose=2)


        return loss, len(data_back[2]), {"accuracy":acc}



# def attribute_train_count(attribute_indexes, part_windows):
#     count = 0;
#     for i in range(0, len(part_windows)):
#         if(i in attribute_indexes):
#             count += len(part_windows[i][0])
#     return count


# necessary data below, window data for each participant
partData = dataPreprocess.getIndividualDatasets(24)
dataPreprocess.normalizeParticipants(partData)
pooledData = dataPreprocess.getCentralDataset(partData)

part_windows = binaryCNN.participantWindows(partData, 50)

binaryCNN.participant_list_to_binary(part_windows)
pooled_windows = binaryCNN.poolWindows(part_windows, 50) #pooling all participants
pooled_men_data = binaryCNN.pool_by_attribute(binaryCNN.male_indexes, part_windows)
pooled_women_data = binaryCNN.pool_by_attribute(binaryCNN.female_indexes, part_windows)




print("what")

#what specifically is the protected attribute and unprotected attribute (does this refer to current state or 'should' state)

#PROT_ATTR_NAME = male
#UNPROT_ATTR_NAME = female
#[  [X_train, y_train, X_test, y_test]     ] -> for participants, for pooled by attribute, for all participants pooled.

#this is for centrlized 

#maybe a description of train_df
#create a csv with columns imgid, gender, pred
#train df has all the features
# so add preditions to test dataframe argmax

# PROT -> male, samples for male and female, 

# is assignwegiths_fb something that the client has access to, are they the ones updating their own weights or is this the server?

# Why are cs0y0, cs0y1, cs1y1, and cs1y0 all with respect to PRO_ATTR_NAME (cs0 was PROT, and cs1 was UNPROT)

#find out preprocessing statistics
#count (s=male) -> number of training data that is associated with male participants / or the sum of all men_training
cs1 = pooled_men_data[0].shape[0] #12959
#count (s=female) -> number of training data that is assiciated with female participants / or sum of all women_training -> should this be 0-1 normalized
cs0 = pooled_women_data[0].shape[0] #9631 women
#count(y=1)
cy1 = np.argmax(pooled_windows[1], axis=1).sum() #the amount of training data associated with sit, std, dws, ups
#count(y=0)
cy0 = pooled_windows[1].shape[0] - cy1 #the amount of training data associated with wlk, jog

print(cs1, cs0)
print(cy1, cy0)

reduce_men_train_label = np.argmax(pooled_men_data[1], axis=1) #turns one hot labels into single values 0 or 1
reduce_women_train_label = np.argmax(pooled_women_data[1], axis=1)

#amount of protected attribute (men) and label_name == 0
cs1y0 = reduce_men_train_label[reduce_men_train_label == 0].shape[0] #4442 labels for men are wlk/jog
cs1y1 = reduce_men_train_label[reduce_men_train_label == 1].shape[0] #8517
cs0y0 = reduce_women_train_label[reduce_women_train_label == 0].shape[0] #3189
cs0y1 = reduce_women_train_label[reduce_women_train_label == 1].shape[0] #6442

print(cs0y0, cs0y1)
print(cs1y0, cs1y1)
tot = pooled_windows[1].shape[0]
print(tot)


#for a,y in zip(train_x[idx],trainy):
#idx will be the index that tells what the gender is
#train_x = [1 0 1 0 0.5], idx=2 is the column that tells if the user is male
#then train_x[2] = 1

#now it will be easier to assign weights with pooled_male and pooled_female then concatentate into combo_attributes -> use this with sample weights to train



#keras sample_weights, fit has a sample_weights #apply a funciton along an axis of the DataFrame #np.apply_along_axis(func1d, axis, array)
def assignweights_fb_male(x): #protected 
    if(x == 0):
        return 1/cs1y0
    elif(x == 1):
        return 1/cs1y1
    else: #error clause
        return -100

def assignweights_fb_female(x): #unprotected
    if (x == 0):
        return 1/cs0y0
    elif(x == 1):
        return 1/cs0y1
    else: #error clause
        return -100

#assignweights_fb_male(1)

reduce_men_train_2d = np.resize(reduce_men_train_label, ( len(reduce_men_train_label), 1) ) #make 2d
reduce_women_train_2d = np.resize(reduce_women_train_label, ( len(reduce_women_train_label), 1) ) #make 2d

#now it will be easier to assign weights with pooled_male and pooled_female then concatentate into combo_attributes -> use this with sample weights to train
men_apply_weights = np.apply_along_axis(func1d=assignweights_fb_male, axis=1, arr=reduce_men_train_2d) #at 82 diff weight -> either 1/3189(82 diff) or 1/6442 (start)
women_apply_weights = np.apply_along_axis(func1d=assignweights_fb_female, axis=1, arr=reduce_women_train_2d) #are these in the correct shape?

# #used to noramlize the weights
# sum_of_attribute_weights = sum(men_apply_weights) + sum(women_apply_weights)
# y_train_length = len(men_apply_weights) + len(men_apply_weights)


# def preprocess_participant_data(attribute_participant_data):

#     for part in range(0, len(attribute_participant_data)):
#         #I just need to add weights based on each train_X and train_y
#         reduce_train_label = np.resize(attribute_participant_data[part][1], ( len(attribute_participant_data[part][1]), 1))
#         train_apply_weights = np.apply

#these should be my assigned weights!

def comboMaleFemale(pooled_men_data, men_apply_weights, pooled_women_data, women_apply_weights):
    combo_train_X = np.concatenate( (pooled_men_data[0], pooled_women_data[0]), axis=0)
    combo_train_y = np.concatenate( (pooled_men_data[1], pooled_women_data[1]), axis=0)
    combo_test_X = np.concatenate( (pooled_men_data[2], pooled_women_data[2]), axis=0)
    combo_test_y = np.concatenate( (pooled_men_data[3], pooled_women_data[3]), axis=0)

    combo_apply_weights = np.concatenate( (men_apply_weights, women_apply_weights), axis=0 )

    #normalize the weights now that I have all of them combined
    combo_apply_weights = combo_apply_weights.ravel() # make it contiguous 1d
    combo_apply_weights = combo_apply_weights * len(combo_apply_weights) / sum(combo_apply_weights)

    print(np.unique(combo_apply_weights)) #are these the sample weights that I need?

    return[combo_train_X, combo_train_y, combo_test_X, combo_test_y, combo_apply_weights] #5 values I need for training a model!

combo_men_women_data = comboMaleFemale(pooled_men_data, men_apply_weights, pooled_women_data, women_apply_weights) #this should combine all necessary data for training
#now I need to concatenate my pooled_male and pooled_female data [train_X, train_y, test_X, test_y, and train_X_weights]
print("done")

#how do I turn combo_men_women_data into a list of individual participant data -> give the men_pooled_data and women_pooled_data, grab based on indexes

def split_preprocessed_data(combo_men_women_data, part_windows, male_indexes, female_indexes):

    preprocessed_part_data = [] #will be in the order male participants, then female indexes. 

    start = 0

    combined_weights = combo_men_women_data[4]

    for part in male_indexes:
        end = start + len(part_windows[part][1])
        part_weights = combined_weights[start:end]

        preprocessed_part_data.append(  [part_windows[part][0], part_windows[part][1], part_windows[part][2], part_windows[part][3], part_weights] )

        start = end

    for part in female_indexes:
        end = start + len(part_windows[part][1])
        part_weights = combined_weights[start:end]

        preprocessed_part_data.append(  [part_windows[part][0], part_windows[part][1], part_windows[part][2], part_windows[part][3], part_weights] )

        start = end

    return preprocessed_part_data

#I train the fl clients with this!
preprocessed_part_data = split_preprocessed_data(combo_men_women_data, part_windows, binaryCNN.male_indexes, binaryCNN.female_indexes)

    

def pre_data_to_file(procecssed_data):

    foldername = './preprocessed_folder'

    for i in range (0, len(procecssed_data)):


        #I create 4 files for each participant
        fileName = '/participantTRAINX' + str(i) + '.npy'
        with open(foldername+fileName, 'wb') as f:
            np.save(f, procecssed_data[i][0])

        fileName = '/participantTRAINy' + str(i) + '.npy'
        with open(foldername+fileName, 'wb') as f:
            np.save(f, procecssed_data[i][1])

        fileName = '/participantTESTX' + str(i) + '.npy'
        with open(foldername+fileName, 'wb') as f:
            np.save(f, procecssed_data[i][2])

        fileName = '/participantTESTy' + str(i) + '.npy'
        with open(foldername+fileName, 'wb') as f:
            np.save(f, procecssed_data[i][3])

        fileName = '/participantWEIGHTS' + str(i) + '.npy'
        with open(foldername+fileName, 'wb') as f:
            np.save(f, procecssed_data[i][4])


def read_pre_data_file(participantID): #I STARTED OUT WITH 139 GB of storate -> will this continue to disappear with ray spilled IO objects? After 10 rounds... -> Why is there no change in first 7 rounds?
    #I open 4 files and grab their numpy contents
    foldername = './preprocessed_folder'

    with open(foldername + "/participantTRAINX" + str(participantID) + '.npy', 'rb' ) as f:
        train_X = np.load(f)

    with open(foldername + "/participantTRAINy" + str(participantID) + '.npy', 'rb' ) as f:
        train_y = np.load(f)

    with open(foldername + "/participantTESTX" + str(participantID) + '.npy', 'rb' ) as f:
        test_X = np.load(f)

    with open(foldername + "/participantTESTy" + str(participantID) + '.npy', 'rb' ) as f:
        test_y = np.load(f)
    
    with open(foldername + "/participantWEIGHTS" + str(participantID) + '.npy', 'rb' ) as f:
        weights = np.load(f)

    return [train_X, train_y, test_X, test_y, weights]



#I don't need pooled data for FL!

#I guess that I"m training with preprocessed_part_data!


def client_fn(cid: str) -> fl.client.Client:
    # Load model
    model = binaryCNN.sensor_activity_binary(n_timesteps=50, n_features=12, n_outputs=2) #as specified by David M

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

        binaryCNN.check_fairness(model, pooled_men_data, pooled_women_data)
        score = model.evaluate(pooled_windows[2], pooled_windows[3], verbose=0) #checking score with pooled test set

        # 5, 3, 7, 1, 4, 6, 8, 2, 0, 9
        #I need to append my server level model's loss in a file then I can disply it as a graph!
        with open("FederatedLoss/preprocessedFL.txt", "a") as f:
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

    #get participant data in windows

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
        ),
    )

if __name__ == "__main__":

    # pre_data_to_file([preprocessed_part_data[0]])

    # data_back = read_pre_data_file(0)
    #make sure this works
    #refer to the files in my fit and evaluate code

    print("Writing out participant data to files ...")
    pre_data_to_file(preprocessed_part_data)
    print("Done writing info to binary files!")


    main()
    
"""
#Run the same exact settings as FL 90 rounds, 35 epochs using some preprocessing


DI: 1.0318633381539122
EOP: 0.029505301887721047
Avg EP diff: 0.01961993913286588
SPD: 0.03450406400136352
-> Pooled Test accuracy: 0.9518072009086609
INFO flower 2022-05-15 17:06:50,657 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-15 17:06:50,658 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-15 17:07:29,652 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0308205252805884
EOP: 0.028568533737838075
Avg EP diff: 0.019151555057924395
SPD: 0.033886207961820736
-> Pooled Test accuracy: 0.9521616101264954
INFO flower 2022-05-15 17:07:30,058 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-15 17:07:30,058 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-15 17:08:08,668 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.030299908853679
EOP: 0.028100149662896645
Avg EP diff: 0.018463642512286712
SPD: 0.03326835192227795
-> Pooled Test accuracy: 0.9521616101264954
INFO flower 2022-05-15 17:08:09,078 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-15 17:08:09,078 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-15 17:08:49,070 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.030299908853679
EOP: 0.028100149662896645
Avg EP diff: 0.018463642512286712
SPD: 0.03326835192227795
-> Pooled Test accuracy: 0.9521616101264954
INFO flower 2022-05-15 17:08:49,483 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-15 17:08:49,484 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-15 17:09:29,092 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0301537590934728
EOP: 0.027950486766268212
Avg EP diff: 0.018388811063972495
SPD: 0.03316182501890852
-> Pooled Test accuracy: 0.9518072009086609
INFO flower 2022-05-15 17:09:29,507 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-15 17:09:29,507 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-15 17:10:09,130 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0301537590934728
EOP: 0.027950486766268212
Avg EP diff: 0.01775509750250228
SPD: 0.03274637009576764
-> Pooled Test accuracy: 0.9519844055175781
INFO flower 2022-05-15 17:10:09,537 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-15 17:10:09,538 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-15 17:10:48,636 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0296334794171629
EOP: 0.027482102691326782
Avg EP diff: 0.017520905465031565
SPD: 0.0324374420759963
-> Pooled Test accuracy: 0.9521616101264954
INFO flower 2022-05-15 17:10:49,038 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-15 17:10:49,039 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-15 17:11:27,844 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.030299908853679
EOP: 0.028100149662896645
Avg EP diff: 0.017829928950816497
SPD: 0.03285289699913718
-> Pooled Test accuracy: 0.9523387551307678
INFO flower 2022-05-15 17:11:28,244 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-15 17:11:28,244 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-15 17:12:06,586 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0300720524956568
EOP: 0.02793109138121197
Avg EP diff: 0.01774539980997416
SPD: 0.032757022786104595
-> Pooled Test accuracy: 0.9532246589660645
INFO flower 2022-05-15 17:12:06,991 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-15 17:12:06,991 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-15 17:12:47,213 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0300720524956568
EOP: 0.02793109138121197
Avg EP diff: 0.01774539980997416
SPD: 0.032757022786104595
-> Pooled Test accuracy: 0.9532246589660645
INFO flower 2022-05-15 17:12:47,651 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-15 17:12:47,651 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-15 17:13:28,793 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0302179487818492
EOP: 0.028080754277840292
Avg EP diff: 0.01782023125828832
SPD: 0.032863549689474136
-> Pooled Test accuracy: 0.9535790085792542
INFO flower 2022-05-15 17:13:29,214 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-15 17:13:29,214 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-15 17:14:08,737 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0310284488422126
EOP: 0.028848464146038477
Avg EP diff: 0.01775036568422044
SPD: 0.033076603496213
-> Pooled Test accuracy: 0.9539333581924438
INFO flower 2022-05-15 17:14:09,142 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-15 17:14:09,142 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-15 17:14:48,146 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.030654756535528
EOP: 0.02852974296772548
Avg EP diff: 0.017591005095063943
SPD: 0.03287420237981109
-> Pooled Test accuracy: 0.9544649124145508
INFO flower 2022-05-15 17:14:48,555 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-15 17:14:48,556 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-15 17:15:27,479 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0301363185292223
EOP: 0.02806135889278405
Avg EP diff: 0.017356813057593228
SPD: 0.03256527436003964
-> Pooled Test accuracy: 0.954642117023468
INFO flower 2022-05-15 17:15:27,882 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-15 17:15:27,882 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-15 17:16:06,916 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0301363185292223
EOP: 0.02806135889278405
Avg EP diff: 0.017356813057593228
SPD: 0.03256527436003964
-> Pooled Test accuracy: 0.954642117023468
INFO flower 2022-05-15 17:16:07,331 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-15 17:16:07,331 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-15 17:16:46,224 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0301363185292223
EOP: 0.02806135889278405
Avg EP diff: 0.016723099496123012
SPD: 0.03214981943689876
-> Pooled Test accuracy: 0.9548192620277405
INFO flower 2022-05-15 17:16:46,643 | server.py:209 | evaluate_round: no clients selected, cancel
INFO flower 2022-05-15 17:16:46,644 | server.py:182 | FL finished in 3619.2758999
INFO flower 2022-05-15 17:16:46,645 | app.py:149 | app_fit: losses_distributed []
INFO flower 2022-05-15 17:16:46,645 | app.py:150 | app_fit: metrics_distributed {}
INFO flower 2022-05-15 17:16:46,646 | app.py:151 | app_fit: losses_centralized []
INFO flower 2022-05-15 17:16:46,646 | app.py:152 | app_fit: metrics_centralized {}

# it got even worse! After switching c1 and c0 in assign_weight function!
DI: 1.0344424692177665
EOP: 0.031958094388944036
Avg EP diff: 0.020846335383477375
SPD: 0.03617653638426388
-> Pooled Test accuracy: 0.9537562131881714

# recent run ->
DI: 1.0351085622436633
EOP: 0.0325761413605139
Avg EP diff: 0.021155358869262307
SPD: 0.03659199130740476
-> Pooled Test accuracy: 0.9539333581924438
INFO flower 2022-05-17 21:42:37,358 | server.py:209 | evaluate_round: no clients selected, cancel
INFO flower 2022-05-17 21:42:37,359 | server.py:182 | FL finished in 3670.4992018000003

AZURE NOTES
1)
Username : david
Password: spring@2022final

#after this setup running state, running, ip address and username

#if I don't have time during mentoring I can have this running in the background
"""