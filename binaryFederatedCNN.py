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
    def __init__(self, model, cid, x_train, y_train, x_val, y_val) -> None:
        super().__init__()
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.cid = cid
        self.acc = []
        self.loss = []


    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        #my_early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5) #added early stopping

        self.model.fit(self.x_train, self.y_train, epochs=35, verbose=0) #I could potentially attach a callback function here to make early stopping?
        #callbacks=[my_early_stop]

        return self.model.get_weights(), len(self.x_train), {}
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)

        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=2)

        # print("client " + str(self.cid) + " -> is working!")
        
        # try: 
        #     f = open("fedBinary/participant" + self.cid + ".txt", "a")
        #     f.write("write something!")
        #     f.write("My cid -> " + self.cid + " Loss: " + str(loss) + " Accuracy: " + str(acc) + "\n")
        #     f.close()
        # except:
        #     print("STILL MESSING UP THE OPEN FILE STUFF!")

        # try:
        #     self.acc.append(acc)
        #     print("At least I got to print the accuracy tho! -> " + str(self.acc[-1]))
        # except:
        #     print("ERROR ON ACCESSING THE ACC LIST?")




        return loss, len(self.x_val), {"accuracy":acc}



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



def client_fn(cid: str) -> fl.client.Client:
    # Load model
    model = binaryCNN.sensor_activity_binary(n_timesteps=50, n_features=12, n_outputs=2) #as specified by David M

    (x_train, y_train, x_val, y_val) = part_windows[int(cid)] #rather than giving workers their data to hold, tell them to reference dataset?

    # Create and return client
    return FlwerClient(model, cid, x_train, y_train, x_val, y_val)

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
        )
    )

if __name__ == "__main__":

    # f = open("FlwrCNNResult/participant" + str(1) + ".txt", "a")
    # f.write("write something!")
    # f.write("My cid -> " + str(1) + " Loss: " + str(.5) + " Accuracy: " + str(.3) + "\n")
    # f.close()
    # print("done with evaluation!")

    
    main()
    
"""
checking to make sure that fl is reproducible ->
test 1)
DI: 1.079021185125488
EOP: 0.06721427268753455
Avg EP diff: 0.042546407952709656
SPD: 0.049353914331064286
-> Pooled Test accuracy: 0.9032601118087769
test 2) 
DI: 1.0852494205367167
EOP: 0.06971685632651403
Avg EP diff: 0.03972916669833208
SPD: 0.053231493613712155
-> Pooled Test accuracy: 0.8795180916786194

#federated learning is slightly different why? -> I'm fitting only 10 random clients a round

Results when fitting all 24 clients ->
with errors (IO spilling ) ->
DI: 1.0927120705076026
EOP: 0.07477760440941061
Avg EP diff: 0.04142340687555379
SPD: 0.057268863251414115
-> Pooled Test accuracy: 0.8793408870697021


#use the callback function for early stopping ->
#use the 10 fold cross validation

#try different splits
# try different attributes

#50 epochs each  for 5 rounds
DI: 1.0175468250910402
EOP: 0.014304241220693403
Avg EP diff: 0.06806453012502103
SPD: 0.037731829173457854
-> Pooled Test accuracy: 0.7840183973312378

# 50 epochs each for 10 rounds
DI: 1.0240811022027074
EOP: 0.020855249635974737
Avg EP diff: 0.019914351336657192
SPD: 0.01798174128876251

# FL 25 epochs, 10 rounds
#accuracy only increases around 7 round -> changes from 1.0 DI to 1.012 -> 1.0099 -> 1.032
DI: 1.0
EOP: 0.0
Avg EP diff: 0.0
SPD: 0.0
-> Pooled Test accuracy: 0.6649539470672607
INFO flower 2022-05-14 11:34:23,193 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 11:34:23,193 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 11:34:52,861 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.012172382728206
EOP: 0.0117732882125271
Avg EP diff: 0.014154823404647188
SPD: 0.013241294088822131
-> Pooled Test accuracy: 0.6527285575866699
INFO flower 2022-05-14 11:34:53,271 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 11:34:53,272 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 11:35:22,835 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0099672074274217
EOP: 0.008716054457609479
Avg EP diff: 0.047303334643138406
SPD: 0.03570781800943823
-> Pooled Test accuracy: 0.6589298248291016
INFO flower 2022-05-14 11:37:14,141 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 11:37:14,141 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 11:37:43,503 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0328417345366498
EOP: 0.02634645947377723
Avg EP diff: 0.08904058923304606
SPD: 0.03669851821077419
-> Pooled Test accuracy: 0.7034018635749817

#try increasing only epochs -> 35
change happens from round 4 to round 10
DI: 1.0066006600660067
EOP: 0.006557377049180357
Avg EP diff: 0.0037324090327571224
SPD: 0.004633920296570948
-> Pooled Test accuracy: 0.6626505851745605
INFO flower 2022-05-14 11:46:52,265 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 11:46:52,266 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 11:47:29,668 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0147020857129447
EOP: 0.013056278459832815
Avg EP diff: 0.04147151905091245
SPD: 0.033172477709245474
-> Pooled Test accuracy: 0.6614103317260742
INFO flower 2022-05-14 11:47:30,080 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 11:47:30,080 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 11:48:07,838 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0071456720493002
EOP: 0.005994331915829765
Avg EP diff: 0.07250841293598675
SPD: 0.055170283255036034
-> Pooled Test accuracy: 0.7416725754737854
INFO flower 2022-05-14 11:48:08,250 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 11:48:08,250 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 11:48:46,736 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0122027510834746
EOP: 0.009842144724310442
Avg EP diff: 0.042734565055472365
SPD: 0.026248228990231492
-> Pooled Test accuracy: 0.8049255609512329
INFO flower 2022-05-14 11:48:47,144 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 11:48:47,145 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 11:49:25,811 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0571585649811712
EOP: 0.046048117923941145
Avg EP diff: 0.027872945305270805
SPD: 0.04291968936754975
-> Pooled Test accuracy: 0.8454996347427368
INFO flower 2022-05-14 11:49:26,226 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 11:49:26,226 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 11:50:04,678 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0814706969276056
EOP: 0.06662662146866483
Avg EP diff: 0.03830826172791707
SPD: 0.05759909665185947
-> Pooled Test accuracy: 0.8752657771110535


#try increasing only rounds -> 20 rounds (25 epochs) rounds 9-20 show the most increases in accuracy however most fluctuation in DI
DI: 1.0
EOP: 0.0
Avg EP diff: 0.0
SPD: 0.0
-> Pooled Test accuracy: 0.6649539470672607
INFO flower 2022-05-14 11:58:22,817 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 11:58:22,817 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 11:58:51,953 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.012172382728206
EOP: 0.0117732882125271
Avg EP diff: 0.014154823404647188
SPD: 0.013241294088822131
-> Pooled Test accuracy: 0.6527285575866699
INFO flower 2022-05-14 11:58:52,368 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 11:58:52,368 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 11:59:21,182 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.010508454591102
EOP: 0.00918443853255102
Avg EP diff: 0.04753752668060918
SPD: 0.03601674602920968
-> Pooled Test accuracy: 0.6587526798248291
INFO flower 2022-05-14 11:59:21,595 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 11:59:21,595 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 11:59:51,054 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0322578782706595
EOP: 0.025878075398835687
Avg EP diff: 0.08898639024887856
SPD: 0.03711397313391496
-> Pooled Test accuracy: 0.7028703093528748
INFO flower 2022-05-14 11:59:51,467 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 11:59:51,468 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 12:00:20,906 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0710554583445622
EOP: 0.054191863780710614
Avg EP diff: 0.09822654129047814
SPD: 0.01733192717820886
-> Pooled Test accuracy: 0.7579730749130249
INFO flower 2022-05-14 12:00:21,319 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 12:00:21,319 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 12:00:50,824 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0263733426764279
EOP: 0.02070095500560154
Avg EP diff: 0.04673150151950965
SPD: 0.018045657430784123
-> Pooled Test accuracy: 0.7996101975440979
INFO flower 2022-05-14 12:00:51,240 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 12:00:51,240 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 12:01:20,574 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0455503120261431
EOP: 0.036888285476909366
Avg EP diff: 0.02879403083223049
SPD: 0.04019260064129193
-> Pooled Test accuracy: 0.8380581140518188
INFO flower 2022-05-14 12:01:21,005 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 12:01:21,005 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 12:01:50,839 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0716880743806094
EOP: 0.057921277895340184
Avg EP diff: 0.03357317658519513
SPD: 0.05115421899800798
-> Pooled Test accuracy: 0.86055988073349
INFO flower 2022-05-14 12:01:51,255 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 12:01:51,255 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 12:02:20,312 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0840891377624218
EOP: 0.06864982066505909
Avg EP diff: 0.03743367214133897
SPD: 0.0578654139102831
-> Pooled Test accuracy: 0.8796952366828918
INFO flower 2022-05-14 12:02:20,726 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 12:02:20,726 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 12:02:49,933 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.073778838735368
EOP: 0.06178761763880003
Avg EP diff: 0.03565586145327919
SPD: 0.0484164775814131
-> Pooled Test accuracy: 0.8954641819000244
INFO flower 2022-05-14 12:02:50,357 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 12:02:50,358 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 12:03:19,880 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0801497207610817
EOP: 0.06746091250944442
Avg EP diff: 0.03975993601154182
SPD: 0.051452494327442366
-> Pooled Test accuracy: 0.9002480506896973
INFO flower 2022-05-14 12:03:20,302 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 12:03:20,302 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 12:03:49,321 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0722551232301616
EOP: 0.061222256638577166
Avg EP diff: 0.03754804909244213
SPD: 0.0467120471275021
-> Pooled Test accuracy: 0.9018426537513733
INFO flower 2022-05-14 12:03:49,734 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 12:03:49,734 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 12:04:19,816 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0631768498720706
EOP: 0.05376690220962643
Avg EP diff: 0.0364489605254081
SPD: 0.039990199524890024
-> Pooled Test accuracy: 0.9020198583602905
INFO flower 2022-05-14 12:04:20,236 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 12:04:20,236 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 12:04:49,911 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0541781668557497
EOP: 0.04631154778067592
Avg EP diff: 0.02991270161018826
SPD: 0.03689026663683914
-> Pooled Test accuracy: 0.902197003364563

try increasing fl rounds again to be 30 rounds of 25 epochs
DI: 1.039672323686208
EOP: 0.03504543441320285
Avg EP diff: 0.023923684223593228
SPD: 0.03863730785209807
-> Pooled Test accuracy: 0.9231041669845581


try increasing fl rounds again to be 40 rounds of 25 epochs
DI: 1.0379568763544023
EOP: 0.0339922939529822
Avg EP diff: 0.02493826857128706
SPD: 0.03911667891726056
-> Pooled Test accuracy: 0.931077241897583

# 40 rounds of 35 epochs
DI: 1.0301891204822846
EOP: 0.02772874251323665
Avg EP diff: 0.01899043540085199
SPD: 0.03324704654160415
-> Pooled Test accuracy: 0.9425939321517944
INFO flower 2022-05-14 14:55:39,336 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 14:55:39,336 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 14:56:19,478 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0304839153511784
EOP: 0.028028068306493403
Avg EP diff: 0.019140098297480368
SPD: 0.033460100348343014
-> Pooled Test accuracy: 0.9433026313781738
INFO flower 2022-05-14 14:56:19,914 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 14:56:19,914 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 14:56:59,552 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0304839153511784
EOP: 0.028028068306493403
Avg EP diff: 0.019140098297480368
SPD: 0.033460100348343014
-> Pooled Test accuracy: 0.9433026313781738
INFO flower 2022-05-14 14:58:11,081 | server.py:209 | evaluate_round: no clients selected, cancel
INFO flower 2022-05-14 14:58:11,081 | server.py:182 | FL finished in 2070.9834481
INFO flower 2022-05-14 14:58:11,083 | app.py:149 | app_fit: losses_distributed []
INFO flower 2022-05-14 14:58:11,084 | app.py:150 | app_fit: metrics_distributed {}
INFO flower 2022-05-14 14:58:11,084 | app.py:151 | app_fit: losses_centralized []
INFO flower 2022-05-14 14:58:11,084 | app.py:152 | app_fit: metrics_centralized {}

50 rounds of 35 epochs for FL
DI: 1.0287635600763674
EOP: 0.02647325318504068
Avg EP diff: 0.018362690736754006
SPD: 0.03242678938565935
-> Pooled Test accuracy: 0.9434797763824463
INFO flower 2022-05-14 15:26:27,670 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 15:26:27,671 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 15:27:05,220 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0285363090033868
EOP: 0.026304194903356004
Avg EP diff: 0.017824441087744697
SPD: 0.032021987152855425
-> Pooled Test accuracy: 0.9441885352134705
INFO flower 2022-05-14 15:27:05,618 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 15:27:05,618 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 15:27:43,630 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0274921097049061
EOP: 0.025367426753473032
Avg EP diff: 0.017176063959499967
SPD: 0.03129760420994321
-> Pooled Test accuracy: 0.9448972344398499
INFO flower 2022-05-14 15:27:44,026 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 15:27:44,027 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 15:28:21,336 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0279361616683071
EOP: 0.02581641544335833
Avg EP diff: 0.017400558304442617
SPD: 0.03161718492005161
-> Pooled Test accuracy: 0.945960283279419
INFO flower 2022-05-14 15:28:21,737 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 15:28:21,737 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 15:28:59,737 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0275633302561373
EOP: 0.025497694265045112
Avg EP diff: 0.018508624838226437
SPD: 0.03224569364993135
-> Pooled Test accuracy: 0.9461374878883362
INFO flower 2022-05-14 15:29:00,142 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 15:29:00,142 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 15:29:37,601 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.028526134602315
EOP: 0.02641506702987184
Avg EP diff: 0.017699884097699372
SPD: 0.03204329253352933
-> Pooled Test accuracy: 0.9473777413368225
INFO flower 2022-05-14 15:29:38,001 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 15:29:38,002 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 15:30:15,918 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0268199281578951
EOP: 0.024860251908419007
Avg EP diff: 0.016922476536972954
SPD: 0.03100998157084578
-> Pooled Test accuracy: 0.9475549459457397
INFO flower 2022-05-14 15:30:16,312 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 15:30:16,313 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 15:30:53,707 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.027634191107615
EOP: 0.025627961776617192
Avg EP diff: 0.017306331471072047
SPD: 0.03153196339735598
-> Pooled Test accuracy: 0.9480864405632019
INFO flower 2022-05-14 15:30:54,109 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 15:30:54,110 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 15:31:32,038 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0284476319846951
EOP: 0.026395671644815488
Avg EP diff: 0.01705647284370098
SPD: 0.03163849030072552
-> Pooled Test accuracy: 0.9487951993942261
INFO flower 2022-05-14 15:31:32,433 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 15:31:32,434 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 15:32:09,434 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.027557605466725
EOP: 0.02560856639156095
Avg EP diff: 0.01666292021707371
SPD: 0.031127161164552053
-> Pooled Test accuracy: 0.9495038986206055

#70 rounds 35 epochs -> last 4 or so epochs show increase over centralized! why!
DI: 1.0304715044570931
EOP: 0.028601824324128722
Avg EP diff: 0.02006816561758595
SPD: 0.034589285524059155
-> Pooled Test accuracy: 0.9581856727600098
INFO flower 2022-05-14 17:03:48,356 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 17:03:48,356 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 17:04:26,120 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0304715044570931
EOP: 0.028601824324128722
Avg EP diff: 0.02006816561758595
SPD: 0.034589285524059155
-> Pooled Test accuracy: 0.9581856727600098
INFO flower 2022-05-14 17:04:26,535 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 17:04:26,536 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 17:05:04,313 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.032303522466609
EOP: 0.03030630234220988
Avg EP diff: 0.022007838696263715
SPD: 0.036453506333024466
-> Pooled Test accuracy: 0.958362877368927
INFO flower 2022-05-14 17:05:04,721 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 17:05:04,721 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 17:05:42,381 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0317884009484122
EOP: 0.02983791826726845
Avg EP diff: 0.02050621953585257
SPD: 0.03531366846697126
-> Pooled Test accuracy: 0.9588943719863892

#90 rounds 35 epochs -> safely fluctuates between 96.04 and 96.06% accuracy
DI: 1.0304715044570931
EOP: 0.028601824324128722
Avg EP diff: 0.02006816561758595
SPD: 0.034589285524059155
-> Pooled Test accuracy: 0.9581856727600098
INFO flower 2022-05-14 17:51:05,228 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 17:51:05,228 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 17:51:42,923 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0304715044570931
EOP: 0.028601824324128722
Avg EP diff: 0.020701879179056165
SPD: 0.03500474044719992
-> Pooled Test accuracy: 0.9580085277557373
INFO flower 2022-05-14 17:51:43,332 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 17:51:43,333 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 17:52:21,742 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0317884009484122
EOP: 0.02983791826726845
Avg EP diff: 0.021773646658793
SPD: 0.03614457831325302
-> Pooled Test accuracy: 0.9585400223731995
INFO flower 2022-05-14 17:52:22,155 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 17:52:22,155 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 17:53:00,332 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.032303522466609
EOP: 0.03030630234220988
Avg EP diff: 0.020740411573323285
SPD: 0.03562259648674271
-> Pooled Test accuracy: 0.9587172269821167
INFO flower 2022-05-14 17:53:00,744 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 17:53:00,745 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 17:53:38,467 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0306156734245966
EOP: 0.028751487220757044
Avg EP diff: 0.019963004012596867
SPD: 0.034589285524059155
-> Pooled Test accuracy: 0.9588943719863892
INFO flower 2022-05-14 17:53:38,876 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 17:53:38,877 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 17:54:17,252 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0316447454899234
EOP: 0.029688255370640015
Avg EP diff: 0.019977667579371382
SPD: 0.03489821354383049
-> Pooled Test accuracy: 0.958362877368927
INFO flower 2022-05-14 17:54:17,664 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 17:54:17,664 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 17:54:56,145 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0306156734245966
EOP: 0.028751487220757044
Avg EP diff: 0.01887556994295968
SPD: 0.03386490258114683
-> Pooled Test accuracy: 0.9588943719863892
INFO flower 2022-05-14 17:54:56,561 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 17:54:56,562 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 17:55:34,081 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0306156734245966
EOP: 0.028751487220757044
Avg EP diff: 0.01887556994295968
SPD: 0.03386490258114683
-> Pooled Test accuracy: 0.9588943719863892
INFO flower 2022-05-14 17:55:34,490 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 17:55:34,491 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 17:56:12,318 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0306156734245966
EOP: 0.028751487220757044
Avg EP diff: 0.018241856381489466
SPD: 0.03344944765800606
-> Pooled Test accuracy: 0.9590715765953064
INFO flower 2022-05-14 17:56:12,754 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 17:56:12,755 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 17:56:53,725 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0295886523250206
EOP: 0.027814719070874183
Avg EP diff: 0.017773472306548035
SPD: 0.032831591618463274
-> Pooled Test accuracy: 0.9594259262084961
INFO flower 2022-05-14 17:56:54,139 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 17:56:54,139 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 17:57:31,724 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0290759089722692
EOP: 0.027346334995932753
Avg EP diff: 0.01753928026907732
SPD: 0.03252266359869194
-> Pooled Test accuracy: 0.9596031308174133
INFO flower 2022-05-14 17:57:32,135 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 17:57:32,136 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 17:58:10,511 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0290759089722692
EOP: 0.027346334995932753
Avg EP diff: 0.01753928026907732
SPD: 0.03252266359869194
-> Pooled Test accuracy: 0.9596031308174133
INFO flower 2022-05-14 17:58:10,923 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 17:58:10,923 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 17:58:48,493 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0292204855652654
EOP: 0.027495997892561186
Avg EP diff: 0.017614111717391537
SPD: 0.032629190502061256
-> Pooled Test accuracy: 0.959957480430603
INFO flower 2022-05-14 17:58:48,903 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 17:58:48,903 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 17:59:26,789 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0287084355724467
EOP: 0.027027613817619645
Avg EP diff: 0.017379919679920766
SPD: 0.03232026248228992
-> Pooled Test accuracy: 0.9601346850395203
INFO flower 2022-05-14 17:59:27,198 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 17:59:27,198 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 18:00:04,971 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0281968948287508
EOP: 0.026559229742678214
Avg EP diff: 0.01669200713428308
SPD: 0.03170240644274713
-> Pooled Test accuracy: 0.9601346850395203
INFO flower 2022-05-14 18:00:05,383 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 18:00:05,384 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 18:00:43,755 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0281968948287508
EOP: 0.026559229742678214
Avg EP diff: 0.016058293572812864
SPD: 0.031286951519606365
-> Pooled Test accuracy: 0.9603118300437927
INFO flower 2022-05-14 18:00:44,164 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 18:00:44,164 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 18:01:22,148 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0281968948287508
EOP: 0.026559229742678214
Avg EP diff: 0.01542458001134265
SPD: 0.030871496596465486
-> Pooled Test accuracy: 0.96048903465271
INFO flower 2022-05-14 18:01:22,559 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 18:01:22,560 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 18:02:01,217 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0281968948287508
EOP: 0.026559229742678214
Avg EP diff: 0.01542458001134265
SPD: 0.030871496596465486
-> Pooled Test accuracy: 0.96048903465271
INFO flower 2022-05-14 18:02:01,629 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 18:02:01,630 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 18:02:39,458 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0281968948287508
EOP: 0.026559229742678214
Avg EP diff: 0.01542458001134265
SPD: 0.030871496596465486
-> Pooled Test accuracy: 0.96048903465271
INFO flower 2022-05-14 18:02:39,873 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 18:02:39,874 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 18:03:18,564 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0281968948287508
EOP: 0.026559229742678214
Avg EP diff: 0.01542458001134265
SPD: 0.030871496596465486
-> Pooled Test accuracy: 0.96048903465271
INFO flower 2022-05-14 18:03:18,981 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 18:03:18,981 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 18:03:56,961 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.02768586257486
EOP: 0.026090845667736784
Avg EP diff: 0.014736667465704966
SPD: 0.0302536405569227
-> Pooled Test accuracy: 0.96048903465271
INFO flower 2022-05-14 18:03:57,368 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 18:03:57,370 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 18:04:36,075 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0271753380529647
EOP: 0.025622461592795354
Avg EP diff: 0.014502475428234251
SPD: 0.029944712537151252
-> Pooled Test accuracy: 0.9606661796569824
INFO flower 2022-05-14 18:04:36,487 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 18:04:36,488 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 18:05:14,429 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0271753380529647
EOP: 0.025622461592795354
Avg EP diff: 0.014048754920067281
SPD: 0.029635784517379915
-> Pooled Test accuracy: 0.96048903465271
INFO flower 2022-05-14 18:05:14,837 | server.py:209 | evaluate_round: no clients selected, cancel
DEBUG flower 2022-05-14 18:05:14,838 | server.py:265 | fit_round: strategy sampled 24 clients (out of 24)
DEBUG flower 2022-05-14 18:05:53,443 | server.py:277 | fit_round received 24 results and 0 failures
DI: 1.0266653205067617
EOP: 0.025154077517853923
Avg EP diff: 0.013814562882596566
SPD: 0.029326856497608467
-> Pooled Test accuracy: 0.9606661796569824
INFO flower 2022-05-14 18:05:53,889 | server.py:209 | evaluate_round: no clients selected, cancel
INFO flower 2022-05-14 18:05:53,891 | server.py:182 | FL finished in 3493.3885524
INFO flower 2022-05-14 18:05:53,891 | app.py:149 | app_fit: losses_distributed []
INFO flower 2022-05-14 18:05:53,892 | app.py:150 | app_fit: metrics_distributed {}
INFO flower 2022-05-14 18:05:53,892 | app.py:151 | app_fit: losses_centralized []
INFO flower 2022-05-14 18:05:53,892 | app.py:152 | app_fit: metrics_centralized {}








#try increasing both rounds and epochs -> 20 rounds at 35 epochs

#if I don't have time during mentoring I can have this running in the background
"""
