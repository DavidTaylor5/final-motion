from gc import callbacks
import os
import math

"""
This code is just for testing out the DPFL values. I should just be changing the optimizer and loss to be XX and XX
"""

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl
import tensorflow as tf
import numpy as np

import binaryCNN
import dataPreprocess

from typing import Optional, Tuple

import dpCNN

from sklearn.model_selection import train_test_split

from tensorflow.python.client import device_lib

import matplotlib.pyplot as plt

from preprocessedBinaryCNN import assignweights_fb_male, assignweights_fb_female, set_weights_participants, combine_part_window_weights

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


# necessary data below, window data for each participant
partData = dataPreprocess.getIndividualDatasets(NUM_CLIENTS) #SETUP TO ONLY LOAD IN 1 participant
dataPreprocess.normalizeParticipants(partData)
pooledData = dataPreprocess.getCentralDataset(partData)

part_windows = binaryCNN.participantWindows(partData, 50)

binaryCNN.participant_list_to_binary(part_windows)
pooled_windows = binaryCNN.poolWindows(part_windows, 50) #pooling all participants
pooled_men_data = binaryCNN.pool_by_attribute(binaryCNN.male_indexes, part_windows)
pooled_women_data = binaryCNN.pool_by_attribute(binaryCNN.female_indexes, part_windows)

#I need the correct data!

print(len(part_windows))

####################################### Preprocess method #############################################
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

reduce_men_train_2d = np.resize(reduce_men_train_label, ( len(reduce_men_train_label), 1) ) #make 2d
reduce_women_train_2d = np.resize(reduce_women_train_label, ( len(reduce_women_train_label), 1) ) #make 2d

#now it will be easier to assign weights with pooled_male and pooled_female then concatentate into combo_attributes -> use this with sample weights to train
men_apply_weights = np.apply_along_axis(func1d=assignweights_fb_male, axis=1, arr=reduce_men_train_2d) #at 82 diff weight -> either 1/3189(82 diff) or 1/6442 (start)
women_apply_weights = np.apply_along_axis(func1d=assignweights_fb_female, axis=1, arr=reduce_women_train_2d) #are these in the correct shape?

#for normalization I need the len of training data.
normalization_length = tot

#I also need a sum of the combined weights I have so far!
normalization_sum = sum(men_apply_weights) + sum(women_apply_weights)

# I use this data to train individuals with weights
part_windows_weights = set_weights_participants(part_windows)

# I use this data to train pooled/centralized models with weights
pooled_windows_weights = combine_part_window_weights(part_windows_weights)
#######################################################################################################


#not set up for k fold validation, withPreprocess is a boolean
def centralized_cnn(pooledData, withPreprocess):

    #print("Participant", counter,  "Local Training ->")
    david_cnn = dpCNN.sensor_activity_binary(n_timesteps=50, n_features=12, n_outputs=2)

    #my callback function is forcing clients to exit way to early? they don't produce predictive models? I would probably need a validation set? Takes 9 rounds before DI changes?
    my_early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30) #patience 7 instead of 3


    if withPreprocess:
        # also split the weights
        X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(pooledData[0], pooledData[1], pooledData[4], test_size=0.2, shuffle=True, random_state=RANDOM_SEED, stratify=pooledData[1])

        history = david_cnn.fit(X_train, y_train, batch_size=32, epochs=600, verbose=1, validation_data=(X_val, y_val), callbacks=[my_early_stop], sample_weight=weights_train) #instead of 20 epochs when should it stop?
        #callbacks=[my_early_stop]
    else:
        #20% of the 80% of the training data will be used as validation data! #I need to account for this split? Training with less data in centralized compared to FL
        X_train, X_val, y_train, y_val = train_test_split(pooledData[0], pooledData[1], test_size=0.2, shuffle=True, random_state=RANDOM_SEED, stratify=pooledData[1])

        history = david_cnn.fit(X_train, y_train, batch_size=512, epochs=600, verbose=1, validation_data=(X_val, y_val), callbacks=[my_early_stop]) #instead of 20 epochs when should it stop?
        #callbacks=[my_early_stop]





    loss_values = history.history['loss']
    loss_validation = history.history['val_loss']

    epoch_num = range(1, len(loss_values)+1)

    fig = plt.figure(figsize=(10, 5))
    
    plt.plot(epoch_num, loss_values, label="CL-DP-loss.py")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    if withPreprocess:
        fig.savefig('Pre-DP-CL-loss.jpg', bbox_inches='tight', dpi=150)
    else:
        fig.savefig('CL-DP-loss.jpg', bbox_inches='tight', dpi=150)
    

    fig2 = plt.figure(figsize=(10, 5))
    plt.plot(epoch_num, loss_validation, label="CL-DP-validation")
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()

    if withPreprocess:
        fig2.savefig('Pre-DP-CL-validation.jpg', bbox_inches='tight', dpi=150)
    else:
        fig2.savefig('CL-DP-validation.jpg', bbox_inches='tight', dpi=150)

    score = david_cnn.evaluate(pooledData[2], pooledData[3], verbose=0)
    #print('Test loss:', score[0]) 
    print('Pooled Test accuracy:', score[1])

    return david_cnn

if __name__ == "__main__":

    #without preprocessing weights
    print("without preprocessing weights")
    trained_model = centralized_cnn(pooled_windows, False)
    binaryCNN.check_fairness(trained_model, pooled_men_data, pooled_women_data)

    # #with preprocessing weights
    # print("with preprocessing weights")
    # trained_model = centralized_cnn(pooled_windows_weights, True)
    # binaryCNN.check_fairness(trained_model, pooled_men_data, pooled_women_data)

    """
    Now for federated learning this might be odd using DP, I can add participants weights first, then figure out batch and microbatch_sizes...
    """

    # my_model = dpCNN.sensor_activity_binary(n_timesteps=50, n_features=12, n_outputs=1) # 22590 / 18 and 
    # some_data = part_windows[0]

    # # testing 1d output values
    # one_dim = np.argmax(some_data[1], axis=1)
    # #one_dim = np.reshape(one_dim, (len(one_dim), 1) ) #reshaping is not helping!

    # one_dim_test = np.argmax(some_data[3], axis=1)



    # #dummy_data = np.ones( (996, 64, 64) )
    # my_model.fit(some_data[0], one_dim, batch_size=83, epochs=150, verbose=2) #why does this only allow microbatches of 1!

    # score = my_model.evaluate(some_data[2], one_dim_test, verbose=0)

    # print('Pooled Test accuracy:', score[1])
"""
Just dp + cl, giving much higher accuracy and is running for 231 epochs... not sure why (maybe I need to adjust learning rate?)
565/565 [==============================] - 5s 10ms/step - loss: 0.0310 - accuracy: 0.9892 - val_loss: 0.0567 - val_accuracy: 0.9849
Epoch 227/600
565/565 [==============================] - 5s 10ms/step - loss: 0.0299 - accuracy: 0.9908 - val_loss: 0.0561 - val_accuracy: 0.9845
Epoch 228/600
565/565 [==============================] - 5s 10ms/step - loss: 0.0279 - accuracy: 0.9903 - val_loss: 0.0605 - val_accuracy: 0.9843
Epoch 229/600
565/565 [==============================] - 5s 10ms/step - loss: 0.0270 - accuracy: 0.9903 - val_loss: 0.0590 - val_accuracy: 0.9847
Epoch 230/600
565/565 [==============================] - 5s 10ms/step - loss: 0.0282 - accuracy: 0.9905 - val_loss: 0.0561 - val_accuracy: 0.9852
Epoch 231/600
565/565 [==============================] - 5s 10ms/step - loss: 0.0323 - accuracy: 0.9881 - val_loss: 0.0543 - val_accuracy: 0.9858
Pooled Test accuracy: 0.9744861721992493


#Now to test for consistency... should be epochs 231 and accuracy .9744
#run this again!

Without preprocessing weights
Epoch 182/600
565/565 [==============================] - 5s 9ms/step - loss: 0.0367 - accuracy: 0.9870 - val_loss: 0.0540 - val_accuracy: 0.9845
Epoch 183/600
565/565 [==============================] - 6s 10ms/step - loss: 0.0365 - accuracy: 0.9868 - val_loss: 0.0536 - val_accuracy: 0.9852
Epoch 184/600
565/565 [==============================] - 5s 9ms/step - loss: 0.0358 - accuracy: 0.9875 - val_loss: 0.0528 - val_accuracy: 0.9843
Pooled Test accuracy: 0.9746633768081665
DI: 1.0036581699128053
EOP: 0.003550223915378181
Avg EP diff: 0.003792414293113333
SPD: 0.013060198353094132

With processing weights
Epoch 173/600
565/565 [==============================] - 5s 9ms/step - loss: 0.0415 - accuracy: 0.9867 - val_loss: 0.0595 - val_accuracy: 0.9812
Epoch 174/600
565/565 [==============================] - 5s 9ms/step - loss: 0.0415 - accuracy: 0.9867 - val_loss: 0.0583 - val_accuracy: 0.9821
Epoch 175/600
565/565 [==============================] - 5s 9ms/step - loss: 0.0410 - accuracy: 0.9858 - val_loss: 0.0527 - val_accuracy: 0.9834
Pooled Test accuracy: 0.9736002683639526
DI: 1.0054476041485394
EOP: 0.00527409731851558
Avg EP diff: 0.006829219133956403
SPD: 0.01274061764298573

Are these results reproducible?

without processing weights
Epoch 171/600
565/565 [==============================] - 6s 10ms/step - loss: 0.0379 - accuracy: 0.9874 - val_loss: 0.0543 - val_accuracy: 0.9854
Epoch 172/600
565/565 [==============================] - 5s 9ms/step - loss: 0.0362 - accuracy: 0.9879 - val_loss: 0.0541 - val_accuracy: 0.9849
Epoch 173/600
565/565 [==============================] - 5s 10ms/step - loss: 0.0382 - accuracy: 0.9866 - val_loss: 0.0530 - val_accuracy: 0.9856
Pooled Test accuracy: 0.9746633768081665
DI: 1.0060542341440064
EOP: 0.0058727489050292014
Avg EP diff: 0.005861117804272783
SPD: 0.013997635102745098

with processing weights
Epoch 170/600
565/565 [==============================] - 6s 10ms/step - loss: 0.0395 - accuracy: 0.9867 - val_loss: 0.0570 - val_accuracy: 0.9845
Epoch 171/600
565/565 [==============================] - 6s 10ms/step - loss: 0.0404 - accuracy: 0.9863 - val_loss: 0.0532 - val_accuracy: 0.9843
Pooled Test accuracy: 0.9720056653022766
DI: 1.0069425276321
EOP: 0.006698644928396336
Avg EP diff: 0.007087772430729811
SPD: 0.013965677031734347


DP + CL without preprocessing 1
Epoch 260/600
565/565 [==============================] - 2s 3ms/step - loss: 0.0265 - accuracy: 0.9907 - val_loss: 0.0545 - val_accuracy: 0.9852
Epoch 261/600
565/565 [==============================] - 2s 3ms/step - loss: 0.0285 - accuracy: 0.9895 - val_loss: 0.0552 - val_accuracy: 0.9849
Epoch 262/600
565/565 [==============================] - 2s 3ms/step - loss: 0.0291 - accuracy: 0.9897 - val_loss: 0.0538 - val_accuracy: 0.9858
Epoch 263/600
565/565 [==============================] - 2s 3ms/step - loss: 0.0272 - accuracy: 0.9902 - val_loss: 0.0535 - val_accuracy: 0.9852
Pooled Test accuracy: 0.9757264256477356
DI: 1.0034789342785233
EOP: 0.003381165633693617
Avg EP diff: 0.003254164644104083
SPD: 0.013273252159832993

DP + CL without preprocessing 2
Epoch 260/600
565/565 [==============================] - 2s 3ms/step - loss: 0.0265 - accuracy: 0.9907 - val_loss: 0.0545 - val_accuracy: 0.9852
Epoch 261/600
565/565 [==============================] - 2s 3ms/step - loss: 0.0285 - accuracy: 0.9895 - val_loss: 0.0552 - val_accuracy: 0.9849
Epoch 262/600
565/565 [==============================] - 2s 3ms/step - loss: 0.0291 - accuracy: 0.9897 - val_loss: 0.0538 - val_accuracy: 0.9858
Epoch 263/600
565/565 [==============================] - 2s 3ms/step - loss: 0.0272 - accuracy: 0.9902 - val_loss: 0.0535 - val_accuracy: 0.9852
Pooled Test accuracy: 0.9757264256477356
DI: 1.0034789342785233
EOP: 0.003381165633693617
Avg EP diff: 0.003254164644104083
SPD: 0.013273252159832993

DP + CL with preprocessing 1
Epoch 184/600
565/565 [==============================] - 2s 3ms/step - loss: 0.0381 - accuracy: 0.9870 - val_loss: 0.0624 - val_accuracy: 0.9803
Epoch 185/600
565/565 [==============================] - 2s 3ms/step - loss: 0.0364 - accuracy: 0.9875 - val_loss: 0.0547 - val_accuracy: 0.9841
Epoch 186/600
565/565 [==============================] - 2s 3ms/step - loss: 0.0350 - accuracy: 0.9876 - val_loss: 0.0593 - val_accuracy: 0.9832
Epoch 187/600
565/565 [==============================] - 2s 3ms/step - loss: 0.0390 - accuracy: 0.9872 - val_loss: 0.0644 - val_accuracy: 0.9801
Epoch 188/600
565/565 [==============================] - 2s 3ms/step - loss: 0.0363 - accuracy: 0.9874 - val_loss: 0.0584 - val_accuracy: 0.9832
Pooled Test accuracy: 0.9709426164627075
DI: 1.0068240806739268
EOP: 0.006568377416824256
Avg EP diff: 0.004574043080805678
SPD: 0.015499664440254413

DP + CL with preprocessing 2
Epoch 184/600
565/565 [==============================] - 2s 3ms/step - loss: 0.0381 - accuracy: 0.9870 - val_loss: 0.0624 - val_accuracy: 0.9803
Epoch 185/600
565/565 [==============================] - 2s 3ms/step - loss: 0.0364 - accuracy: 0.9875 - val_loss: 0.0547 - val_accuracy: 0.9841
Epoch 186/600
565/565 [==============================] - 2s 3ms/step - loss: 0.0350 - accuracy: 0.9876 - val_loss: 0.0593 - val_accuracy: 0.9832
Epoch 187/600
565/565 [==============================] - 2s 3ms/step - loss: 0.0390 - accuracy: 0.9872 - val_loss: 0.0644 - val_accuracy: 0.9801
Epoch 188/600
565/565 [==============================] - 2s 3ms/step - loss: 0.0363 - accuracy: 0.9874 - val_loss: 0.0584 - val_accuracy: 0.9832
Pooled Test accuracy: 0.9709426164627075
DI: 1.0068240806739268
EOP: 0.006568377416824256
Avg EP diff: 0.004574043080805678
SPD: 0.015499664440254413

Actual DP + CL 1
Epoch 29/600
565/565 [==============================] - 2s 4ms/step - loss: 2662.8315 - accuracy: 0.6002 - val_loss: 975.7964 - val_accuracy: 0.7191
Epoch 30/600
565/565 [==============================] - 2s 4ms/step - loss: 2871.5085 - accuracy: 0.6185 - val_loss: 913.4511 - val_accuracy: 0.7125
Epoch 31/600
565/565 [==============================] - 2s 4ms/step - loss: 2774.0593 - accuracy: 0.6149 - val_loss: 1054.1639 - val_accuracy: 0.7098
Pooled Test accuracy: 0.6956059336662292
DI: 1.0206543792139062
EOP: 0.010122943582588195
Avg EP diff: 0.015227341428018664
SPD: 0.005496788213863402

Actual DP + CL 2
Epoch 29/600
565/565 [==============================] - 2s 4ms/step - loss: 2662.8315 - accuracy: 0.6002 - val_loss: 975.7964 - val_accuracy: 0.7191
Epoch 30/600
565/565 [==============================] - 2s 4ms/step - loss: 2871.5085 - accuracy: 0.6185 - val_loss: 913.4511 - val_accuracy: 0.7125
Epoch 31/600
565/565 [==============================] - 2s 4ms/step - loss: 2774.0593 - accuracy: 0.6149 - val_loss: 1054.1639 - val_accuracy: 0.7098
Pooled Test accuracy: 0.6956059336662292
DI: 1.0206543792139062
EOP: 0.010122943582588195
Avg EP diff: 0.015227341428018664
SPD: 0.005496788213863402

Actual DP + CL + Pre 1
Epoch 27/600
565/565 [==============================] - 2s 4ms/step - loss: 2588.6169 - accuracy: 0.5836 - val_loss: 1035.6552 - val_accuracy: 0.6910
Epoch 28/600
565/565 [==============================] - 2s 4ms/step - loss: 2843.4690 - accuracy: 0.5827 - val_loss: 1024.3511 - val_accuracy: 0.7003
Epoch 29/600
565/565 [==============================] - 2s 4ms/step - loss: 2896.3669 - accuracy: 0.5506 - val_loss: 1085.1254 - val_accuracy: 0.7293
Epoch 30/600
565/565 [==============================] - 2s 4ms/step - loss: 3164.6316 - accuracy: 0.5745 - val_loss: 1111.6979 - val_accuracy: 0.7067
Epoch 31/600
565/565 [==============================] - 2s 4ms/step - loss: 3010.2771 - accuracy: 0.5735 - val_loss: 1355.4709 - val_accuracy: 0.6877
Pooled Test accuracy: 0.6624734401702881
DI: 1.026043979439764
EOP: 0.009915384014149986
Avg EP diff: 0.011178665970763552
SPD: 0.002077274615704172

Actual DP + CL + Pre 2
Epoch 27/600
565/565 [==============================] - 2s 4ms/step - loss: 2588.6169 - accuracy: 0.5836 - val_loss: 1035.6552 - val_accuracy: 0.6910
Epoch 28/600
565/565 [==============================] - 2s 4ms/step - loss: 2843.4690 - accuracy: 0.5827 - val_loss: 1024.3511 - val_accuracy: 0.7003
Epoch 29/600
565/565 [==============================] - 2s 4ms/step - loss: 2896.3669 - accuracy: 0.5506 - val_loss: 1085.1254 - val_accuracy: 0.7293
Epoch 30/600
565/565 [==============================] - 2s 4ms/step - loss: 3164.6316 - accuracy: 0.5745 - val_loss: 1111.6979 - val_accuracy: 0.7067
Epoch 31/600
565/565 [==============================] - 2s 4ms/step - loss: 3010.2771 - accuracy: 0.5735 - val_loss: 1355.4709 - val_accuracy: 0.6877
Pooled Test accuracy: 0.6624734401702881
DI: 1.026043979439764
EOP: 0.009915384014149986
Avg EP diff: 0.011178665970763552
SPD: 0.002077274615704172


NOW WITH NOSIE MULTIPLIER 0.2 (much less than 1.1!)


batch size 128
Epoch 26/600
142/142 [==============================] - 1s 9ms/step - loss: 420.1629 - accuracy: 0.6457 - val_loss: 188.9225 - val_accuracy: 0.6948
Epoch 27/600
142/142 [==============================] - 1s 9ms/step - loss: 430.2824 - accuracy: 0.6254 - val_loss: 257.7808 - val_accuracy: 0.5954
Epoch 28/600
142/142 [==============================] - 1s 9ms/step - loss: 418.1857 - accuracy: 0.6309 - val_loss: 286.4505 - val_accuracy: 0.5500
Epoch 29/600
142/142 [==============================] - 1s 9ms/step - loss: 432.3177 - accuracy: 0.5887 - val_loss: 370.2603 - val_accuracy: 0.4852
Epoch 30/600
142/142 [==============================] - 1s 9ms/step - loss: 396.5377 - accuracy: 0.5972 - val_loss: 190.8235 - val_accuracy: 0.6350
Epoch 31/600
142/142 [==============================] - 1s 9ms/step - loss: 373.7565 - accuracy: 0.6202 - val_loss: 242.9039 - val_accuracy: 0.5981
Pooled Test accuracy: 0.5986889004707336
DI: 1.0373009367681498
EOP: 0.014754387844014782
Avg EP diff: 0.036612551228345225
SPD: 0.029667742588390722

batch size 256
Epoch 28/600
71/71 [==============================] - 1s 15ms/step - loss: 165.9925 - accuracy: 0.6038 - val_loss: 76.3476 - val_accuracy: 0.6403
Epoch 29/600
71/71 [==============================] - 1s 15ms/step - loss: 169.0782 - accuracy: 0.6089 - val_loss: 83.5505 - val_accuracy: 0.6552
Epoch 30/600
71/71 [==============================] - 1s 15ms/step - loss: 172.4424 - accuracy: 0.6123 - val_loss: 64.1237 - val_accuracy: 0.7089
Epoch 31/600
71/71 [==============================] - 1s 15ms/step - loss: 170.8914 - accuracy: 0.6135 - val_loss: 59.7973 - val_accuracy: 0.7083
Pooled Test accuracy: 0.6640680432319641
DI: 1.0175764972347745
EOP: 0.009255361955517905
Avg EP diff: 0.04444064921847349
SPD: 0.021571697932312783

batch size 512
Epoch 28/600
36/36 [==============================] - 1s 25ms/step - loss: 91.4995 - accuracy: 0.6057 - val_loss: 44.0121 - val_accuracy: 0.6514
Epoch 29/600
36/36 [==============================] - 1s 26ms/step - loss: 93.1825 - accuracy: 0.5998 - val_loss: 39.3895 - val_accuracy: 0.6487
Epoch 30/600
36/36 [==============================] - 1s 26ms/step - loss: 101.3905 - accuracy: 0.6069 - val_loss: 53.3812 - val_accuracy: 0.6337
Epoch 31/600
36/36 [==============================] - 1s 25ms/step - loss: 104.6482 - accuracy: 0.5822 - val_loss: 61.4713 - val_accuracy: 0.6512
Epoch 32/600
36/36 [==============================] - 1s 26ms/step - loss: 87.5924 - accuracy: 0.6165 - val_loss: 32.2613 - val_accuracy: 0.6837
Pooled Test accuracy: 0.6904677748680115
DI: 1.0531168064734915
EOP: 0.04189634758845884
Avg EP diff: 0.028254856654520266
SPD: 0.03583565029348157
"""