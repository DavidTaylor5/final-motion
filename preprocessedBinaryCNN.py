from tabnanny import check, verbose
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import math

import tensorflow as tf
import os

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

from torch import int64

import dataPreprocess

from sklearn.model_selection import train_test_split

from binaryCNN import sensor_activity_binary

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



###################################  Sensitive Attributes   #######################################
#INDEXES OF SENTSITIVE ATTRIBUTES ZERO INDEXED
#the indexes of males "1"
male_indexes = [0, 1, 3, 5, 8, 10, 11, 12, 13, 14, 16, 19, 20, 21]
#the datasets of females "0"
female_indexes = [2, 4, 6, 7, 9, 15, 17, 18, 22, 23]


def pool_by_attribute(attr_indexes, participantData):

    attr_train_X = np.array([])
    attr_train_y = np.array([])

    attr_test_X = np.array([])
    attr_test_y = np.array([])

    for p in range(0, len(participantData)):

        # this is an index associated with attribute
        if(p in attr_indexes):

            attr_train_X = np.concatenate( [attr_train_X, participantData[p][0]] ) if len(attr_train_X) != 0 else participantData[p][0]
            attr_train_y = np.concatenate( [attr_train_y, participantData[p][1]] ) if len(attr_train_y) != 0 else participantData[p][1]

            attr_test_X = np.concatenate( [attr_test_X, participantData[p][2]] ) if len( attr_test_X ) != 0 else participantData[p][2]
            attr_test_y = np.concatenate( [attr_test_y, participantData[p][3]] ) if len(attr_test_y) != 0 else participantData[p][3]

    #return training data and labels that have been pooled by an attribute
    return( [attr_train_X, attr_train_y, attr_test_X, attr_test_y] ) 


def check_fairness(model, prot_data, unprot_data): #I might have to change these if I add more data to attribute pooling

    prot_test_pred = model.predict(prot_data[2])
    prot_test_truth = prot_data[3]

    #change back to single value
    prot_test_pred = np.argmax(prot_test_pred, axis=1)
    prot_test_truth = np.argmax(prot_test_truth, axis=1)

    tn, fp, fn, tp = confusion_matrix(y_true=prot_test_truth, y_pred=prot_test_pred, labels=[0, 1]).ravel()

    prot_tpr = tp / (tp + fn)
    prot_fpr = fp/(fp+tn)
    prot_dp = (tp+fp)/len(prot_test_truth)

    unprot_test_pred = model.predict(unprot_data[2])
    unprot_test_truth = unprot_data[3]

    #change back to single value
    unprot_test_pred = np.argmax(unprot_test_pred, axis=1)
    unprot_test_truth = np.argmax(unprot_test_truth, axis=1)


    tn, fp, fn, tp = confusion_matrix(y_true=unprot_test_truth, y_pred=unprot_test_pred, labels=[0, 1]).ravel()
    unprot_tpr = tp / (tp + fn)
    unprot_fpr = fp/(fp+tn)
    unprot_dp = (tp+fp)/len(unprot_test_truth)

    print("DI:", max(prot_tpr / unprot_tpr, unprot_tpr / prot_tpr))
    print("EOP:", abs(prot_tpr-unprot_tpr))
    print("Avg EP diff:", 0.5 * (abs(prot_tpr-unprot_tpr) + abs(prot_fpr-unprot_fpr)))
    print("SPD:", abs(prot_dp-unprot_dp))



################################### PREPROCESSING FUNCTIONS #######################################




# by calculating window size I won't have to contatenate windows/numpy arrays (slow)
def calcWindowAmount(participantData, windowSize):
    trainCounter = 0
    testCounter = 0

    start = 0
    end = start + windowSize

    #loop through training dataset
    while(end < len(participantData[0])):
        x = participantData[1][start]
        y = participantData[1][end-1]
        if participantData[1][start].all() == participantData[1][end-1].all():
            trainCounter +=1

        start = end
        end = end+ windowSize

    start = 0
    end = start + windowSize

    while(end < len(participantData[2])):

        if participantData[3][start].all() == participantData[3][end-1].all():
            testCounter +=1

        start = end
        end = end+ windowSize

    return [trainCounter, testCounter]
    
def windowData(participantData, windowSize):
    #go through the data looking at each 50 window -> creating a window for this if label is same for first and last
    #calculate the number of viable windows -> initialize a dataframe (3 dimensions) / numpy array of zeros (50 reduced -> 1)

    trainSize, testSize = calcWindowAmount(participantData, windowSize)
    trainSet = np.zeros( (trainSize, windowSize, 12) ) 
    trainLabels = np.zeros( (trainSize, 6), dtype= np.int64 )
    testSet = np.zeros( (testSize, windowSize, 12) )
    testLabels = np.zeros( (testSize, 6) , dtype= np.int64) 

    trainCounter = 0
    testCounter = 0

    start = 0
    end = start + windowSize

    #loop through training dataset
    while(end < len(participantData[0])):

        # if the label is consistent throughout the window
        if(participantData[1][start].all() == participantData[1][end-1].all()):
            windowArray = participantData[0].iloc[start:end, :].to_numpy().astype(np.float32)
            trainSet[trainCounter] = windowArray
            trainLabels[trainCounter] = participantData[1][start]
            trainCounter +=1
        start = end
        end = end+ windowSize

    start = 0
    end = start + windowSize

    while(end < len(participantData[2])):

        # if the label is consistent throughout the window
        if(participantData[3][start].all() == participantData[3][end-1].all()):
            windowArray = participantData[2].iloc[start:end, :].to_numpy().astype(np.float32)
            testSet[testCounter] = windowArray
            testLabels[testCounter] = participantData[3][start]
            testCounter +=1

        start = end
        end = end+ windowSize

    return [trainSet, trainLabels, testSet, testLabels]

def getPoolSize(windowDatasets, windowSize):

    pool_train_size = 0
    pool_test_size = 0

    for participant in windowDatasets:
        pool_train_size += len(participant[0])
        pool_test_size += len(participant[2])

    return [pool_train_size, pool_test_size]

def poolWindows(windowDatasets, windowSize):
    
    trainSize, testSize = getPoolSize(windowDatasets, windowSize)

    pool_train_windows = np.zeros( (trainSize, windowSize, 12) )
    pool_train_labels = np.zeros( (trainSize, 2), dtype=np.int64)
    pool_test_windows = np.zeros( (testSize, windowSize, 12) )
    pool_test_labels = np.zeros( (testSize, 2), dtype=np.int64)


    start_train = 0
    start_test = 0

    for participant in windowDatasets:
        xc = len(participant[0])
        pool_train_windows[start_train:(start_train + len(participant[0]))] = participant[0]
        pool_train_labels[start_train:(start_train + len(participant[1]))] = participant[1]
        pool_test_windows[start_test:(start_test + len(participant[2]))] = participant[2]
        pool_test_labels[start_test:(start_test + len(participant[3]))] = participant[3]

        start_train += len(participant[0])
        start_test += len(participant[2])
    
    return[pool_train_windows, pool_train_labels, pool_test_windows, pool_test_labels]

def participantWindows(participantData, windowSize):
    partWindows = []

    for participant in participantData:
        partWindows.append(windowData(participant, windowSize))
    
    return partWindows

########################## CNN / TRAINING FUNCTIONS #####################################################

def printAverages(acc_list, loss_list):

    sum_acc = 0
    sum_loss = 0

    for acc in acc_list:
        sum_acc += acc
    for loss in loss_list:
        sum_loss += loss
    print("Average Loss -> ", (sum_loss/len(loss_list)))
    print("Average Accuracy -> ", (sum_acc/len(acc_list)))


def individual_cnn(readyData):
    counter = 1
    avg_accuracy = []
    avg_loss = []
    for participant in readyData:
        #print("Participant", counter,  "Local Training ->")
        david_cnn = sensor_activity_binary(n_timesteps=50, n_features=12, n_outputs=2)

        david_cnn.fit(participant[0], participant[1], batch_size=32, epochs=20, verbose=0, validation_data=(participant[2], participant[3]))
        score = david_cnn.evaluate(participant[2], participant[3], verbose=0)
        #print('Test loss:', score[0]) 
        print('participant -> ' + str(counter) +' Test accuracy:', score[1])

        avg_accuracy.append(score[1])
        avg_loss.append(score[0])

        counter +=1
    
    printAverages(avg_accuracy, avg_loss)

#not set up for k fold validation #I might need to make sure that my training dataset is in the same order, weird splitting could be the reason I get different results.
def preprocessed_centralized_cnn(pooledData):
    # counter = 1
    # avg_accuracy = []
    # avg_loss = []

    #print("Participant", counter,  "Local Training ->")
    david_cnn = sensor_activity_binary(n_timesteps=50, n_features=12, n_outputs=2)

    #my callback function is forcing clients to exit way to early? they don't produce predictive models? I would probably need a validation set? Takes 9 rounds before DI changes?
    my_early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30) #patience 7 instead of 3

    #20% of the 80% of the training data will be used as validation data! -> (.16 of the data)
    #generate a shuffled validation set from the training set. # shouldn't need the weights associated to the validation data
    X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(pooledData[0], pooledData[1], pooledData[4], test_size=0.2, shuffle=True, random_state=RANDOM_SEED, stratify=pooledData[1])
    #sample_weight is how I add my preprocess weights
    history = david_cnn.fit(X_train, y_train, batch_size=32, epochs=600, verbose=1, validation_data=(X_val, y_val), callbacks=[my_early_stop], sample_weight=weights_train) #instead of 20 epochs when should it stop?
    #callbacks=[my_early_stop] #EPOCHS SHOULD BE 400?
    loss_values = history.history['loss']
    loss_validation = history.history['val_loss']
    epoch_num = range(1, len(loss_values)+1)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(epoch_num, loss_values, label="preprocessed CL training loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    fig.savefig('preprocessCL-train.jpg', bbox_inches='tight', dpi=150)

    fig2 = plt.figure(figsize=(10, 5))
    plt.plot(epoch_num, loss_validation, label="preprocessed CL validation loss")
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()

    fig2.savefig('preprocessCL-val.jpg', bbox_inches='tight', dpi=150)
    #get the loss curve

    score = david_cnn.evaluate(pooledData[2], pooledData[3], verbose=0)
    #print('Test loss:', score[0]) 
    print('Pooled Test accuracy:', score[1])

    return david_cnn

    # avg_accuracy.append(score[1])
    # avg_loss.append(score[0])

    # counter +=1
    
    #printAverages(avg_accuracy, avg_loss)

#based on  walking upstairs and walking downstairs are now -> [1, 0]
# all other activities are -> [0, 1]
#I'll transform the labels for training and testing for participants to represent new binary
# def transform_to_binary(pooled_windows):
#     for(x in pooled_windows[])
def produce_binary_labels(pooled_labels):
    binary_labels = np.zeros( (len(pooled_labels), 2), dtype=np.int64)

    for i in range(0, len(pooled_labels)):
        binary_labels[i] = to_binary_class(pooled_labels[i])

    return binary_labels

def to_binary_class(oneShot):
    index_val = np.argmax(oneShot)
    if(index_val == 1 or index_val == 5): #testing walking and jogging #sitting standing and walking test (gentle movements) #walking down stairs, sitting, walking
        return [1, 0]
    else:
        return [0, 1]

def participant_list_to_binary(part_windows):
    for part in part_windows:
        part[1] = produce_binary_labels(part[1])
        part[3] = produce_binary_labels(part[3])



#data stuff!

partData = dataPreprocess.getIndividualDatasets(24)
dataPreprocess.normalizeParticipants(partData)
pooledData = dataPreprocess.getCentralDataset(partData)

part_windows = participantWindows(partData, 50)
participant_list_to_binary(part_windows)
pooled_windows = poolWindows(part_windows, 50) #pooling all participants

# This code is for testing gender for discrimination
pooled_men_data = pool_by_attribute(male_indexes, part_windows)
pooled_women_data = pool_by_attribute(female_indexes, part_windows)


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

reduce_men_train_2d = np.resize(reduce_men_train_label, ( len(reduce_men_train_label), 1) ) #make 2d
reduce_women_train_2d = np.resize(reduce_women_train_label, ( len(reduce_women_train_label), 1) ) #make 2d

#now it will be easier to assign weights with pooled_male and pooled_female then concatentate into combo_attributes -> use this with sample weights to train
men_apply_weights = np.apply_along_axis(func1d=assignweights_fb_male, axis=1, arr=reduce_men_train_2d) #at 82 diff weight -> either 1/3189(82 diff) or 1/6442 (start)
women_apply_weights = np.apply_along_axis(func1d=assignweights_fb_female, axis=1, arr=reduce_women_train_2d) #are these in the correct shape?

#for normalization I need the len of training data.
normalization_length = tot
#I also need a sum of the combined weights I have so far!
normalization_sum = sum(men_apply_weights) + sum(women_apply_weights)

#men weights ->    min 0.00011741223435481978, max 0.00022512381809995497, after normalization min 0.6630855935189339, max 1.2713867627196667
#female weights -> min 0.0001552312946289972,  max 0.00031357792411414236, after normalization min 0.8766687364173797, max 1.7709313264348572
#right so men weights are lower than female weights -> makes sense

#to keep the distribution of data exactly the same as in both examples I must -> go through part windows and give each participant weights
def set_weights_participants(part_windows):
    ordered_with_weights = [] #participant data in the same order that includes weights

    for part in range(0, len(part_windows)):

        #resize one hot vectors to single values, then put into a 2d array for np.apply_along_axis
        train_labels = np.argmax(part_windows[part][1], axis=1)
        train_labels = np.resize(train_labels, (len(train_labels), 1) )
        weights = []

        if(part in male_indexes):
            #add male weights
            weights = np.apply_along_axis(func1d=assignweights_fb_male, axis=1, arr=train_labels)
        elif(part in female_indexes):
            #add female weights
            weights = np.apply_along_axis(func1d=assignweights_fb_female, axis=1, arr=train_labels)
        else:
            print("Error setting weights for each participant!")

        #normalize the weights before adding them!
        weights = weights * normalization_length / normalization_sum

        #I have weights in array that is same order as part_windows
        ordered_with_weights.append( [part_windows[part][0], part_windows[part][1], part_windows[part][2], part_windows[part][3], weights]  )
    
        #append to the order_with_weights array
    return ordered_with_weights

# I use this data to train with
part_windows_weights = set_weights_participants(part_windows)


""" Takes into account preprocessed information, attaches weights to each participant. """
def combine_part_window_weights(part_windows_weights):

    pooled_train_X = np.array([])
    pooled_train_y = np.array([])

    pooled_test_X = np.array([])
    pooled_test_y = np.array([])

    pooled_weights = np.array([])

    for part in part_windows_weights:
        pooled_train_X = np.concatenate( (pooled_train_X, part[0]), axis=0) if pooled_train_X.shape[0] != 0 else part[0]
        pooled_train_y = np.concatenate( (pooled_train_y, part[1]), axis=0) if pooled_train_y.shape[0] != 0 else part[1]

        pooled_test_X = np.concatenate( (pooled_test_X, part[2]), axis=0) if pooled_test_X.shape[0] != 0 else part[2]
        pooled_test_y = np.concatenate( (pooled_test_y, part[3]), axis=0) if pooled_test_y.shape[0] != 0 else part[3]

        pooled_weights = np.concatenate( (pooled_weights, part[4]), axis=0) if pooled_weights.shape[0] != 0 else part[4]

    return [ pooled_train_X, pooled_train_y, pooled_test_X, pooled_test_y, pooled_weights  ]

# pooled data with weights
pooled_windows_weights = combine_part_window_weights(part_windows_weights)


def comboMaleFemale(pooled_men_data, men_apply_weights, pooled_women_data, women_apply_weights):
    combo_train_X = np.concatenate( (pooled_men_data[0], pooled_women_data[0]), axis=0)
    combo_train_y = np.concatenate( (pooled_men_data[1], pooled_women_data[1]), axis=0)
    combo_test_X = np.concatenate( (pooled_men_data[2], pooled_women_data[2]), axis=0)
    combo_test_y = np.concatenate( (pooled_men_data[3], pooled_women_data[3]), axis=0)

    combo_apply_weights = np.concatenate( (men_apply_weights, women_apply_weights), axis=0 )

    #normalize the weights now that I have all of them combined
    combo_apply_weights = combo_apply_weights.ravel() # make it contiguous 1d
    combo_apply_weights = combo_apply_weights * len(combo_apply_weights) / sum(combo_apply_weights) #testing new weights

    print(np.unique(combo_apply_weights)) #are these the sample weights that I need?

    return[combo_train_X, combo_train_y, combo_test_X, combo_test_y, combo_apply_weights] #5 values I need for training a model!

combo_men_women_data = comboMaleFemale(pooled_men_data, men_apply_weights, pooled_women_data, women_apply_weights) #this should combine all necessary data for training
#now I need to concatenate my pooled_male and pooled_female data [train_X, train_y, test_X, test_y, and train_X_weights]
print("done") #maybe I should try testing the models with validation data and making sure that the data is in the same exact order for both experiements


if __name__ == "__main__":


    print("training with pooled_windows_weights")
    trained_model = preprocessed_centralized_cnn(pooled_windows_weights) # -> Pooled Test accuracy: 0.7921686768531799 #maybe the order matters?
    check_fairness(trained_model, pooled_men_data, pooled_women_data) #5644 = 3237(men) + 2407 (women)


"""
#this test shows lower accuracy and higher unfairness than without the preprocessing!
Epoch 203/400
706/706 [==============================] - 3s 4ms/step - loss: 0.1117 - accuracy: 0.9605 - val_loss: 0.1482 - val_accuracy: 0.9554
Pooled Test accuracy: 0.9553508162498474
DI: 1.0099456403108849
EOP: 0.009312100693891656
Avg EP diff: 0.0050010849539169075
SPD: 0.01814153164381671

Epoch 200/400
706/706 [==============================] - 2s 3ms/step - loss: 0.1117 - accuracy: 0.9605 - val_loss: 0.1402 - val_accuracy: 0.9578
Epoch 201/400
706/706 [==============================] - 2s 3ms/step - loss: 0.1122 - accuracy: 0.9610 - val_loss: 0.1348 - val_accuracy: 0.9600
Epoch 202/400
706/706 [==============================] - 2s 3ms/step - loss: 0.1111 - accuracy: 0.9619 - val_loss: 0.1235 - val_accuracy: 0.9647
Epoch 203/400
706/706 [==============================] - 2s 3ms/step - loss: 0.1112 - accuracy: 0.9606 - val_loss: 0.1453 - val_accuracy: 0.9562
Pooled Test accuracy: 0.956236720085144
DI: 1.0201187940885916
EOP: 0.01877125893418019
Avg EP diff: 0.010638105090395113
SPD: 0.025076433053167535

#got even worse when I switched assignweights values


Epoch 399/400
706/706 [==============================] - 2s 3ms/step - loss: 0.0808 - accuracy: 0.9732 - val_loss: 0.1242 - val_accuracy: 0.9688
Epoch 400/400
706/706 [==============================] - 2s 4ms/step - loss: 0.0833 - accuracy: 0.9717 - val_loss: 0.1413 - val_accuracy: 0.9646
Pooled Test accuracy: 0.9645641446113586
DI: 1.014572174151902
EOP: 0.013821382977799446
Avg EP diff: 0.007019377390095613
SPD: 0.021017758034791667

#accuracy is much better but DI still tanks for some reason!

#testing order of dataset
Results from combo_male_female (not great!)
Epoch 299/400
565/565 [==============================] - 2s 4ms/step - loss: 0.1077 - accuracy: 0.9617 - val_loss: 0.0867 - val_accuracy: 0.9697
Pooled Test accuracy: 0.9627923369407654
DI: 1.0172878493252349
EOP: 0.016332361634191495
Avg EP diff: 0.008274866718291637
SPD: 0.022658272346681163

Results from pooled_window_weights
Epoch 264/400
565/565 [==============================] - 2s 4ms/step - loss: 0.1076 - accuracy: 0.9640 - val_loss: 0.1105 - val_accuracy: 0.9655
Pooled Test accuracy: 0.9521616101264954
DI: 1.0154327855997238
EOP: 0.014334058006675532
Avg EP diff: 0.008419504626642784
SPD: 0.022040416307138377


Epoch 262/400
565/565 [==============================] - 5s 9ms/step - loss: 0.1101 - accuracy: 0.9614 - val_loss: 0.0979 - val_accuracy: 0.9695
Epoch 263/400
565/565 [==============================] - 5s 9ms/step - loss: 0.1067 - accuracy: 0.9625 - val_loss: 0.1079 - val_accuracy: 0.9653
Epoch 264/400
565/565 [==============================] - 5s 9ms/step - loss: 0.1079 - accuracy: 0.9638 - val_loss: 0.1111 - val_accuracy: 0.9648
Pooled Test accuracy: 0.9505669474601746
DI: 1.017847083237203
EOP: 0.01652631548475436
Avg EP diff: 0.00860819234934826
SPD: 0.02286067346308307
above stats
12959 9631
14959 7631
3189 6442
4442 8517
22590
[0.66308559 0.87666874 1.27138676 1.77093133]



TEN TESTS .... (using 10 epochs to make sure that results are reproducible!)
WHERE IS THE RANDOMNESS COMING FROM? CLEARLY FROM STARTING LOSS, IT MUST HAVE SOMEING TO DO WITH TRAIN DATA OR KERAS
565/565 [==============================] - 9s 11ms/step - loss: 0.7052 - accuracy: 0.4903 - val_loss: 0.6830 - val_accuracy: 0.6622
Epoch 2/10
565/565 [==============================] - 6s 10ms/step - loss: 0.6982 - accuracy: 0.5003 - val_loss: 0.7031 - val_accuracy: 0.3378
Epoch 3/10
565/565 [==============================] - 6s 10ms/step - loss: 0.6957 - accuracy: 0.4965 - val_loss: 0.7019 - val_accuracy: 0.3378
Epoch 4/10
565/565 [==============================] - 6s 10ms/step - loss: 0.6938 - accuracy: 0.5020 - val_loss: 0.6979 - val_accuracy: 0.3391
Epoch 5/10
565/565 [==============================] - 6s 10ms/step - loss: 0.6928 - accuracy: 0.5044 - val_loss: 0.6905 - val_accuracy: 0.6027
Epoch 6/10
565/565 [==============================] - 6s 10ms/step - loss: 0.6921 - accuracy: 0.5118 - val_loss: 0.6881 - val_accuracy: 0.6886
Epoch 7/10
565/565 [==============================] - 6s 10ms/step - loss: 0.6904 - accuracy: 0.5334 - val_loss: 0.7022 - val_accuracy: 0.3468
Epoch 8/10
565/565 [==============================] - 6s 10ms/step - loss: 0.6887 - accuracy: 0.5265 - val_loss: 0.6832 - val_accuracy: 0.6565
Epoch 9/10
565/565 [==============================] - 6s 10ms/step - loss: 0.6858 - accuracy: 0.5429 - val_loss: 0.6667 - val_accuracy: 0.6605
Epoch 10/10
565/565 [==============================] - 6s 10ms/step - loss: 0.6827 - accuracy: 0.5557 - val_loss: 0.6762 - val_accuracy: 0.5781
Pooled Test accuracy: 0.5458894371986389
DI: 1.0152045240477472
EOP: 0.00600475331675554
Avg EP diff: 0.07014266082888007    
SPD: 0.04371864114282059
training with pooled_windows_weights
Epoch 1/10
565/565 [==============================] - 6s 11ms/step - loss: 0.7083 - accuracy: 0.5004 - val_loss: 0.6881 - val_accuracy: 0.7023
Epoch 2/10
565/565 [==============================] - 5s 10ms/step - loss: 0.6981 - accuracy: 0.5116 - val_loss: 0.6966 - val_accuracy: 0.3659
Epoch 3/10
565/565 [==============================] - 6s 10ms/step - loss: 0.6945 - accuracy: 0.5160 - val_loss: 0.6861 - val_accuracy: 0.7087
Epoch 4/10
Epoch 5/10
565/565 [==============================] - 6s 10ms/step - loss: 0.6917 - accuracy: 0.5234 - val_loss: 0.6794 - val_accuracy: 0.7151
Epoch 6/10
565/565 [==============================] - 6s 10ms/step - loss: 0.6873 - accuracy: 0.5412 - val_loss: 0.6617 - val_accuracy: 0.6622
Epoch 7/10
565/565 [==============================] - 6s 10ms/step - loss: 0.6843 - accuracy: 0.5559 - val_loss: 0.6965 - val_accuracy: 0.5228
Epoch 8/10
565/565 [==============================] - 6s 10ms/step - loss: 0.6753 - accuracy: 0.5741 - val_loss: 0.6723 - val_accuracy: 0.5788
Epoch 9/10
565/565 [==============================] - 6s 10ms/step - loss: 0.6628 - accuracy: 0.5922 - val_loss: 0.6309 - val_accuracy: 0.6715
Epoch 10/10
565/565 [==============================] - 6s 10ms/step - loss: 0.6454 - accuracy: 0.6069 - val_loss: 0.6379 - val_accuracy: 0.5817
Pooled Test accuracy: 0.5563430190086365
DI: 1.0128664361919633
EOP: 0.0050097990117038416
Avg EP diff: 0.03507961674999151
SPD: 0.021827362500399516
training with pooled_windows_weights
Epoch 1/10
565/565 [==============================] - 6s 11ms/step - loss: 0.7049 - accuracy: 0.4932 - val_loss: 0.6852 - val_accuracy: 0.6622
Epoch 2/10
565/565 [==============================] - 6s 10ms/step - loss: 0.6967 - accuracy: 0.5107 - val_loss: 0.6943 - val_accuracy: 0.5058
Epoch 3/10
565/565 [==============================] - 6s 10ms/step - loss: 0.6922 - accuracy: 0.5240 - val_loss: 0.6882 - val_accuracy: 0.5748
Epoch 4/10
565/565 [==============================] - 6s 10ms/step - loss: 0.6881 - accuracy: 0.5378 - val_loss: 0.6937 - val_accuracy: 0.5376
Epoch 5/10
565/565 [==============================] - 6s 10ms/step - loss: 0.6794 - accuracy: 0.5681 - val_loss: 0.6665 - val_accuracy: 0.5806
Epoch 6/10
565/565 [==============================] - 6s 10ms/step - loss: 0.6671 - accuracy: 0.5829 - val_loss: 0.6333 - val_accuracy: 0.6866
Epoch 7/10
565/565 [==============================] - 6s 10ms/step - loss: 0.6511 - accuracy: 0.6032 - val_loss: 0.6727 - val_accuracy: 0.5744
Epoch 8/10
565/565 [==============================] - 5s 9ms/step - loss: 0.6296 - accuracy: 0.6146 - val_loss: 0.6372 - val_accuracy: 0.5803
Epoch 9/10
565/565 [==============================] - 5s 8ms/step - loss: 0.6143 - accuracy: 0.6195 - val_loss: 0.5964 - val_accuracy: 0.5945
Epoch 10/10
565/565 [==============================] - 6s 10ms/step - loss: 0.5983 - accuracy: 0.6300 - val_loss: 0.6131 - val_accuracy: 0.5861
Pooled Test accuracy: 0.5632529854774475
DI: 1.0219611101511403
EOP: 0.008645999484719669
Avg EP diff: 0.02195484321626026
SPD: 0.009630032064597949

Reproducible test 1
Epoch 170/600
565/565 [==============================] - 2s 3ms/step - loss: 0.1294 - accuracy: 0.9530 - val_loss: 0.1207 - val_accuracy: 0.9584
Epoch 171/600
565/565 [==============================] - 2s 3ms/step - loss: 0.1262 - accuracy: 0.9529 - val_loss: 0.1163 - val_accuracy: 0.9595
Epoch 172/600
565/565 [==============================] - 2s 3ms/step - loss: 0.1253 - accuracy: 0.9531 - val_loss: 0.1175 - val_accuracy: 0.9595
Epoch 173/600
565/565 [==============================] - 2s 3ms/step - loss: 0.1289 - accuracy: 0.9545 - val_loss: 0.1151 - val_accuracy: 0.9606
Epoch 174/600
565/565 [==============================] - 2s 3ms/step - loss: 0.1299 - accuracy: 0.9526 - val_loss: 0.1180 - val_accuracy: 0.9588
Pooled Test accuracy: 0.950921356678009
DI: 1.0213993284766236
EOP: 0.01978560862428813
Avg EP diff: 0.011145279935449084
SPD: 0.025651678331362504

Reproducible test 2
Epoch 170/600
565/565 [==============================] - 2s 3ms/step - loss: 0.1294 - accuracy: 0.9530 - val_loss: 0.1207 - val_accuracy: 0.9584
Epoch 171/600
565/565 [==============================] - 2s 3ms/step - loss: 0.1262 - accuracy: 0.9529 - val_loss: 0.1163 - val_accuracy: 0.9595
Epoch 172/600
565/565 [==============================] - 2s 3ms/step - loss: 0.1253 - accuracy: 0.9531 - val_loss: 0.1175 - val_accuracy: 0.9595
Epoch 173/600
565/565 [==============================] - 2s 3ms/step - loss: 0.1289 - accuracy: 0.9545 - val_loss: 0.1151 - val_accuracy: 0.9606
Epoch 174/600
565/565 [==============================] - 2s 3ms/step - loss: 0.1299 - accuracy: 0.9526 - val_loss: 0.1180 - val_accuracy: 0.9588
Pooled Test accuracy: 0.950921356678009
DI: 1.0213993284766236
EOP: 0.01978560862428813
Avg EP diff: 0.011145279935449084
SPD: 0.025651678331362504

This is also reproducible!
"""