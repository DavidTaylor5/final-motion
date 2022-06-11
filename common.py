import keras
from typing import List, Tuple
import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent

XY = Tuple[np.ndarray, np.ndarray]
XYList = List[XY]
PartitionedDataset = List[Tuple[XY, XY]]


def compute_epsilon(
        epochs: int, num_train_examples: int, batch_size: int, noise_multiplier: float
) -> float:
    """Computes epsilon value for given hyperparameters.

    Based on
    github.com/tensorflow/privacy/blob/master/tutorials/mnist_dpsgd_tutorial_keras.py
    """
    if noise_multiplier == 0.0:
        return float("inf")
    steps = epochs * num_train_examples // batch_size
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    sampling_probability = batch_size / num_train_examples
    rdp = compute_rdp(
        q=sampling_probability,
        noise_multiplier=noise_multiplier,
        steps=steps,
        orders=orders,
    )
    # Delta is set to approximate 1 / (number of training points).
    return get_privacy_spent(orders, rdp, target_delta=1 / num_train_examples)[0]



def create_mlp_model() -> tf.keras.Model:
    """Returns a sequential keras CNN Model."""
    '''
    return tf.keras.Sequential(
        [


            tf.keras.layers.Dense(1,
                                  activation = 'sigmoid',
                                  kernel_regularizer=L2(l2=1),
                                  input_dim = 3875)
        ]
    )
    '''
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(3875,)))
    model.add(tf.keras.layers.Dense(units=2,kernel_initializer='glorot_uniform', activation='sigmoid')) #,kernel_regularizer=l2(0.) #kernel_initializer
    return model




def load_data(idx: int):
    #print(str(idx))

    X_train = np.load("/Users/sikha/0FAIRFL/Data/strat/X_train_"+str(idx)+".npy",allow_pickle=True)
    y_train = np.load("/Users/sikha/0FAIRFL/Data/strat/y_train_"+str(idx)+".npy",allow_pickle=True)
    y_train_2 = []
    for label in y_train:
        if label == 1:
            x = [0,1]
        else:
            x = [1,0]
        y_train_2.append(np.array(x))
    y_train_2 = np.array(y_train_2)
    X_test = np.load("/Users/sikha/0FAIRFL/Data/strat/X_test_"+str(idx)+".npy",allow_pickle=True)
    y_test = np.load("/Users/sikha/0FAIRFL/Data/strat/y_test_"+str(idx)+".npy",allow_pickle=True)
    print(X_train.shape, X_test.shape)
    return (X_train, y_train_2), (X_test, y_test)

    