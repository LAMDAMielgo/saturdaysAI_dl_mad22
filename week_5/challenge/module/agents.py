"""
This is where the learning Agents for the Actor-Critic Reinforment Learning
are defined.
-------------------------------------------------------------------------------

"""
import numpy as np
import tensorflow as tf

from tensorflow import keras 

from keras import Model
from keras.layers import Input, Dense 

# ---------------------------------------------------------------------------------
# globals

from .env import ENV, SEED, NUM_INPUTS, N_ACTIONS

np.random.seed(SEED)
NHIDDEN = 128

# ---------------------------------------------------------------------------------
# functions

def Actor() -> Model:
    """ The critic is mdae of three layers: 
      - common
      - action
    # note: elaborate on that
    """
    inputs = Input(shape=NUM_INPUTS)

    common = Dense(NHIDDEN,  activation="relu")(inputs)
    action = Dense(N_ACTIONS, activation="softmax")(common) 

    return Model(inputs=inputs, outputs=action)



def Critic() -> Model:
    """ The critic is mdae of three layers: 
      - common
      - critic
    # note: elaborate on that
    """
    inputs = Input(shape=NUM_INPUTS)

    common = Dense(NHIDDEN,  activation="relu")(inputs)
    common = Dense(NHIDDEN/2, activation="relu")(common)
    critic = Dense(1)(common) 

    return Model(inputs=inputs, outputs=critic)


# ---------------------------------------------------------------------------------
# end