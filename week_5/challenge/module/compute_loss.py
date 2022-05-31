"""
This is where the learning Agents for the Actor-Critic Reinforment Learning
are defined.
-------------------------------------------------------------------------------

"""
import numpy as np
import tensorflow as tf

from tensorflow import keras 

# ---------------------------------------------------------------------------------
# globals

from .env import ENV, SEED
np.random.seed(SEED)

# ---------------------------------------------------------------------------------
# functions


def update_loss(history:dict,losses_history:dict, loss):
    """ At this point of the game:
          * We have already taken an action with 'log_prob'.
          * Based on the action, we have recieved a total reward 'ret'

        We need to update:
          * the actor in order to predict an action that gets higher rewards 
            (than critic's estimations) with higher probability.
          * the critic so that it predicts a better estimate of future rewards
    
    Arguments:
        history (dict):
        losses_history (dict)
        loss (keras.losses) : instantiated loss for this environment
    
    Returns:
        losses_history (dict)
    """
    actor_losses, critic_losses = [], []
    hist = zip(history['actor'], history['critic'], history['returns'])

    compute_actor_loss  = lambda prob, c, r: -prob * (r-c)
    compute_critic_loss = lambda c, r: loss(tf.expand_dims(c, 0), tf.expand_dims(r, 0))

    for log_prob,  critic_value,  ret in hist:

        actor_losses.append( compute_actor_loss(prob=log_prob, c=critic_value, r=ret))
        critic_losses.append(compute_critic_loss(c=critic_value, r=ret)) 


    losses_history['actor']  += [sum(actor_losses)]
    losses_history['critic'] += [sum(critic_losses)]

    return losses_history



def update_gradient(tape, optimizer, 
        losses, models 
        ):
    """ In order to update the gradient descend there are two option:
      * Opt A: consider the actor and the critic together   to compute the GD.
      * Opt B: condider the actor and the critic separately to compute the GD.
    
    Opt B was chosen because of faster learning

    Arguments:
      loss (float)
      tape: current game
      optimizer (keras.optimizers): GD optimizer function chosen
    
    Returns:
      NoneType
    """    

    for loss, model in zip(losses, models):

      grads = tape.gradient(loss,  model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))   
    
    return tape, optimizer

# ---------------------------------------------------------------------------------
# end