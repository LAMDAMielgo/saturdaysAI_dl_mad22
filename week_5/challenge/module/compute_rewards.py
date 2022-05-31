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

from .env import ENV, SEED, N_ACTIONS
np.random.seed(SEED)

# ---------------------------------------------------------------------------------
# functions

def play_episode(
        step:int,
        history:dict,        
        state,
        actor_model,
        critic_model,
        max_steps:int
    ) -> tuple:
    """ <forward propagation> of the learning algorithm.

    The Actor-Critic method learns by playing a number of episodes.
    For each episode, the agent makes a numbers of steps.
    Each step is defined by its state, action is made randomly from
    an array of possible actions and the critic estimation of the future rewards.
    After taken an action, we collect a history with all the critics, 
    the actions (log(action)) and the reward.
        
    Args:
        episode_count (int): number of episoded played
        max_steps (int): number of max steps to make in an episode, default at 250.
        history (dict):  place where to store game history

    Returns:  
        step ():
        episode_reward (float): reward after an action is taken
        history (dict): hashmap where history for action, critic and rewards are stored.
    """
    episode_reward = 0

    for timestep in range(1, max_steps):

        # compute state as an array(1,4) with the values that define it
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)                                     

        # Predict action probabilities and estimated future rewards from env state.
        # action_probs = array(1,2), critic_value = array(1,1) 
        action_probs = actor_model(state)
        critic_value = critic_model(state)                                                                                    

        # Sample action from action probability distribution.
        # Randomly chosen action given its probability (action_probs)
        action = np.random.choice(N_ACTIONS, p=np.squeeze(action_probs))

        # Once the step is made, we compute the new state, modifying the env
        state, reward, done, _ = ENV.step(action) 
        episode_reward += reward
        step += 1
        
        # Update history
        history['critic'] += [critic_value[0, 0]]
        history['actor']  += [tf.math.log(action_probs[0, action])]
        history['rewards']+= [reward]

        # When the done condition is satisfied (win or lose) finish the episode.
        if done:
            break
        
    return step, episode_reward, history



def compute_game_return(
    running_reward: float, 
    episode_reward:float, 
    history:dict,
    gamma:int
    ) -> tuple:
    """ Compute the COST FUNCTION of the learning algorith.

    After the agent has played an episode, we need to compute the normalized
    return of the game, in order to know how far is the agent from winning.

    Arguments:
      running_reward (int): 
      episode_reward (float): 
      history (dict)

    Returns:
      running_reward (int)
      history (dict)
    """
    def cost_function(episode_reward, running_reward):
        return 0.05 * episode_reward + (1 - 0.05) * running_reward

    def update_dis_sum(discounted_sum, r, gamma=gamma):
        return r + gamma*discounted_sum
    
    def normalize(ret:list):
        """ scalar tensors are not iterable, therefore normalization has to 
        be done without iteration"""
        ret = np.array(ret)
        ret = (ret - np.mean(ret)) / (np.std(ret))
        return ret.tolist()

    
    discounted_sum, returns = 0, []
    running_reward = cost_function(episode_reward, running_reward)

    # Computed the return sum, multiplying each step by its gamma factor
    for r in history['rewards'][::-1]:  
        discounted_sum =  update_dis_sum(discounted_sum, r=r)
        returns.insert(0, discounted_sum)

    # Normalize the return array.
    history['returns'] = normalize(returns)

    return running_reward, history

# ---------------------------------------------------------------------------------
# end
