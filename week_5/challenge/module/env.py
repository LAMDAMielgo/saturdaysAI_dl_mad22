"""
This is were the environment is defined
--------------------------------------
"""
import datetime
import gym 
import numpy as np
import tensorflow as tf

from collections import defaultdict
from tensorflow  import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses     import Huber

# ---------------------------------------------------------------------------------
# globals

SEED = 42
ENV  = gym.make("CartPole-v0")

NUM_INPUTS = ENV.observation_space.shape
N_ACTIONS  = ENV.action_space.n

TRAIN_PARAMS = {
    "n_hidden": 128,
    'learning_rate': 0.002,
    "delta": 1.0,
    "gamma": 0.985,
    "max_episode_steps": 200,
    "max_game_steps": 50000
}

# ---------------------------------------------------------------------------------
# functions

from .agents          import Actor, Critic
from .compute_rewards import play_episode, compute_game_return
from .compute_loss    import update_loss, update_gradient


def print_log(episode_count:int, num_steps:int, history:dict):

    dnow = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print_str = f"{dnow} - " +\
                "the running reward is {:<10}".format(round(history['running'][-1], 2)) +\
                "the episode reward is {:<10}".format(round(history['episode'][-1], 2)) +\
                "at episode {:<6}".format(episode_count) +\
                f"({num_steps} steps in total)"

    print(print_str)



def train_env():
    """ function that performs the training of the agents until the game is won
    """
    # Init agents, optimizer, loss, saving dict and counters
    actor_model  = Actor()
    critic_model = Critic()
    optimizer    = Adam(learning_rate=TRAIN_PARAMS['learning_rate'])
    huber_loss   = Huber(delta=TRAIN_PARAMS['delta'])
    
    reward_history = defaultdict(list, { k:[] for k in ('episode','running')})
    losses_hist    = defaultdict(list, { k:[] for k in ('actor','critic')})
    
    max_steps, gamma = TRAIN_PARAMS['max_episode_steps'], TRAIN_PARAMS['gamma']
    step, running_reward, episode_count = 0, 0, 0

    while step < TRAIN_PARAMS['max_game_steps']:

        state = ENV.reset()
        history = defaultdict(
            list, { k:[] for k in ('actor', 'critic', 'rewards', 'returns')}
        )
        
        with tf.GradientTape(persistent=True) as tape:

            # step 0 -- play episode by max_episode_steps
            episode_count += 1

            step, episode_reward, history = play_episode(
                step, history, state, actor_model, critic_model, max_steps
            )

            # step 1 -- update the running reward after taken an action
            running_reward, history = compute_game_return(
                running_reward,  episode_reward,  history, gamma 
            )

            reward_history['episode'] += [episode_reward]
            reward_history['running'] += [running_reward]

            # step 2 -- compute the losses for the game agents
            losses_hist = update_loss(history, losses_hist, loss=huber_loss)

            # step 3 -- update gradient desdend
            tape, optimizer = update_gradient(tape, optimizer, 
                losses= losses_hist.values(), 
                models= [actor_model, critic_model]
            ) # NOTE: probably if return NoneType would actually work 
            
        
        # print training
        if episode_count % 10 == 0:
            print_log(episode_count, step, reward_history)

        if running_reward > 195:
            print_log(episode_count, step, reward_history)
            print("Solved at episode {:<10}".format(episode_count))
    
        # end while loop    

    return reward_history, losses_hist, episode_count
    
    


# ---------------------------------------------------------------------------------
# end