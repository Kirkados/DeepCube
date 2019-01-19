"""
Main script that runs the D4PG learning algorithm 
(https://arxiv.org/pdf/1804.08617)

It features the standard DDPG algorithm with a number 
of improvements from other researchers.
Namely:
    Distributed rollouts           (https://arxiv.org/pdf/1602.01783)
    A distributional critic        (http://arxiv.org/abs/1707.06887)
    N-step returns                 (https://arxiv.org/pdf/1602.01783)
****Prioritized experience replay  (http://arxiv.org/abs/1511.05952) **** NOT YET IMPLEMENTED

This implementation does not use the 
ApeX framework (https://arxiv.org/abs/1803.00933) as the original authors did.
Instead, it uses python's 'threading' library.


===== Notes =====
Need to implement:  
    - Prioritized experience replay (http://arxiv.org/abs/1511.05952)
       Note: I may not even implement this because some results show it doesn't
             really help much.
    - My own dynamics
    
Things to remember:
    - Seed is set but it does not fully work. Results are similar but not identical.
       More similar than not setting the seed. Strange.
    - CPU is used to run everything as it was noticed to be faster than using the 
       GPU (maybe due to all the copying). This could be re-investigated for a 
       larger mini_batch_size/neuron count (was done with batch = 128, neurons = 40)
    - The choise of gaussian or uniform noise is in the Settings. I prefer uniform but other use gaussian.
    - If attempting to render episodes using gym, you must run this on Linux or a WSL image!
    - I'm not satisfied with the critic loss function -> loss doesn't go to 0 but it's consistent with everyone's implementation
    - The action side and the state side of the critic are added together after the relu, not before (as msinto93 does)
    - Make sure to update Qmin and Qmax when changing the environment
    - State is now normalized by dividing by the maximum state possible. Occurs in the agent.
    
Temporary Notes:
    - REPLAY_BUFFER_START_TRAINING_FULLNESS has been set to 0 (previously 10000)
    - Reward signal is divided by 100.0. I do not agree with this, but I'm doing it in an effort to mimic msinto93's solution.
    - MIN_Q is -20.0 instead of -2000.0 to test the effect of scaling the reward signal 
    - NUMBER_OF_EPISODES = 50 to see if critic overfits

@author: Kirk Hovell

Special thanks to:
    - msinto93 (https://github.com/msinto93)
    - SuReLI   (https://github.com/SuReLI)
    for publishing their codes!

Code started: October 15, 2018
"""

# Importing libraries & other classes
# Others'
from shutil import copyfile
import os
import time
import threading
import random
import datetime
import tensorflow as tf
import numpy as np

# My own
from agent import Agent
from learner import QNetwork
from replay_buffer import ReplayBuffer
from settings import Settings

import saver

#%%
##########################
##### SETTING UP RUN #####
##########################

tf.reset_default_graph() # clearing tensorflow graph

# Set random seeds
tf.set_random_seed(Settings.RANDOM_SEED)
np.random.seed(Settings.RANDOM_SEED)
random.seed(Settings.RANDOM_SEED)


############################################################
##### New run or continuing a partially completed one? #####
############################################################
if Settings.RESUME_TRAINING: # If we're continuing a run
    filename                  = Settings.RUN_NAME # Reuse the name too
    starting_episode_number   = np.zeros(Settings.NUMBER_OF_ACTORS, dtype = np.int8) # initializing
    starting_iteration_number = 0 # initializing

    try:
        # Grab the tensorboard path
        old_tensorboard_filename = [i for i in os.listdir(Settings.MODEL_SAVE_DIRECTORY + filename) if i.endswith(Settings.TENSORBOARD_FILE_EXTENSION)][0]
        
        # For every entry in the tensorboard file
        for tensorboard_entry in tf.train.summary_iterator(Settings.MODEL_SAVE_DIRECTORY + filename + "/" + old_tensorboard_filename):
            # Search each one for the Loss value so you can find the final iteration number
            for tensorboard_value in tensorboard_entry.summary.value:
                if tensorboard_value.tag == 'Logging_Learning/Loss':
                    starting_iteration_number = max(tensorboard_entry.step, starting_iteration_number)
        
            # Search also for the actors so you can find what episode they were on
            for agent_number in range(Settings.NUMBER_OF_ACTORS):
                for tensorboard_value in tensorboard_entry.summary.value:
                    if tensorboard_value.tag == 'Agent_' + str(agent_number + 1) + '/Number_of_timesteps':
                        starting_episode_number[agent_number] = max(tensorboard_entry.step, starting_episode_number[agent_number])

    except:
        print("Couldn't load in old tensorboard file! Quitting run.")
        raise SystemExit

else: # Otherwise, we are starting from scratch
    # Generate a filename using Settings.RUN_NAME with the current timestamp
    filename                  = Settings.RUN_NAME + '-{:%Y-%m-%d %H-%M}'.format(datetime.datetime.now())
    starting_episode_number   = np.ones(Settings.NUMBER_OF_ACTORS) # All actors start at episode 1
    starting_iteration_number = 1
    
    
# Generate writer that will log tensorboard scalars & graph
writer = tf.summary.FileWriter(Settings.MODEL_SAVE_DIRECTORY + filename, filename_suffix = Settings.TENSORBOARD_FILE_EXTENSION)

# Saving a copy of the settings.py file in the 'Settings.MODEL_SAVE_DIRECTORY' directory, for reference
copyfile('settings.py', Settings.MODEL_SAVE_DIRECTORY + filename + '/settings.py')


#######################################
##### Starting tensorflow session #####
#######################################
with tf.Session() as sess:
    
    ##############################
    ##### Initializing items #####
    ##############################    
    
    # Initializing saver class (for loading & saving data)
    saver = saver.Saver(sess, filename) 
    
    # Initializing replay buffer
    replay_buffer = ReplayBuffer(Settings.PRIORITIZED_REPLAY_BUFFER)
    
    # Initializing thread list
    threads = []
    
    # Event() is used to communicate with the threads while they're running.
    # In this case, it is used to signal to the threads when it is time to stop
    # gracefully.
    stop_run_flag = threading.Event() 

    # Generating the critic (which is a Q-network) and assigning it to a thread
    if Settings.USE_GPU_WHEN_AVAILABLE: 
        # Allow GPU use when appropriate
        critic = QNetwork(sess, saver, replay_buffer, writer)
    else:
        # Forcing to the CPU only
        with tf.device('/device:CPU:0'): 
            critic = QNetwork(sess, saver, replay_buffer, writer)            
    threads.append(threading.Thread(target = critic.run, args = (stop_run_flag, starting_iteration_number)))
     
    # Generating the actors and placing them into their own threads
    for i in range(Settings.NUMBER_OF_ACTORS):
        if Settings.USE_GPU_WHEN_AVAILABLE: 
            # Allow GPU use when appropriate
            actor = Agent(sess, i+1, replay_buffer, writer, filename)
        else:            
            with tf.device('/device:CPU:0'):
                # Forcing to the CPU only
                actor = Agent(sess, i+1, replay_buffer, writer, filename)                
        threads.append(threading.Thread(target = actor.run, args = (stop_run_flag, starting_episode_number)))
    
    # If desired, try to load in partially-trained parameters
    if Settings.RESUME_TRAINING == True:
        if not saver.load(): 
            # If loading was not successful -> quit program
            print("Could not load in parameters... quitting program")
            raise SystemExit
    else: 
        # Don't try to load in parameters, just initialize them instead        
        # Initialize saver
        saver.initialize()
        # Initialize tensorflow variables
        sess.run(tf.global_variables_initializer())

    #############################################
    ##### STARTING EXECUTION OF ALL THREADS #####
    #############################################
    #                                           #
    #                                           #
    for each_thread in threads:                 #
    #                                           #
        each_thread.start()                     #
    #                                           #
    #                                           #
    #############################################
    ############## THREADS STARTED ##############
    #############################################

    # Now that the tensorflow computation graph has been fully constructed, write it to file
    writer.add_graph(sess.graph)    
    print('Done starting!')
    
    
    ####################################################
    ##### Waiting until all threads have completed #####
    ####################################################    
    print("Running until threads finish or until you press Ctrl + C")
    try:
        while True:
            time.sleep(0.5)
            # If all threads have ended on their own
            if not any(each_thread.is_alive() for each_thread in threads): 
                print("All threads ended naturally.")
                break
    except KeyboardInterrupt: # if someone pressed Ctrl + C
        print("Interrupted by user!")
        print("Stopping all the threads!!")        
        # Gracefully stop all threads, ending episodes and saving data
        stop_run_flag.set() 
        # Join threads (suspends program until threads finish)
        for each_thread in threads:
            each_thread.join()
    print("Done closing! Goodbye :)")
    