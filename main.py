"""
Main script that runs the DQN learning algorithm 
(https://arxiv.org/pdf/1804.08617)

It features the standard DQN algorithm with a number 
of improvements from other researchers.
Namely:
    Distributed rollouts           (https://arxiv.org/pdf/1602.01783) *using python's 'threading' library
    N-step returns                 (https://arxiv.org/pdf/1602.01783)

The algorithm is trained on PTStephD's Rubix Cube environment. It will (hopefully)
learn how to solve a Rubix cube from any initial state in as few moves as possible.


===== Notes =====
Need to implement:  
    - Full code sweep and cleanup
    
Things to remember:
    - Currently training on openAI's 'CartPole-v0' gym environment with discrete actions
      and states (to ensure it's learning properly)
    

    
Temporary Notes:


@author: Kirkados (khovell@gmail.com) and PTStephD (stephane.magnan.11@gmail.com)

Code started: January 20, 2019
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
    