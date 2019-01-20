"""
This code produces the actor/agent that will execute episodes.

The agent calculates the N-step returns and dumps data into the ReplayBuffer.
It collects its updated parameters after each episode.
"""

import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import time

#from environment import Environment
from settings import Settings
from collections import deque
from build_neural_networks import build_Q_network
from neural_network_utilities import get_variables, copy_variables

from pyvirtualdisplay import Display # for rendering


# This class builds one Agent. In use, many Agents will be created
class Agent:
    
    def __init__(self, sess, n_agent, replay_buffer, writer, filename):
        
        print("Initializing agent " + str(n_agent) + "...")
        
        self.n_agent = n_agent
        self.sess = sess
        self.replay_buffer = replay_buffer
        
        # Make an instance of the environment
        self.env = gym.make(Settings.ENVIRONMENT)
        
        # Record video if desired
        if self.n_agent == 1 and Settings.RECORD_VIDEO:
            # Generates a fake display that collects the frames during rendering.
            # This is needed because I need to run this on a Linux server (actually
            # the Ubuntu app on my Windows 10 PC) which does not have display functionality.
            try:
                display = Display(visible = 0, size = (1400,900))
                display.start()
            except:
                print("You must run on Linux if you want to record gym videos!")
                raise SystemExit
            
            # Start the gym's Monitor
            self.env = wrappers.Monitor(self.env, Settings.MODEL_SAVE_DIRECTORY + '/' + filename + '/videos', video_callable=lambda episode_number: episode_number%(Settings.VIDEO_RECORD_FREQUENCY*Settings.CHECK_GREEDY_PERFORMANCE_EVERY_NUM_EPISODES) == 0, force = True)
        
        # Build this agent's actor network and the operations to update it
        self.build_actor()
        self.build_actor_update_operation()
        
        # Establish the summary functions for TensorBoard logging.
        self.create_summary_functions()        
        self.writer = writer
        
        print("Agent %i initialized!" % self.n_agent)
    
    
    def create_summary_functions(self):
            
        # Logging the timesteps used for each episode for each agent
        self.timestep_number_placeholder      = tf.placeholder(tf.float32)            
        timestep_number_summary               = tf.summary.scalar("Agent_" + str(self.n_agent) + "/Number_of_timesteps", self.timestep_number_placeholder)
        self.regular_episode_summary          = tf.summary.merge([timestep_number_summary])
        
        # If this is agent 1, the agent who will also test performance, additionally log the reward
        if self.n_agent == 1:
            self.episode_reward_placeholder   = tf.placeholder(tf.float32)
            test_time_episode_reward_summary  = tf.summary.scalar("Test_agent/Episode_reward", self.episode_reward_placeholder)
            test_time_timestep_number_summary = tf.summary.scalar("Test_agent/Number_of_timesteps", self.timestep_number_placeholder)
            self.test_time_episode_summary    = tf.summary.merge([test_time_episode_reward_summary, test_time_timestep_number_summary])
            
            
    def build_actor(self):
        
        # Build the actor's policy neural network
        agent_name = 'agent_' + str(self.n_agent) # agent name 'agent_3', for example
        self.state_placeholder = tf.placeholder(dtype = tf.float32, shape = [None, *Settings.STATE_SIZE], name = 'state_placeholder') # the * lets Settings.STATE_SIZE be not restricted to only a scalar
                
        #############################
        #### Generate this Actor ####
        #############################
        self.policy = build_Q_network(self.state_placeholder, trainable = False, reuse = False, scope = agent_name)
        
        # Getting the non-trainable parameters from this actor, so that we will know where to place the updated parameters
        self.variables = get_variables(scope = agent_name, trainable = False) 

    def build_actor_update_operation(self):
        
        # Grab the most up-to-date parameters for the actor neural network. 
        # The parameters come from the learner thread who is constantly training
            
        # Get the up-to-date parameters from the actor that is being trained
        self.newly_trained_parameters = get_variables(scope = 'learned_actor', trainable = True) 

        # Make the operation to copy the up-to-date variables to this agent's policy
        self.update_actor_parameters = copy_variables(source_variables = self.newly_trained_parameters, destination_variables = self.variables, tau = 1)
        
            
    def run(self, stop_run_flag, starting_episode_number):
        
        # Runs the agent in its own environment
        # Runs for a specified number of episodes or until told to stop
        print("Starting to run agent %i at episode %i." % (self.n_agent, starting_episode_number[self.n_agent -1]))
        
        # Initializing parameters for agent network
        self.sess.run(self.update_actor_parameters)
        
        # Getting the starting episode number. If we are restarting a training
        # run that has crashed, the starting episode number will not be 1.
        episode_number = starting_episode_number[self.n_agent - 1] 
        start_time = time.time()
        
        # Creating the temporary memory space for calculating N-step returns
        n_step_memory = deque()
        
        # For all requested episodes or until user flags for a stop (via Ctrl + C)
        while episode_number <= Settings.NUMBER_OF_EPISODES and not stop_run_flag.is_set():              
            ####################################
            #### Getting this episode ready ####
            ####################################            
            # Resetting the environment for this episode
            state = self.env.reset()
            
            # Possibly normalize the state here
            
            # Clearing the N-step memory for this episode
            n_step_memory.clear()
            
            # Checking if this is a test time (when we run an agent in a noise-free environment to see how the training is going)
            test_time = (self.n_agent == 1) and (episode_number % Settings.CHECK_GREEDY_PERFORMANCE_EVERY_NUM_EPISODES == 0)
            
            # Calculating the noise scale for this episode. The noise scale 
            # allows for changing the amount of noise added to the actor during training.
            if test_time:
                # It's test time! Run this episode without exploration to evaluate performance.
                exploration_probability = 0
                
                # Additionally, if it's time to render, make a statement to the user
                if Settings.RECORD_VIDEO and episode_number % (Settings.CHECK_GREEDY_PERFORMANCE_EVERY_NUM_EPISODES*Settings.VIDEO_RECORD_FREQUENCY) == 0:
                    print("Rendering Actor %i at episode %i" % (self.n_agent, episode_number))
                
            else:
                # Regular training episode, use exploration.
                # Exploration probability starts at 1.0 but is reduced to 0.1 ovwe many episodes
                exploration_probability = np.max([Settings.NOISE_SCALE * Settings.NOISE_SCALE_DECAY ** episode_number, 0.1])
            
            # Resetting items for this episode
            episode_reward = 0
            timestep_number = 0
            done = False
            
            #####################
            ##### Debugging #####
            #####################
            # checking what the actor's parameters are and what the newly trained actor's parameters are
#            my_parameters, newly_trained_parameters = self.sess.run([self.variables, self.newly_trained_parameters])
#            #print("Printing RESULTS")
#            #self.sess.run(self.update_actor_parameters)
#            print("I'm agent %i" % self.n_agent)
#            print(my_parameters)
#            print("These are the newly trained parameters as seen by agent %i" % self.n_agent)
#            print(newly_trained_parameters)
            #########################
            ##### End Debugging #####
            #########################
            
            # Stepping through time until we reach the max time length or until we finish.
            while timestep_number < Settings.MAX_NUMBER_OF_TIMESTEPS and not done:
                ##############################
                ##### Running the Policy #####
                ##############################
                # Gets the expected value for each action
                action_values = self.sess.run(self.policy, feed_dict = {self.state_placeholder: state[None]})[0]
                
                # Extract the action number that we should perform
                action = np.argmax(action_values)
                
                # Checking for exploration
                if np.random.uniform() < exploration_probability:
                    # Let's explore! Instead of the selected action, we'll try a 
                    # completely random action instead.
                    action = np.random.randint(0, Settings.ACTION_SIZE)  

                ################################################
                #### Step the dynamics forward one timestep ####
                ################################################

                next_state, reward, done, _ = self.env.step(action)

                
                # Add reward we just received to running total
                episode_reward += reward
                
                # Normalize the state here if needed
                
                                
                # Store the data in this temporary buffer until we calculate the n-step return
                n_step_memory.append((state, action, reward))
                
                # If the n-step memory is full enough and we aren't testing performance
                if (len(n_step_memory) >= Settings.N_STEP_RETURN) and not test_time: 
                    # Grab the oldest data from the n-step memory
                    state_0, action_0, reward_0 = n_step_memory.popleft() 
                    n_step_reward = reward_0 # N-step reward starts with reward_0
                    discount_factor = Settings.DISCOUNT_FACTOR # initialize gamma
                    for i, (state_i, action_i, reward_i) in enumerate(n_step_memory):
                        # calculate the n-step reward
                        n_step_reward += reward_i*discount_factor
                        discount_factor *= Settings.DISCOUNT_FACTOR # for the next step, gamma**(i+1)
                        
                     # Dump data into large replay buffer
                    self.replay_buffer.add((state_0, action_0, n_step_reward, next_state, done, discount_factor))
                    
                # End of timestep -> next state becomes current state
                state = next_state
                timestep_number += 1
            
                # If it's time to render an episode
                if self.n_agent == 1 and Settings.RECORD_VIDEO and episode_number % (Settings.CHECK_GREEDY_PERFORMANCE_EVERY_NUM_EPISODES*Settings.VIDEO_RECORD_FREQUENCY) == 0:
                    self.env.render()
                    
                # If this episode is done, drain the N-step buffer, calculate 
                # returns, and dump in replay buffer unless it is test time.
                if (done or (timestep_number == Settings.MAX_NUMBER_OF_TIMESTEPS)) and not test_time:                    
                    # Episode has just finished, calculate the remaining N-step entries
                    while len(n_step_memory) > 0:
                        # Grab the oldest data from the n-step memory
                        state_0, action_0, reward_0 = n_step_memory.popleft()                         
                        # N-step reward starts with reward_0
                        n_step_reward = reward_0 
                        discount_factor = Settings.DISCOUNT_FACTOR # initialize gamma
                        for i, (state_i, action_i, reward_i) in enumerate(n_step_memory):                            
                            # Calculate the n-step reward
                            n_step_reward += reward_i*discount_factor
                            discount_factor *= Settings.DISCOUNT_FACTOR # for the next step, gamma**(i+1)
                        
                        
                        # dump data into large replay buffer
                        self.replay_buffer.add((state_0, action_0, n_step_reward, next_state, done, discount_factor))
                        
            
            ################################
            ####### Episode Complete #######
            ################################       
                 
            # Periodically update the agent with the learner's most recent version of the actor network parameters
            if episode_number % Settings.UPDATE_ACTORS_EVERY_NUM_EPISODES == 0:
                self.sess.run(self.update_actor_parameters)
            
            # Periodically display how long it's taking to run these episodes
            if episode_number % Settings.DISPLAY_ACTOR_PERFORMANCE_EVERY_NUM_EPISODES == 0:
                print("Actor " + str(self.n_agent) + " ran " + str(Settings.DISPLAY_ACTOR_PERFORMANCE_EVERY_NUM_EPISODES) + " episodes in %.1f s, and is now at episode %i" % (time.time() - start_time, episode_number))
                start_time = time.time()

            ###################################################
            ######## Log training data to tensorboard #########
            ###################################################
            # Write training data to tensorboard if we just checked the greedy performance of the agent (i.e., if it's test time).
            if test_time:                
                # Write test to tensorboard summary -> reward and number of timesteps
                feed_dict = {self.episode_reward_placeholder:  episode_reward,
                             self.timestep_number_placeholder: timestep_number}
                            
                summary = self.sess.run(self.test_time_episode_summary, feed_dict = feed_dict)
                self.writer.add_summary(summary, episode_number)
            else: 
                # it's not test time -> simply write the number of timesteps taken (reward is irrelevant)
                feed_dict = {self.timestep_number_placeholder: timestep_number}                
                summary = self.sess.run(self.regular_episode_summary, feed_dict = feed_dict)
                self.writer.add_summary(summary, episode_number)                
            
            ######################################
            ##### Done logging relevant data #####
            ######################################
            
            # Incrementing episode
            episode_number += 1
                
            
        self.env.close()
        print("Actor %i finished after running %i episodes!" % (self.n_agent, episode_number - 1))