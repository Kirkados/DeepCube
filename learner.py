"""
This script builds the QNetwork and generates all the operations needed to train it. 
For stability, target networks are used. The most recent version of the QNetwork
is copied to each agent periodically.

Training:
    The critic is trained using supervised learning to minimize the 
    least mean squares loss between the Q network and the target value 
    y = r_t + gamma * Q_target(next_state, Action(next_state))
        
"""

import tensorflow as tf
import numpy as np
import time

from build_neural_networks import build_Q_network
from neural_network_utilities import get_variables, copy_variables, l2_regularization

from settings import Settings

class QNetwork:
    
    def __init__(self, sess, saver, replay_buffer, writer):      
        """
        When initialized, create the trainable actor, the critic, and their respective target networks.
        """
        print('Q-Network initializing...')
        
        self.sess = sess
        self.saver = saver
        self.replay_buffer = replay_buffer
                
        with tf.variable_scope("Preparing_placeholders"):
            
            # Creating placeholders for training data
            self.state_placeholder           = tf.placeholder(dtype = tf.float32, shape = [None, *Settings.STATE_SIZE], name = 'state_placeholder') # the '*' unpacks the STATE_SIZE (incase it's pixels of higher dimension)
            self.action_placeholder          = tf.placeholder(dtype = tf.int32  , shape = [None], name = 'action_placeholder')
            self.reward_placeholder          = tf.placeholder(dtype = tf.float32, shape = [None], name = 'reward_placeholder')
            self.next_state_placeholder      = tf.placeholder(dtype = tf.float32, shape = [None, *Settings.STATE_SIZE], name = 'next_state_placeholder')
            self.done_placeholder            = tf.placeholder(dtype = tf.float32, shape = [None], name = 'not_done_placeholder')
            self.discount_factor_placeholder = tf.placeholder(dtype = tf.float32, shape = [None], name = 'discount_factor_placeholder')

            # Changing to column vectors
            self.reward_placeholder          = tf.expand_dims(self.reward_placeholder, 1)
            self.done_placeholder            = tf.expand_dims(self.done_placeholder, 1)
            self.discount_factor_placeholder = tf.expand_dims(self.discount_factor_placeholder, 1)
        
            # Getting the batch size
            self.batch_size = tf.shape(self.reward_placeholder)[0]

        ####################################################
        #### Build the networks and training operations ####
        ####################################################
        self.build_models()
        self.build_targets()
        self.build_target_parameter_update_operations()
        self.build_network_training_operations()
        
        # Create summary functions
        self.writer = writer
        self.create_summary_functions()   
        
        print('Q-Network created!')
        
    def create_summary_functions(self):
        
        # Creates the operation that, when run, will log data to tensorboard
        with tf.variable_scope("Logging_Learning"):
            
            # The loss is the only logged item
            self.iteration_loss_placeholder = tf.placeholder(tf.float32)        
            iteration_loss_summary = tf.summary.scalar("Loss", self.iteration_loss_placeholder)        
            self.iteration_summary = tf.summary.merge([iteration_loss_summary])
        
    def build_models(self):        
        ###################################
        ### Build the learned Q-network ###
        ###################################
        self.q_network = build_Q_network(state = self.state_placeholder, 
                                         trainable = True, 
                                         reuse = False, 
                                         scope = 'learned_critic')

        
    def build_targets(self):
        # Build the target network
        # Returns the next action values because it uses the next state.
        self.target_q_network = build_Q_network(state = self.next_state_placeholder,
                                                trainable = False,
                                                reuse = False,
                                                scope = 'target_critic')
        
        
    def build_target_parameter_update_operations(self):
        # Update the slowly-changing target networks according to tau
        
        with tf.variable_scope("Update_Target_Networks"):
                    
            # Grab parameters from the main networks
            self.critic_parameters = get_variables(scope = 'learned_critic', trainable = True)
            
            self.parameters = self.critic_parameters
            
            # Grab parameters from target networks
            self.target_critic_parameters = get_variables(scope = 'target_critic', trainable = False)
            
            self.target_parameters = self.target_critic_parameters
            
            # Operation to initialize the target networks to be identical to the main networks
            self.initialize_target_network_parameters = copy_variables(source_variables = self.parameters, 
                                                                       destination_variables = self.target_parameters, 
                                                                       tau = 1)
            
            # Updating target networks at rate Settings.TARGET_NETWORK_TAU 
            self.update_target_network_parameters     = copy_variables(source_variables = self.parameters, 
                                                                       destination_variables = self.target_parameters, 
                                                                       tau = Settings.TARGET_NETWORK_TAU)
            
    def build_network_training_operations(self):
        # Builds the operations that are used to train the critic
        
        with tf.variable_scope("Train_Critic"): # grouping tensorboard graph
            
            #########################################
            ###### Generating Target Q-Values  ######
            #########################################
            """
            Bellman projection. Q(this_state) = this_reward + discount_factor*Q(next_state) 
            This recursive function calculates what the Q-network 'should' have output,
            based on the information we learned from the most recent batch of training data.
            """
            q_value_targets = self.reward_placeholder + (self.discount_factor_placeholder)*(1.0 - self.done_placeholder)*tf.reduce_max(self.target_q_network, axis = 1) # [batch_size, 1]
            
            ####################################
            ##### Grabbing Current Q-Value #####
            ####################################
            # Building one-hot from action
            current_action_one_hots = tf.one_hot(self.action_placeholder, Settings.ACTION_SIZE) # [batch_size, action_size]
            # Finding the corresponding Q-value for this action
            current_q_values = tf.reduce_sum(self.q_network * current_action_one_hots, axis = 1) # [batch_size, 1]
            
            ###########################################
            ##### Generating Critic Loss Function #####
            ###########################################
            """
            We now know what the Q-value was for the selected action, and we 
            know how that panned out. Let's make a loss function that updates 
            the Q-value to have an output closer to what was actually observed.
            We'll sum the squared errors between the q-values and the q_value targets.
            """
            self.critic_loss = tf.reduce_mean(tf.square(current_q_values - q_value_targets))
            
            # Optional L2 Regularization
            if Settings.L2_REGULARIZATION:
                # Penalize the q-network for having large weights -> L2 Regularization
                self.critic_loss += l2_regularization(self.critic_parameters) 

            
            ##############################################################################
            ##### Develop the Operation that Trains the Q-network with Gradient Descent #####
            ##############################################################################
            critic_trainer             = tf.train.AdamOptimizer(Settings.CRITIC_LEARNING_RATE)
            self.train_critic_one_step = critic_trainer.minimize(self.critic_loss) # RUN THIS TO TRAIN THE CRITIC ONE STEP
            
            
    def run(self, stop_run_flag, starting_training_iteration):
        # Continuously train the actor and the critic, by applying gradient
        # descent to batches of data sampled from the replay buffer
        
        # Initializing the counter of training iterations
        self.total_training_iterations = starting_training_iteration
        start_time = time.time()
        
        # Initialize the target networks to be identical to the main networks
        self.sess.run(self.initialize_target_network_parameters) 
           
        ###############################
        ##### Start Training Loop #####
        ###############################
        while self.total_training_iterations < Settings.MAX_TRAINING_ITERATIONS and not stop_run_flag.is_set():
            
            # If we don't have enough data yet to train OR we want to wait before we start to train
            if (self.replay_buffer.how_filled() < Settings.MINI_BATCH_SIZE) or (self.replay_buffer.how_filled() < Settings.REPLAY_BUFFER_START_TRAINING_FULLNESS):
                
                continue # Skip this training iteration. Wait for more training data.

            # Sample mini-batch of data from the replay buffer
            sampled_batch = np.asarray(self.replay_buffer.sample())
            
            # Assemble this data into a dictionary that will be used for training
            training_data_dict = {self.state_placeholder: np.stack(sampled_batch[:, 0]),
                                  self.action_placeholder: np.stack(sampled_batch[:, 1]),
                                  self.reward_placeholder: np.reshape(sampled_batch[:, 2], [-1, 1]),
                                  self.next_state_placeholder: np.stack(sampled_batch[:, 3]),
                                  self.done_placeholder: np.reshape(sampled_batch[:, 4], [-1, 1]),
                                  self.discount_factor_placeholder: np.reshape(sampled_batch[:, 5], [-1, 1])}
            
            #####################
            ##### Debugging #####
            #####################
#            if np.any(np.reshape(sampled_batch[:, 4], [-1, 1])):                
#                print("Bins")
#                print(self.sess.run(self.bins))
#                print("Dones")
#                print(np.reshape(sampled_batch[:, 4], [-1, 1]))
#                print("Rewards")
#                print(np.reshape(sampled_batch[:, 2], [-1, 1]))            
#                print("Gammas")
#                print(np.reshape(sampled_batch[:, 5], [-1, 1]))
#                print("target_Z_dist")
#                print(self.sess.run(self.target_q_network, feed_dict = training_data_dict))
#                print("\n\nResulting labels")
#                print(self.sess.run(self.TEST_PROBS, feed_dict = training_data_dict))
#                raise SystemExit
            #########################
            ##### End Debugging #####
            #########################
            
            ##################################
            ##### TRAIN ACTOR AND CRITIC #####
            ##################################
            critic_loss, _ = self.sess.run([self.critic_loss, self.train_critic_one_step], feed_dict = training_data_dict)
            

            # If it's time to update the target networks
            if self.total_training_iterations % Settings.UPDATE_TARGET_NETWORKS_EVERY_NUM_ITERATIONS == 0:
                # Update the targets!
                self.sess.run(self.update_target_network_parameters)
                
            # If it's time to log the training performance to TensorBoard
            if self.total_training_iterations % Settings.LOG_TRAINING_PERFORMANCE_EVERY_NUM_ITERATIONS == 0:
                
                feed_dict = {self.iteration_loss_placeholder: critic_loss} 
                summary = self.sess.run(self.iteration_summary, feed_dict = feed_dict)
                self.writer.add_summary(summary, self.total_training_iterations)
            
            # If it's time to save a checkpoint. Be it a regular checkpoint, the final planned iteration, or the final unplanned iteration
            if (self.total_training_iterations % Settings.SAVE_CHECKPOINT_EVERY_NUM_ITERATIONS == 0) or (self.total_training_iterations == Settings.MAX_TRAINING_ITERATIONS) or stop_run_flag.is_set():

                self.saver.save(self.total_training_iterations)
                
            # If it's time to print the training performance to the screen
            if self.total_training_iterations % Settings.DISPLAY_TRAINING_PERFORMANCE_EVERY_NUM_ITERATIONS == 0:
                
                print("Trained actor and critic %i iterations in %.1f s and is now at iteration %i" % (Settings.DISPLAY_TRAINING_PERFORMANCE_EVERY_NUM_ITERATIONS, time.time() - start_time, self.total_training_iterations))
                start_time = time.time() # resetting the timer for the next PERFORMANCE_UPDATE_EVERY_NUM_ITERATIONS of iterations
            
            self.total_training_iterations += 1
        
        print("Learner finished after running " + str(self.total_training_iterations) + " training iterations!")