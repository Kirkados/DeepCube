"""
This script builds the QNetwork (critic in the case of D4PG) and generates
all the functions needed to train it. It also creates the agent who is 
continually trained. Target networks are used as well.

Inspired by flyyufelix, msinto93, and SuRELI's github code

Training:
    The critic is trained using supervised learning to minimize the 
    cross-entropy loss between the Q value and the target value 
    y = r_t + gamma * Q_target(next_state, Action(next_state))
        
    To train the actor, we apply the policy gradient
    Grad = grad(Q(s,a), A)) * grad(A, params)  
"""

import tensorflow as tf
import numpy as np
import time

from build_neural_networks import build_actor_network, build_Q_network
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
            self.action_placeholder          = tf.placeholder(dtype = tf.float32, shape = [None, Settings.ACTION_SIZE], name = 'action_placeholder')
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
            
            # Setting up distributional critic items            
            self.bin_width = (Settings.MAX_Q - Settings.MIN_Q) / (Settings.NUMBER_OF_BINS - 1)
            self.bins      = tf.range(Settings.MIN_Q, Settings.MAX_Q + self.bin_width, self.bin_width)
                        
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
        # Build the critic and the trained actor
        
        ###############################
        ### Build the learned actor ###
        ###############################
        self.actor = build_actor_network(state = self.state_placeholder, 
                                         trainable = True, 
                                         scope = 'learned_actor')
        
        ################################
        ### Build the learned critic ###
        ################################
        # The '_' is the q_network softmax outputs, which isn't needed.
        _, self.q_network_logits = build_Q_network(state = self.state_placeholder, 
                                                   action = self.action_placeholder, 
                                                   trainable = True, 
                                                   reuse = False, 
                                                   scope = 'learned_critic')
        
        #####################################################################################
        ### Build another form of the critic, where the actions are provided by the actor ###
        #####################################################################################
        # Note: This is called "learned_critic_1" in tensorboard even though it has the same scope
        self.q_network_with_actor, _ = build_Q_network(state = self.state_placeholder,
                                                       action = self.actor,
                                                       trainable = False,
                                                       reuse = True, # this means this network takes parameters from the main self.q_network
                                                       scope = 'learned_critic')
        
        # Turn the Q-Network distribution into a Q-value. 
        with tf.variable_scope("Q-distribution_to_Q-value"):
            self.q_network_value = tf.reduce_sum(self.bins * self.q_network_with_actor, axis = 1)

        
    def build_targets(self):
        # Build the target networks
        
        # Target Actor -> Returns the next action because it receives the next state
        self.target_actor = build_actor_network(state = self.next_state_placeholder,
                                                trainable = False,
                                                scope = 'target_actor')
        
        # Target Q Network -> Returns the next probabilities because it uses the next state and next action.
        self.target_q_network, _ = build_Q_network(state = self.next_state_placeholder,
                                                   action = self.target_actor,
                                                   trainable = False,
                                                   reuse = False,
                                                   scope = 'target_critic')
        
        
    def build_target_parameter_update_operations(self):
        # Update the slowly-changing target networks according to tau
        
        with tf.variable_scope("Update_Target_Networks"):
                    
            # Grab parameters from the main networks
            self.actor_parameters  = get_variables(scope = 'learned_actor',  trainable = True)
            self.critic_parameters = get_variables(scope = 'learned_critic', trainable = True)
            
            self.parameters = self.actor_parameters + self.critic_parameters
            
            # Grab parameters from target networks
            self.target_actor_parameters  = get_variables(scope = 'target_actor',  trainable = False)
            self.target_critic_parameters = get_variables(scope = 'target_critic', trainable = False)
            
            self.target_parameters = self.target_actor_parameters + self.target_critic_parameters
            
            # Operation to initialize the target networks to be identical to the main networks
            self.initialize_target_network_parameters = copy_variables(source_variables = self.parameters, 
                                                                       destination_variables = self.target_parameters, 
                                                                       tau = 1)
            
            # Updating target networks at rate Settings.TARGET_NETWORK_TAU 
            self.update_target_network_parameters     = copy_variables(source_variables = self.parameters, 
                                                                       destination_variables = self.target_parameters, 
                                                                       tau = Settings.TARGET_NETWORK_TAU)
            
    def build_network_training_operations(self):
        # Builds the operations that are used to train the real actor and critic
        
        with tf.variable_scope("Train_Critic"): # grouping tensorboard graph
            
            ##################################################
            ###### Generating Updated Bin Probabilities ######
            ##################################################
            
            # Initializing the matrix that will hold the new bin probabilities as they get generated
            new_bin_probabilities = tf.zeros([Settings.MINI_BATCH_SIZE, Settings.NUMBER_OF_BINS])
            
            # For each bin, project where that bin's probability should end up after receiving the reward
            # by calculating the new expected reward. Then, find out what bin the projection lies in. 
            # Then, distribute the probability into that bin. Then, build a loss function to minimize
            # the difference between the current distribution and the calculated distribution.
            for this_bin in range(Settings.NUMBER_OF_BINS): # for each bin
                
                # Bellman projection. reward + gamma^N*not_done*bin -> The new                 
                # expected reward, according to the recently-received reward. 
                # If the new reward is outside of the current bin, then we will
                # adjust the probability that is assigned to the bin.
                # If the episode terminates here, the new expected reward from 
                # this state-action pair is just the reward.
                projection = self.reward_placeholder + (self.discount_factor_placeholder)*(1.0 - self.done_placeholder)*self.bins[this_bin] 
                
                # Clipping projected reward to its bounds.
                projection = tf.squeeze(tf.clip_by_value(projection, Settings.MIN_Q, Settings.MAX_Q)) # Squeezing -> shape [batch_size]
                
                # Which bin number the projected value ends up in (so we know which bin to increase the probability of)
                new_bin = (projection - Settings.MIN_Q)/self.bin_width # shape [batch_size]
                
                # However, it is unlikely the new bin number will lie directly 
                # on an existing bin number. Therefore, determine the nearby
                # bins so we know where we should distribute the probability into.         
                adjacent_bin_upper = tf.ceil(new_bin) # shape [batch_size]
                adjacent_bin_lower = tf.floor(new_bin) # shape [batch_size]
                
                # Checking if the upper and lower bins are the same bin (!!!).
                # This occurs when the projection lies directly on a bin.
                # Common causes are: 1) The reward is large and pushes the projection  
                # to one of the bounds (where it is clipped). 2) There is a 
                # reward of 0 for bin[i] = 0.
                are_bins_identical = tf.equal(adjacent_bin_upper, adjacent_bin_lower) # shape [batch_size]
                are_bins_different = tf.logical_not(are_bins_identical)               # shape [batch_size]
                
                # Generating two one-hot matrices that will be used to place the 
                # projected next-state probabilities from the target critic 
                # network into the appropriate bin. The appropriate bin is the 
                # one who we would like to increase their probability.
                # Only one element in each row is a 1, all others are 0.
                one_hot_upper = tf.one_hot(tf.to_int32(adjacent_bin_upper), depth = Settings.NUMBER_OF_BINS) # shape [batch_size, #atoms]
                one_hot_lower = tf.one_hot(tf.to_int32(adjacent_bin_lower), depth = Settings.NUMBER_OF_BINS) # shape [batch_size, #atoms]               
                
                # Disributing the next-state bin probabilities (from the target 
                # q_network) into both bins dictated by the projection.
                # Accumulating the new bin probabilities as we loop through all bins.
                # Note: the "upper" part gets multiplied by the one_hot_lower because
                #       the (upper - new_bin) is essentially "how far" the new bin is from the
                #       upper bin. Therefore, the larger that number, the more we
                #       should put in the lower bin.
                # This accumulation only applies to samples in the batch that 
                # have been assigned different bins (by multiplying by are_bins_different)
                new_bin_probabilities += tf.reshape(self.target_q_network[:, this_bin] * (adjacent_bin_upper - new_bin) * tf.to_float(are_bins_different), [-1, 1]) * one_hot_lower # [batch_size, 1] * [batch_size, #atoms] = [batch_size, #atoms]
                new_bin_probabilities += tf.reshape(self.target_q_network[:, this_bin] * (new_bin - adjacent_bin_lower) * tf.to_float(are_bins_different), [-1, 1]) * one_hot_upper # [batch_size, 1] * [batch_size, #atoms] = [batch_size, #atoms]
                
                # If, by chance, the new_bin lies directly on a bin, then the 
                # adjacent_bin_upper and adjacent_bin_lower will be identical.
                # In that case, the full next-state probability is added to that 
                # bin.
                new_bin_probabilities += tf.reshape(self.target_q_network[:, this_bin] * tf.to_float(are_bins_identical), [-1, 1]) * one_hot_upper # [batch_size, 1] * [batch_size, #atoms] = [batch_size, #atoms]

                
            ###########################################
            ##### Generating Critic Loss Function #####
            ###########################################
            
            # DEBUGGING
            #self.TEST_PROBS = new_bin_probabilities
            # END DEBUGGING
            
            # We've now got the new distribution (bin probabilities),
            # now we must generate a loss function for the critic!
            self.critic_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.q_network_logits, 
                                                                            labels = tf.stop_gradient(new_bin_probabilities)) # not sure if tf.stop_gradients is needed, but it certainly doesn't hurt
            
            # Taking the mean loss across the batch
            self.critic_loss = tf.reduce_mean(self.critic_losses)
            
            # Optional L2 Regularization
            if Settings.L2_REGULARIZATION:
                # Penalize the critic for having large weights -> L2 Regularization
                self.critic_loss += l2_regularization(self.critic_parameters) 

            
            ##############################################################################
            ##### Develop the Operation that Trains the Critic with Gradient Descent #####
            ##############################################################################
            critic_trainer             = tf.train.AdamOptimizer(Settings.CRITIC_LEARNING_RATE)
            self.train_critic_one_step = critic_trainer.minimize(self.critic_loss) # RUN THIS TO TRAIN THE CRITIC ONE STEP
            
        
        ##############################################################################
        ##### Develop the Operation that Trains the Actor with Gradient Descent ###### Note: dQ/dActor_parameter = (dQ/dAction)*(dAction/dActor_parameter)
        ##############################################################################
        with tf.variable_scope("Train_Actor"):
            self.dQ_dAction             = tf.gradients(self.q_network_value, self.actor)[0] # also called 'action_gradients'
            self.actor_gradients        = tf.gradients(self.actor, self.actor_parameters, -self.dQ_dAction) # pushing the gradients through to the actor parameters
            self.scaled_actor_gradeints = [tf.div(x, Settings.MINI_BATCH_SIZE) for x in self.actor_gradients] # tf.gradients sums over the batch but we want the mean, so we divide by batch size
            actor_trainer               = tf.train.AdamOptimizer(Settings.ACTOR_LEARNING_RATE)                    # establishing the training method
            self.train_actor_one_step   = actor_trainer.apply_gradients(zip(self.scaled_actor_gradeints, self.actor_parameters))    # RUN THIS TO TRAIN THE ACTOR
            
            
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
            critic_loss, _, _ = self.sess.run([self.critic_loss, self.train_critic_one_step, self.train_actor_one_step], feed_dict = training_data_dict)
            

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