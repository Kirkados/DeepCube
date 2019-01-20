""" 
    All settings relating to D4PG are contained in this file.
    This file is copied on each run and placed in the Tensorboard directory
    so that all settings are preserved for future reference. 
"""

class Settings:
    
    #%% 
    ########################
    ##### Run Settings #####
    ########################
    
    RUN_NAME               = 'debugging_cartpole' # use just the name. If trying to restore from file, use name along with timestamp
    USE_GYM                = 1 # 0 = use (your own) dynamics; 1 = use openAI's gym (for testing)
    ENVIRONMENT            = 'CartPole-v0'
    RECORD_VIDEO           = True
    VIDEO_RECORD_FREQUENCY = 1000 # Multiples of "CHECK_GREEDY_PERFORMANCE_EVERY_NUM_EPISODES"
    RESUME_TRAINING        = False # If True, be sure to set "RUN_NAME" to the previous run's filename
    USE_GPU_WHEN_AVAILABLE = False # As of Nov 19, 2018, it appears better to use CPU. Re-evaluate again later
    RANDOM_SEED            = 41
    
    #%% 
    #############################
    ##### Training Settings #####
    #############################
    
    # Hyperparameters
    NUMBER_OF_ACTORS        = 4
    NUMBER_OF_EPISODES      = 1e5 # that each agent will perform
    MAX_TRAINING_ITERATIONS = 5e5
    MAX_NUMBER_OF_TIMESTEPS = 1000 # per episode
    CRITIC_LEARNING_RATE    = 0.0001
    TARGET_NETWORK_TAU      = 0.001
    DISCOUNT_FACTOR         = 0.99
    N_STEP_RETURN           = 5
    L2_REGULARIZATION       = False # optional for training the critic
    L2_REG_PARAMETER        = 1e-6
    
    # Periodic events
    UPDATE_TARGET_NETWORKS_EVERY_NUM_ITERATIONS       = 1 
    UPDATE_ACTORS_EVERY_NUM_EPISODES                  = 1
    CHECK_GREEDY_PERFORMANCE_EVERY_NUM_EPISODES       = 5    
    LOG_TRAINING_PERFORMANCE_EVERY_NUM_ITERATIONS     = 100
    DISPLAY_TRAINING_PERFORMANCE_EVERY_NUM_ITERATIONS = 50000
    DISPLAY_ACTOR_PERFORMANCE_EVERY_NUM_EPISODES      = 1000
    
    # Buffer settings
    PRIORITIZED_REPLAY_BUFFER             = False
    REPLAY_BUFFER_SIZE                    = 1000000
    REPLAY_BUFFER_START_TRAINING_FULLNESS = 0 # how full the buffer should be before training begins
    MINI_BATCH_SIZE                       = 256
        
    # Exploration noise
    NOISE_SCALE           = 1 # 1 is best for uniform -> noise scaled to the action range
    NOISE_SCALE_DECAY     = 0.9999 # 1 means the noise does not decay during training
    
#%%
    ####################################
    ##### Model Structure Settings #####
    ####################################
    Q_NETWORK_HIDDEN_LAYERS = [400, 300] # number of hidden neurons in each layer

    #%%     
    ##############################
    #### Environment Settings ####
    ##############################
    
    # Get state & action shapes from environment & action bounds
    if USE_GYM:
        import gym
        test_env_to_get_settings = gym.make(ENVIRONMENT)
        STATE_SIZE               = list(test_env_to_get_settings.observation_space.shape) # dimension of the observation/state space            
        ACTION_SIZE              = test_env_to_get_settings.action_space.n # number of available actions
        #del test_env_to_get_settings # getting rid of this test environment
        print(test_env_to_get_settings.reset())
        print(test_env_to_get_settings.step(1))

    #########################
    #### TO BE COMPLETED ####
    #########################        
    else: # use your own dynamics
        from Dynamics import Dynamics
        STATE_SIZE  = 0 # INCOMPLETE
        ACTION_SIZE = 0 # INCOMPLETE
        
    #%% 
    #########################
    ##### Save Settings #####
    #########################
    
    MODEL_SAVE_DIRECTORY                 = 'Tensorboard/' # where to save all data
    TENSORBOARD_FILE_EXTENSION           = '.tensorboard' # file extension for tensorboard file
    SAVE_CHECKPOINT_EVERY_NUM_ITERATIONS = 50000 # how often to save the neural network parameters
