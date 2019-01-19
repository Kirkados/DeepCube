"""
Generates and manages the large experience replay buffer.

The experience replay buffer holds all the data that is dumped into it from the 
many agents who are running episodes of their own. The learner then trains off 
this heap of data continually and in its own thread.

Future work will incorporate priority into the sampling (http://arxiv.org/abs/1511.05952), if requested.
(though I'm not sure if I'll ever do this as the benefits don't seem all that large)
"""

import random
import numpy as np

from collections import deque

from settings import Settings


def ReplayBuffer(prioritized):
    if prioritized:
        return PrioritizedReplayBuffer()
    else:
        return RegularReplayBuffer()
    

class RegularReplayBuffer:
    # Generates and manages a non-prioritized replay buffer
    
    def __init__(self):
        # Generate the buffer
        self.buffer = deque(maxlen = Settings.REPLAY_BUFFER_SIZE)
        random.seed(Settings.RANDOM_SEED)

    # Query how many entries are in the buffer
    def how_filled(self):
        return len(self.buffer)
    
    # Add new experience to the buffer
    def add(self, experience):
        self.buffer.append(experience)
        
    # Randomly sample data from the buffer
    def sample(self):
        # Decide how much data to sample
        # (maybe the buffer doesn't contain enough samples yet to fill a MINI_BATCH)
        batch_size = min(Settings.MINI_BATCH_SIZE, len(self.buffer)) 
        return random.sample(self.buffer, batch_size)
        
    
class PrioritizedReplayBuffer():
    
    def __init__(self):
        self.buffer = SumTree(capacity = Settings.REPLAY_BUFFER_SIZE) # initialize replay buffer using the SumTree class
        #self.buffer = deque(maxlen = Settings.REPLAY_BUFFER_SIZE) # Initialize the buffer
        random.seed(Settings.RANDOM_SEED)
        
    # Add experience to the replay buffer
    def add(self, experience):
        self.buffer.add(self.buffer.max(), experience)
        
    # Sample data from replay buffer in a prioritized fashion
    def sample(self, beta):
        data, index, priorities = self.buffer.sample(Settings.MINI_BATCH_SIZE)
        probabilities = priorities / self.buffer.total()
        weights = (self.buffer.n_entries * probabilities) ** -beta
        weights /= np.max(weights)
        
        return data, index, weights
    
    # Update priorities??!?!
    def update(self, index, errors):
        priorities = (np.abs(errors) + 1e-6) ** Settings.ALPHA
        
        for i in range(len(index)):
            self.buffer.update(index[i], priorities[i])