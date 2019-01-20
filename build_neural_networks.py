"""
Builds the tensorflow graph neural networks for the actor and critic
"""

import tensorflow as tf

from settings import Settings

def build_Q_network(state, trainable, reuse, scope):
    """
    Defines a Q network that predicts the Q-value (expected future return)
    from taking a certain action from a given state, and then following the policy
    thereafter.
    """
    with tf.variable_scope(scope):
        # The state is effectively the "0th" layer
        layer = state
        
        ##################################
        ##### Fully connected layers #####
        ##################################
        for i, number_of_neurons in enumerate(Settings.Q_NETWORK_HIDDEN_LAYERS):
            layer = tf.layers.dense(inputs = layer,
                                         units = number_of_neurons,
                                         trainable = trainable,
                                         reuse = reuse,
                                         activation = tf.nn.relu,
                                         name = 'fully_connected_layer_' + str(i))
        
        
        ##############################################
        ##### Final Layer to get Action Q-values #####
        ##############################################
        """
        The final layer does not go through an activation function, because we 
        don't need the output Q-value to be within a certain range
        """
        
        q_values = tf.layers.dense(inputs = layer,
                                   units = Settings.ACTION_SIZE,
                                   trainable = trainable,
                                   reuse = reuse,
                                   activation = None,
                                   name = 'output_layer')
        
        return q_values