"""
Builds the tensorflow graph neural networks for the actor and critic
"""

import tensorflow as tf

from settings import Settings

def build_actor_network(state, trainable, scope):
    """ 
    Build the actor network that receives the state and outputs the action.
    
    Inputs:
        state:      A placeholder where the input will be fed when used.
        trainable:  Whether the network can be trained (learner) or if its weights are frozen (actors)
        scope:      Name of the tensorflow scope
    """   
    
    # Making sure all variables generated here are under the name "scope"
    with tf.variable_scope(scope):
        
        # The first layer is the state (input)
        layer = state
        
        # If learning from pixels include convolutional layers
        if Settings.LEARN_FROM_PIXELS:
            
            # Build convolutional layers
            for i, conv_layer_settings in enumerate(Settings.CONVOLUTIONAL_LAYERS):
                layer = tf.layers.conv2d(inputs = layer,
                                         activation = tf.nn.relu,
                                         trainable = trainable,
                                         name = 'conv_layer' + str(i),
                                         **conv_layer_settings) 
                # ** means that named arguments are being passed to the function.
                # The conv2d function is able to accept the keywords.
            
            # Flattening image into a column for subsequent fully-connected layers
            layer = tf.layers.flatten(layer) 
        
        
        # Building fully-connected layers
        for i, number_of_neurons in enumerate(Settings.ACTOR_HIDDEN_LAYERS):
            layer = tf.layers.dense(inputs = layer,
                                    units = number_of_neurons,
                                    trainable = trainable,
                                    activation = tf.nn.relu,
                                    name = 'fully_connected_layer_' + str(i))
        
        # Convolutional layers (optional) have been applied, followed by fully-connected hidden layers
        # The final layer goes from the output of the last fully-connected layer
        # to the action size. It is squished with a signmoid and then scaled to the action range.
        # Sigmoid forces output between 0 and 1, which I need to scale to the action range
        actions_out_unscaled = tf.layers.dense(inputs = layer,
                                               units = Settings.ACTION_SIZE,
                                               trainable = trainable,
                                               activation = tf.nn.tanh,
                                               name = 'output_layer') 
        
        # Scaling actions to the correct range
        #action_scaled = Settings.LOWER_ACTION_BOUND + tf.multiply(actions_out_unscaled, Settings.ACTION_RANGE) # for sigmoid
        action_scaled = tf.multiply(0.5, tf.multiply(actions_out_unscaled, Settings.ACTION_RANGE) + Settings.LOWER_ACTION_BOUND + Settings.UPPER_ACTION_BOUND) # for tanh
        
        # Return the chosen action
        return action_scaled
        
        
def build_Q_network(state, action, trainable, reuse, scope):
    """
    Defines a critic network that predicts the Q-value (expected future return)
    from a given state and action. 
    
    The network archetectire is as given in the D4PG paper. The action goes through
    two layers on its own before being added to the state who has went through
    one layer. Then, the sum of the two goes through the final layer. This is different
    than my original implementation where the [state, action] were concatenated
    and ran through the layers together. I changed to the updated method Jan 7, 2019.
    """
    with tf.variable_scope(scope):
        # Two sides flow through the network independently.
        state_side  = state
        action_side = action
        
        ######################
        ##### State Side #####
        ######################
        # If learning from pixels (a state-only feature), use convolutional layers
        if Settings.LEARN_FROM_PIXELS:            
            # Build convolutional layers
            for i, conv_layer_settings in enumerate(Settings.CONVOLUTIONAL_LAYERS):
                state_side = tf.layers.conv2d(inputs = state_side,
                                              activation = tf.nn.relu,
                                              trainable = trainable,
                                              reuse = reuse,
                                              name = 'state_conv_layer' + str(i),
                                              **conv_layer_settings) # the "**" allows the passing of keyworded arguments
            
            # Flattening image into a column for subsequent layers 
            state_side = tf.layers.flatten(state_side) 
                
        # Fully connected layers on state side from the second layer onwards Settings.CRITIC_HIDDEN_LAYERS[1:]
        for i, number_of_neurons in enumerate(Settings.CRITIC_HIDDEN_LAYERS[1:]):
            state_side = tf.layers.dense(inputs = state_side,
                                         units = number_of_neurons,
                                         trainable = trainable,
                                         reuse = reuse,
                                         activation = tf.nn.relu,
                                         name = 'state_fully_connected_layer_' + str(i))
        #######################
        ##### Action Side #####
        #######################
        # Fully connected layers on action side
        for i, number_of_neurons in enumerate(Settings.CRITIC_HIDDEN_LAYERS):
            action_side = tf.layers.dense(inputs = action_side,
                                          units = number_of_neurons,
                                          trainable = trainable,
                                          reuse = reuse,
                                          activation = tf.nn.relu,
                                          name = 'action_fully_connected_layer_' + str(i))
        
        ################################################
        ##### Combining State Side and Action Side #####
        ################################################
        
        layer = tf.add(state_side, action_side)
        
        #################################################
        ##### Final Layer to get Value Distribution #####
        #################################################
        """
        On the state side, convolutional layers were (optionally) applied, followed
        by M-1 fully connected layers. The action side passes through M fully
        connected layers. The two sides are summed together, and then passed 
        through the final layer to yield the distributional output of correct 
        size.        
        """

        # Calculating the final layer logits as an intermediate step,
        # since the cross_entropy loss function needs logits.
        q_value_logits = tf.layers.dense(inputs = layer,
                                         units = Settings.NUMBER_OF_BINS,
                                         trainable = trainable,
                                         reuse = reuse,
                                         activation = None,
                                         name = 'output_layer')
        
        # Calculating the softmax of the last layer to convert logits to probabilities
        q_value = tf.nn.softmax(q_value_logits, name = 'output_probabilities') # softmax ensures that all outputs add up to 1, relative to their weights
 
        return q_value, q_value_logits