"""
This script contains certain functions useful to neural networks.

It includes:
    get_variables:  grabs all the variables with a given scope (name) (e.g., all the trainable parameters from 'agent_4')
    copy_variables: copies all the variables from a source to a destination (useful for updating target or agent networks)
"""

import tensorflow as tf
from settings import Settings

def get_variables(scope, trainable):
    # Gets every tensorflow variable from a defined scope    
    if trainable:
        # Grab only trainable variables
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = scope) 
    else:
        # grab non-trainable variables
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,    scope = scope) 


def copy_variables(source_variables, destination_variables, tau):
    # Copy every variable from the source to the corresponding variable in the destination.
    # tau is the update rate (1 for an exact copy, << 1 for updating targets)
    
    # Generating a list that will hold the tensorflow operations that perform
    # the appropriate update
    update_operations = []

    for source_variable, destination_variable in zip(source_variables, destination_variables):
        # Check if the source and destination variables represent the same thing (so we copy correctly)
        source_name = source_variable.name
        destination_name = destination_variable.name
        # Will throw an error if not all variable names are identical
        assert source_name[source_name.find("/"):] == destination_name[destination_name.find("/"):]
        
        ###################################################################
        #### Assigning variables from source to destination one-by-one ####
        ###################################################################
        operation = destination_variable.assign(tau * source_variable + (1 - tau) * destination_variable)
        
        # Append this newest operation
        update_operations.append(operation) 
        
    # Return the list of operations that will perform the desired copy
    return update_operations
        
            
def l2_regularization(parameters):
    # For a given set of parameters, calculate the sum of the square of all the weights (ignore the biases)
    # This is added to the critic loss function to penalize it for having large weights
    running_total = 0    
    for each_parameter in parameters:        
        if not 'bias' in each_parameter.name: # if not a bias            
            running_total += Settings.L2_REG_PARAMETER * tf.nn.l2_loss(each_parameter)  
    return running_total
    