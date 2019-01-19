#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 08:58:40 2018

@author: StephaneMagnan - stephane.magnan.11@gmail.com
"""
import numpy as np


# definition of cube class
class Cube():
    def __init__(self,dimension):
        #initialize the cube - each face has 
        state = np.empty((6,dimension,dimension))
        
        for face_ind in range(0,6):
            state[face_ind,:,:]=face_ind+1
        cube_slices = np.empty((6,dimension,dimension))      
        
        self.state = cube_slices
        self.dimension = dimension
        self.solved = True
        self.reward = 0
        return        
    
    
    def scramble(self,quarter_turns):
        #randomly applies user inputted n quarter_turns to the cube (no redundancy checks)
        moves = (self.dimension-1)*6
        for turn in range(0,quarter_turns):
            #for every turn, create a 1hot, and send for parsing (will apply rotations)
            one_hot=np.zeros((moves,1))
            
            #randomly select a move
            this_ind = np.random.randint(0,high=moves-1)
            one_hot[this_ind]=1
            
            #parse 1hot, where rotation is applied
            self.applyStep(one_hot)
            
    
    def index_cube(self):
        #overrides the state of the cube with cubie indices - solve detection will not work
        test_state = np.empty((6,self.dimension,self.dimension))
        for face_ind in range(0,6):
            for row_ind in range(0,self.dimension):
                for col_ind in range(0,self.dimension):
                    test_state[face_ind,row_ind,col_ind]=10000*(face_ind+1)+100*row_ind+col_ind
        
        self.state = test_state
        self.solved = True


    def move_U(self,clockwise,depth): 
        # rotation of the U face, depth is for high level cubes and direction is cw 1/0
        face_ind = 0
        #initialize buffer
        buffer = np.empty([self.dimension*4,depth])
        
        #populate buffer by four adjacent faces
        buffer[0:self.dimension,:]=np.rot90(self.state[1,:depth,:],k=1)
        buffer[self.dimension:2*self.dimension,:]=np.rot90(self.state[4,:depth,:],k=1)
        buffer[2*self.dimension:3*self.dimension,:]=np.rot90(self.state[3,:depth,:],k=1)
        buffer[3*self.dimension:,:]=np.rot90(self.state[2,:depth,:],k=1)
        
        #establish direction
        if clockwise == 1:
            #rotate U face clockwise
            self.state[face_ind,:,:]=np.rot90(self.state[face_ind,:,:], k=-1, axes=(0,1))
            
            #roll buffer forwards
            buffer = np.concatenate((buffer[-self.dimension:,:],buffer[:-self.dimension,:]),axis=0)
        else:
            #rotate U face counterclockwise
            self.state[face_ind,:,:]=np.rot90(self.state[face_ind,:,:], k=1, axes=(0,1))
            
            #roll buffer backwards
            buffer=np.concatenate((buffer[self.dimension:,:],buffer[:self.dimension,:]),axis=0)
        
        #repopulate state from rolled buffer
        self.state[1,:depth,:]=np.rot90(buffer[0:self.dimension,:],k=-1)
        self.state[4,:depth,:]=np.rot90(buffer[self.dimension:2*self.dimension,:],k=-1)
        self.state[3,:depth,:]=np.rot90(buffer[2*self.dimension:3*self.dimension,:],k=-1)
        self.state[2,:depth,:]=np.rot90(buffer[3*self.dimension:,:],k=-1)


    def move_F(self,clockwise,depth):    
        # rotation of the F face, depth is for high level cubes and direction is cw 1/0
        face_ind = 1
        #initialize buffer
        buffer = np.empty([self.dimension*4,depth])
        
        #populate buffer by four adjacent faces
        buffer[0:self.dimension,:]=np.rot90(self.state[0,-depth:,:],k=-1)
        buffer[self.dimension:2*self.dimension,:]=np.rot90(self.state[2,:,:depth],k=0)
        buffer[2*self.dimension:3*self.dimension,:]=np.rot90(self.state[5,:depth,:],k=1)
        buffer[3*self.dimension:,:]=np.rot90(self.state[4,:,-depth:],k=2)
        
        #establish direction
        if clockwise == 1:
            #rotate U face clockwise
            self.state[face_ind,:,:]=np.rot90(self.state[face_ind,:,:], k=-1, axes=(0,1))
            
            #roll buffer forwards
            buffer = np.concatenate((buffer[-self.dimension:,:],buffer[:-self.dimension,:]),axis=0)
        else:
            #rotate U face counterclockwise
            self.state[face_ind,:,:]=np.rot90(self.state[face_ind,:,:], k=1, axes=(0,1))
            
            #roll buffer backwards
            buffer=np.concatenate((buffer[self.dimension:,:],buffer[:self.dimension,:]),axis=0)
        
        #repopulate state from rolled buffer
        self.state[0,-depth:,:]=np.rot90(buffer[0:self.dimension,:],k=1)
        self.state[2,:,:depth]=np.rot90(buffer[self.dimension:2*self.dimension,:],k=0)
        self.state[5,:depth,:]=np.rot90(buffer[2*self.dimension:3*self.dimension,:],k=-1)
        self.state[4,:,-depth:]=np.rot90(buffer[3*self.dimension:,:],k=2) 
        
        
    def move_R(self,clockwise,depth):    
        # rotation of the R face, depth is for high level cubes and direction is cw 1/0
        face_ind = 2
        #initialize buffer
        buffer = np.empty([self.dimension*4,depth])
        
        #populate buffer by four adjacent faces
        buffer[0:self.dimension,:]=np.rot90(self.state[0,:,-depth:],k=2)
        buffer[self.dimension:2*self.dimension,:]=np.rot90(self.state[3,:,:depth],k=0)
        buffer[2*self.dimension:3*self.dimension,:]=np.rot90(self.state[5,:,-depth:],k=2)
        buffer[3*self.dimension:,:]=np.rot90(self.state[1,:,-depth:],k=2)
        
        #establish direction
        if clockwise == 1:
            #rotate U face clockwise
            self.state[face_ind,:,:]=np.rot90(self.state[face_ind,:,:], k=-1, axes=(0,1))
            
            #roll buffer forwards
            buffer = np.concatenate((buffer[-self.dimension:,:],buffer[:-self.dimension,:]),axis=0)
        else:
            #rotate U face counterclockwise
            self.state[face_ind,:,:]=np.rot90(self.state[face_ind,:,:], k=1, axes=(0,1))
            
            #roll buffer backwards
            buffer=np.concatenate((buffer[self.dimension:,:],buffer[:self.dimension,:]),axis=0)
        
        #repopulate state from rolled buffer
        self.state[0,:,-depth:]=np.rot90(buffer[0:self.dimension,:],k=2)
        self.state[3,:,:depth]=np.rot90(buffer[self.dimension:2*self.dimension,:],k=0)
        self.state[5,:,-depth:]=np.rot90(buffer[2*self.dimension:3*self.dimension,:],k=2)
        self.state[1,:,-depth:]=np.rot90(buffer[3*self.dimension:,:],k=2)  
        
        
    def move_B(self,clockwise,depth):    
        # rotation of the B face, depth is for high level cubes and direction is cw 1/0
        face_ind = 3
        #initialize buffer
        buffer = np.empty([self.dimension*4,depth])
        
        #populate buffer by four adjacent faces
        buffer[0:self.dimension,:]=np.rot90(self.state[0,:depth,:],k=1)
        buffer[self.dimension:2*self.dimension,:]=np.rot90(self.state[4,:,:depth],k=0)
        buffer[2*self.dimension:3*self.dimension,:]=np.rot90(self.state[5,-depth:,:],k=-1)
        buffer[3*self.dimension:,:]=np.rot90(self.state[2,:,-depth:],k=2)
        
        #establish direction
        if clockwise == 1:
            #rotate U face clockwise
            self.state[face_ind,:,:]=np.rot90(self.state[face_ind,:,:], k=-1, axes=(0,1))
            
            #roll buffer forwards
            buffer = np.concatenate((buffer[-self.dimension:,:],buffer[:-self.dimension,:]),axis=0)
        else:
            #rotate U face counterclockwise
            self.state[face_ind,:,:]=np.rot90(self.state[face_ind,:,:], k=1, axes=(0,1))
            
            #roll buffer backwards
            buffer=np.concatenate((buffer[self.dimension:,:],buffer[:self.dimension,:]),axis=0)
        
        #repopulate state from rolled buffer
        self.state[0,:depth,:]=np.rot90(buffer[0:self.dimension,:],k=-1)
        self.state[4,:,:depth]=np.rot90(buffer[self.dimension:2*self.dimension,:],k=0)
        self.state[5,-depth:,:]=np.rot90(buffer[2*self.dimension:3*self.dimension,:],k=1)
        self.state[2,:,-depth:]=np.rot90(buffer[3*self.dimension:,:],k=2) 
        
        
    def move_L(self,clockwise,depth):    
       # rotation of the R face, depth is for high level cubes and direction is cw 1/0
        face_ind = 4
        #initialize buffer
        buffer = np.empty([self.dimension*4,depth])
        
        #populate buffer by four adjacent faces
        buffer[0:self.dimension,:]=np.rot90(self.state[0,:,:depth],k=0)
        buffer[self.dimension:2*self.dimension,:]=np.rot90(self.state[1,:,:depth],k=0)
        buffer[2*self.dimension:3*self.dimension,:]=np.rot90(self.state[5,:,:depth],k=0)
        buffer[3*self.dimension:,:]=np.rot90(self.state[3,:,-depth:],k=2)
        
        #establish direction
        if clockwise == 1:
            #rotate U face clockwise
            self.state[face_ind,:,:]=np.rot90(self.state[face_ind,:,:], k=-1, axes=(0,1))
            
            #roll buffer forwards
            buffer = np.concatenate((buffer[-self.dimension:,:],buffer[:-self.dimension,:]),axis=0)
        else:
            #rotate U face counterclockwise
            self.state[face_ind,:,:]=np.rot90(self.state[face_ind,:,:], k=1, axes=(0,1))
            
            #roll buffer backwards
            buffer=np.concatenate((buffer[self.dimension:,:],buffer[:self.dimension,:]),axis=0)
        
        #repopulate state from rolled buffer
        self.state[0,:,:depth]=np.rot90(buffer[0:self.dimension,:],k=0)
        self.state[1,:,:depth]=np.rot90(buffer[self.dimension:2*self.dimension,:],k=0)
        self.state[5,:,:depth]=np.rot90(buffer[2*self.dimension:3*self.dimension,:],k=0)
        self.state[3,:,-depth:]=np.rot90(buffer[3*self.dimension:,:],k=2)  
        
        
    def move_D(self,clockwise,depth):    
        # rotation of the D face, depth is for high level cubes and direction is cw 1/0
        face_ind = 5
        #initialize buffer
        buffer = np.empty([self.dimension*4,depth])
        
        #populate buffer by four adjacent faces
        buffer[0:self.dimension,:]=np.rot90(self.state[1,-depth:,:],k=-1)
        buffer[self.dimension:2*self.dimension,:]=np.rot90(self.state[2,-depth:,:],k=-1)
        buffer[2*self.dimension:3*self.dimension,:]=np.rot90(self.state[3,-depth:,:],k=-1)
        buffer[3*self.dimension:,:]=np.rot90(self.state[4,-depth:,:],k=-1)
        
        #establish direction
        if clockwise == 1:
            #rotate U face clockwise
            self.state[face_ind,:,:]=np.rot90(self.state[face_ind,:,:], k=-1, axes=(0,1))
            
            #roll buffer forwards
            buffer = np.concatenate((buffer[-self.dimension:,:],buffer[:-self.dimension,:]),axis=0)
        else:
            #rotate U face counterclockwise
            self.state[face_ind,:,:]=np.rot90(self.state[face_ind,:,:], k=1, axes=(0,1))
            
            #roll buffer backwards
            buffer=np.concatenate((buffer[self.dimension:,:],buffer[:self.dimension,:]),axis=0)
        
        #repopulate state from rolled buffer
        self.state[1,-depth:,:]=np.rot90(buffer[0:self.dimension,:],k=1)
        self.state[2,-depth:,:]=np.rot90(buffer[self.dimension:2*self.dimension,:],k=1)
        self.state[3,-depth:,:]=np.rot90(buffer[2*self.dimension:3*self.dimension,:],k=1)
        self.state[4,-depth:,:]=np.rot90(buffer[3*self.dimension:,:],k=1) 

   
    def parse1Hot(self,one_hot):     
        #'convert 1XN vector (1-hot) output from network to a move
        #'call the move_ method indicated by vector
        ind = np.where(one_hot)
        
        depth = ind[0][0]//12+1
        
        clockwise = np.mod(ind[0][0],2)
        
        face_ind = np.mod(np.ceil((ind[0][0]+1)/2),6)
        #
        print('Face: ',face_ind,' Clockwise: ',clockwise,' Depth: ',depth)
        if face_ind == 0:
            self.move_U(clockwise,depth)
        elif face_ind == 1:
             self.move_F(clockwise,depth)
        elif face_ind == 2:
            self.move_R(clockwise,depth)
        elif face_ind == 3:
            self.move_B(clockwise,depth)
        elif face_ind == 4:
            self.move_L(clockwise,depth)
        elif face_ind == 5:
            self.move_D(clockwise,depth)
        else:    
            #should not occur
            pass


    def applyStep(self,one_hot):
        #apply move with 1hot input
        
        self.parse1Hot(one_hot)
        self.is_complete()

        #must return state, reward (this step only), done (T/F)
        return self.state, self.solved, self.reward

    
    def is_complete(self):
        self.solved = True
        self.reward = 100
        for face_ind in range(0,6):
            if np.max(self.state[face_ind,:,:]) != np.min(self.state[face_ind,:,:]):
                self.solved = False
                self.reward = -1
                break

# misc. debugging code
dim = 3
bob = Cube(dim)
#bob.scramble(2)
#print(bob.solved)
#bob.index_cube()
#blah = bob.test_state
depth = 2
if False:
    print('U')
    bob.move_U(1,depth)
    print(bob.state)
    print('UP')
    bob.move_U(0,depth)
    print(bob.state)
    print('D')
    bob.move_D(1,depth)
    print(bob.state)
    print('DP')
    bob.move_D(0,depth)
    print(bob.state)
    print('F')
    bob.move_F(1,depth)
    print(bob.state)
    print('FP')
    bob.move_F(0,depth)
    print(bob.state)
    print('B')
    bob.move_B(1,depth)
    print(bob.state)
    print('BP')
    bob.move_B(0,depth)
    print(bob.state)
    print('R')
    bob.move_R(1,depth)
    print(bob.state)
    print('RP')
    bob.move_R(0,depth)
    print(bob.state)
    print('L')
    bob.move_L(1,depth)
    print(bob.state)
    print('LP')
    bob.move_L(0,depth)
    print(bob.state)     