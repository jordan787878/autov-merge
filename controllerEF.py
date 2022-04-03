# Ego Follower Controller

import numpy as np
from itertools import product
import controllerHL
from car import Car

import copy
from trajectory import *
from visualize import *


dt = 0.1

# For Merging Behavior Reward
GOAL_X = 200
MERGE_START_X = 50
MERGE_END_X = 175
# For Collision Detect
MERGE_SAFE_DIST = 12

### Helper ###
IsVis = True 

class ControllerEF:
    def __init__(self,name,car_my,car_op):
        print("init EF controller")
        self.name = name

        # Link Interacting Cars
        self.car_my = car_my
        self.car_op = car_op

        # Obtain horizon
        self.horizon = self.car_my.horizon

        # Init action
        self.action = None

        # Get action set
        self.act_set_my = self.car_my.get_all_actions()
        self.act_set_op = self.car_op.get_all_actions()

        # Prepare num actions
        self.num_act_my = self.act_set_my.shape[0]
        self.num_act_op = self.act_set_op.shape[0]

        # Discount Vector
        self.discount = np.zeros(self.horizon)
        for i in range(self.horizon):
            self.discount[i] = pow(0.9,i)

        
    def select_action(self):
        acts_opt, traj_opt = self.select_opt_actions()

#        print("actual ego follower opt. actions\t", acts_opt)
        
        self.action = acts_opt[0]

        # Continue Mergine Step
        if self.car_my.IsMerging and self.car_my.MergeCount < self.car_my.merge_steps:
            print("controllerMerge overwrite to continue merging")
            self.action = 3
        
        return self.action

    def select_opt_actions(self,call_from_hum=False):
#        if call_from_hum == True:
#            print("ego select opt actions call from human")
#            print(self.num_actions)
#            print(self.num_opp_actions)
        
        # Init Q Matrix
        Qfi = np.zeros(self.num_act_my)
        Qf  = np.zeros((self.num_act_my,
                        self.num_act_op))

        # Obtain Predicting Trajectories
        ### Debug ###
        #print("ego follower select opt actions fcn()")
        #print(self.opp_car)
        #print(self.opp_car.s)

        #traj_ego = self.get_traj_ego(self.car_my.s,self.act_set_my)
        traj_ego = get_traj_ego(self.car_my.s,
                                self.act_set_my,
                                self.car_my.horizon,
                                self.car_my.merge_steps,
                                self.car_my.lane_width,
                                self.car_my.dynamics)
#        vis_traj(traj_ego,'red','dashed')
        
        traj_hum = get_hum_traj(self.car_op.s,
                                self.act_set_op,
                                self.car_op.horizon,
                                self.car_op.dynamics) 
#        vis_traj(traj_hum,'blue','dashed')

        for i in range(self.num_act_my):
            
            traj_ego0 = traj_ego[i,:,:]
            
            for j in range(self.num_act_op):
                
                traj_hum0 = traj_hum[j,:,:]
                
                Qf[i,j] = self.compute_reward(traj_ego0,traj_hum0,i)
            
            Qfi[i] = np.min(Qf[i,:])


#        vis_Qmatrix(Qf, self.action_set, self.opp_action_set, name = "ego follower Qf")
#        vis_Qmatrix(Qfi, self.action_set, np.array([1]), name = "ego follower Qfi")

        # argmax Qf
        traj_opt_idx = np.argmax(Qfi)
        traj_opt =     traj_ego[traj_opt_idx,:,:]
        acts_opt = self.act_set_my[traj_opt_idx,:]

#        if call_from_hum == False:
#            print("ego opt actions: ",opt_actions)

        return acts_opt, traj_opt


    def compute_reward(self,traj_ego,traj_hum,act_idx):

        s_pred = np.hstack((traj_ego.T,traj_hum.T))

        ##### Ego Follower  #####
        # Init
        R_distance = np.zeros(s_pred.shape[0])
        R_collision = -10
        R = 0

        ###### Collision Reward ##### (entire horizon)
        #if self.if_collision(s_pred) == 1:
        #    R_collision = -10
        R_collision = self.if_collision(s_pred)*R_collision

        ##### Distance and Merge Reward #####
        goal = [GOAL_X,2.5]

        for k in range(s_pred.shape[0]):
            x_ego = s_pred[k,0]
            y_ego = s_pred[k,1]

            if x_ego <= MERGE_START_X :
                R_distance[k] = -abs(x_ego - goal[0])/goal[0]
            elif x_ego > MERGE_START_X and x_ego <= MERGE_END_X:
                #R_distance[k] = -abs(x_ego - goal[0])/goal[0] + 100*abs(y_ego)/goal[1]
                R_distance[k] = -1/(abs(x_ego-MERGE_END_X)+1) + 1*abs(y_ego)/goal[1]
            elif x_ego > MERGE_END_X and y_ego >= 2:
                R_distance[k] = -abs(x_ego - goal[0])/goal[0]

        #print("{:4.4f}".format(R_distance.sum()),end=' ')

        #R = np.minimum(R_distance, R_collision)
        R = R_distance + R_collision

        # Log
        '''
        print(self.action_set[act_idx,:], end='\t')
        print("R dist:\t",R_distance,end='\t')
        print("R coli:\t",R_collision,end='\t')
        print("sum: {:4.5f}".format(R.sum()))
        '''

        #print("[{:4.4f},{:4.4f}]".format(R_distance.sum(), R_collision.sum()), end=' ')

        # Discount factor for each step
        R = R*self.discount

        # Cumulative reward over entire horizon
        R = R.sum()

        return R

    def if_collision(self,s_pred):
        #col_flag = 0
        col_flag = np.zeros(self.horizon)

        x_ego = s_pred[:,0]
        y_ego = s_pred[:,1]
        x_other = s_pred[:,4]
        y_other = s_pred[:,5]

        x_diff = (x_ego-x_other)
        y_diff = (y_ego-y_other)

        for k in range(self.horizon):
            if abs(x_diff[k]) >= MERGE_SAFE_DIST or abs(y_diff[k]) >= 2.4:
                col_flag[k] = 0
            else:
                col_flag[k] = 1
        
        return col_flag

