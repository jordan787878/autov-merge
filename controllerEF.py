# Ego Follower Controller

import numpy as np
from itertools import product
import controllerHL
from car import Car

import copy
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
    def __init__(self,name,horizon):
        self.name = name
        self.horizon = horizon
        self.action = None
        self.action_set = None
        self.opp_action_set = None
        
        self.num_actions = None
        self.num_opp_actions = None

        self.my_car = None
        self.opp_car = None

        ### Estimated Copy of Opp. Car ###
        self.est_opp_car = None

        ### Helper Data ###
        self.lane_width = None
        self.discount = np.zeros(self.horizon)


    def setup(self,my_car,opp_car,lane_width=2.5):
        print("Set up Ego Follower Controller")
        # Opp Car for their States
        self.my_car = my_car
        self.opp_car = opp_car

        # Est opp Car for Trajectory Prediction
        #self.est_opp_car = copy.deepcopy(self.opp_car)

        # action set (follower)
        self.action_set = self.get_all_actions(my_car.actionspace)
        self.num_actions = self.action_set.shape[0]

        # action set of opponent (leader)
        self.opp_action_set = self.get_all_actions(opp_car.actionspace)
        self.num_opp_actions = self.opp_action_set.shape[0]

        self.lane_width = lane_width
        for i in range(self.horizon):
            self.discount[i] = pow(0.9,i)


    def setup_est_opp_control(self):
        print("Ego Follower setup Estimate Human Leader")
        self.est_opp_car = copy.deepcopy(self.opp_car)
        
    def select_action(self):
        opt_actions, opt_traj = self.select_opt_actions()
        
        self.action = opt_actions[0]

        # Continue Mergine Step
        if self.my_car.IsMerging and self.my_car.MergeCount < self.my_car.Merge_Steps.shape[0]:
            print("controllerMerge overwrite to continue merging")
            self.action = 3
        
        return self.action

    def select_opt_actions(self,call_from_hum=False):
#        if call_from_hum == True:
#            print("ego select opt actions call from human")
#            print(self.num_actions)
#            print(self.num_opp_actions)
        
        # Init Q Matrix
        Qfi = np.zeros(self.num_actions)
        Qf = np.zeros((self.num_actions,self.num_opp_actions))

        # Obtain Predicting Trajectories
        ### Debug ###
        #print("ego follower select opt actions fcn()")
        #print(self.opp_car)
        #print(self.opp_car.s)

        traj_ego = self.get_my_traj(self.my_car.s,self.action_set)
        traj_hum = self.get_opp_traj(self.opp_car.s,self.opp_action_set)

        # Update Est opp car Pos
        #print("update est opp car pos")
        #self.est_opp_car.s = copy.deepcopy(self.opp_car.s)

        for i in range(self.num_actions):
            
            traj_ego_i = traj_ego[i,:,:]
            
            for j in range(self.num_opp_actions):
                
                traj_hum_j = traj_hum[j,:,:]
                
                Qf[i,j] = self.compute_reward(traj_ego_i,traj_hum_j,i)
            
            Qfi[i] = np.min(Qf[i,:])


#        vis_Qmatrix(Qf, self.action_set, self.opp_action_set, name = "ego follower Qf")
#        vis_Qmatrix(Qfi, self.action_set, np.array([1]), name = "ego follower Qfi")

        # argmax Qf
        opt_traj_idx = np.argmax(Qfi)
        opt_actions = self.action_set[opt_traj_idx,:]
        opt_traj = traj_ego[opt_traj_idx,:,:]

#        if call_from_hum == False:
#            print("ego opt actions: ",opt_actions)

        return opt_actions, opt_traj


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

    def get_all_actions(self, actionspace):
        return np.array(list(product(actionspace, repeat=self.horizon)))

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

    ### helper function ###

    def get_my_traj(self,s0,act_set):
        num_traj = act_set.shape[0]
        traj = np.zeros((num_traj,s0.shape[0],self.horizon))
        traj[:,:,0] = s0

        for i in range(num_traj):
            acti = act_set[i,:]

            # Init
            my_car_IsMerging = False
            my_car_MergeCount = 0

            for k in range(self.horizon-1):
                x, y, v, yaw = traj[i,:,k]
                ### HighWay Follower Dynamics ###
                act = acti[k]

                # Ego Vehicle Dynamics Prediction
                # Overwirte Merging Action
                if act != 3:
                    if my_car_IsMerging and my_car_MergeCount < self.my_car.Merge_Steps.shape[0]:
                        act = 3
                #print("overwrite: ",k," act:\t",act)

                # Apply Action
                if act == 0:
                    u = -1*self.my_car.accel
                elif act == 1:
                    u = 0
                elif act == 2:
                    u = 1*self.my_car.accel
                else:
                    if my_car_MergeCount == 0:
                        u = 0
                        y += self.my_car.Merge_Steps[0]
                        y = np.clip(y,0,2.5)
                        my_car_IsMerging = True
                        my_car_MergeCount += 1
                    elif my_car_MergeCount < self.my_car.Merge_Steps.shape[0]:
                        u = 0
                        y += self.my_car.Merge_Steps[0]
                        y = np.clip(y,0,2.5)
                        my_car_MergeCount += 1
                    else:
                        u = 0

                v = v + u*dt
                v = np.clip(v,self.my_car.min_v,self.my_car.max_v)
                x = x + v*dt
                traj[i,:,k+1] = np.array([x,y,v,yaw])

        return traj

    def get_opp_traj(self,s0,act_set):
        num_traj = act_set.shape[0]
        traj = np.zeros((num_traj,s0.shape[0],self.horizon))
        traj[:,:,0] = s0

        for i in range(num_traj):
            acti = act_set[i,:]
            for k in range(self.horizon-1):
                x, y, v, yaw = traj[i,:,k]
                ### HighWay Follower Dynamics ###
                act = acti[k]
                # Decl
                if act == 0:
                    u = -1*self.opp_car.accel
                # Cruise
                elif act == 1:
                    u = 0
                # Acel
                else:
                    u = 1*self.opp_car.accel

                v = v + u*dt
                v = np.clip(v,self.opp_car.min_v,self.opp_car.max_v)
                x = x + v*dt
                traj[i,:,k+1] = np.array([x,y,v,yaw])

        return traj


    def get_all_actions(self, actionspace):
        return np.array(list(product(actionspace, repeat=self.horizon)))

