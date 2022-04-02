# Human Follwer Controller

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from visualize import *

import copy

dt = 0.1
MERGE_SAFE_DIST = 12 

# Debug
R_COL = []

### NOTE: TO DO Parameterize the Goal X ###
GOAL_X = 200 

# Helper #
IsVis = False 

class ControllerHF:
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
        # Opp Car for their States
        self.my_car = my_car
        self.opp_car = opp_car
        
        # Est opp Car for Trajectory Prediction
        self.est_opp_car = copy.deepcopy(self.opp_car)

        # action set (follower)
        self.action_set = self.get_all_actions(my_car.actionspace)
        self.num_actions = self.action_set.shape[0]

        # action set of opponent (leader)
        self.opp_action_set = self.get_all_actions(opp_car.actionspace)
        self.num_opp_actions = self.opp_action_set.shape[0]

        self.lane_width = lane_width
        for i in range(self.horizon):
            self.discount[i] = pow(0.9,i)


    def show_traj(self,tra,ax,c,ls):
        x0, y0 = tra[0,0:2,0]
        ax.scatter(x0,y0,color=c,linestyle=ls)
        for i in range(tra.shape[0]):
            trai = tra[i,0:2,:]
            ax.plot(trai[0,:],trai[1,:],color=c)

    def select_action(self):
        opt_actions, opt_traj = self.select_opt_actions()
        self.action = opt_actions[0]
        return self.action

    # bool call_from_ego is the flag to differentiate who call this function

    def select_opt_actions(self, call_from_ego=False):
        Qfi = np.zeros(self.num_actions)
        Qf = np.zeros((self.num_actions,self.num_opp_actions))

        # Obtain Predicting Trajectories
        traj_f = self.get_my_traj(self.my_car.s,self.action_set)
        traj_l = self.get_opp_traj(self.est_opp_car.s,self.opp_action_set)

        # Update Est opp car Pos
        self.est_opp_car.s = copy.deepcopy(self.opp_car.s)
        
        for i in range(self.num_actions):
            traj_f_i = traj_f[i,:,:]

            for j in range(self.num_opp_actions):
                traj_l_j = traj_l[j,:,:]
                Qf[i,j] = self.compute_returns(traj_l_j,traj_f_i,i,j)
                
            Qfi[i] = np.min(Qf[i,:])

        # Visualize Qf Matrix
        if IsVis and self.my_car.id == "human2":
            vis_Qmatrix(Qf, self.action_set, self.opp_action_set, name = self.name)
            R_COL.clear()
      
        # argmax Qf
        opt_traj_idx = np.argmax(Qfi)
        opt_actions = self.action_set[opt_traj_idx,:]
        opt_traj = traj_f[opt_traj_idx,:,:]
        
        return opt_actions, opt_traj

    def compute_returns(self,tra_l,tra_f,idx_i,idx_j):
        returns = 0
        
        # states predict: (time, (x,y,v,yaw,x_opp,y_opp,v_opp,yaw_opp))
        states_predict = np.hstack((tra_f.T,tra_l.T))
        
        returns = self.compute_reward(states_predict,idx_i,idx_j)

        return returns
   
    def compute_reward(self,s_pred,idx_i,idx_j):
        # NOTE if highway is follower, then ego is leader
        
        ##### Highway Follwer #####
        goal = GOAL_X

        ##################################################
        R_collision = 0

        # Collision Penalty over entire horizon
        if self.if_collision(s_pred,idx_i,idx_j) == 1:
            R_collision = -10

        if self.my_car.id == "human2":
            R_COL.append(R_collision)

        ##################################################

        # Normalized Distance Reward for each step, (min: -1, max: 0)
        R_distance = -abs(s_pred[:,0] - goal)/(goal)

        # Reward
        #R = R_distance
        R = np.minimum(R_distance, R_collision)

        # Discount factor for each step
        R = R*self.discount

        # Cumulative reward over entire horizon
        predict_reward = R.sum()
        
        return predict_reward

    def get_all_actions(self, actionspace):
        return np.array(list(product(actionspace, repeat=self.horizon)))
        
    def if_collision(self,s_pred,idx_i,idx_j):
        col_flag = 0

        x_ego = s_pred[:,4]
        y_ego = s_pred[:,5]
        x_other = s_pred[:,0]
        y_other = s_pred[:,1]

        x_diff = (x_ego-x_other)
        y_diff = (y_ego-y_other)

        hf_action = self.action_set[idx_i,:]
        el_action = self.opp_action_set[idx_j,:]

        if min(abs(x_diff)) >= MERGE_SAFE_DIST or min(abs(y_diff)) >= 2.4:
            #return 0
            col_flag = 0
        else:
            #return 1
            col_flag = 1

        ### Visualize ###
        #if self.my_car.id == "human2":
        #    vis_collision_reward(x_diff,y_diff,hf_action,el_action,col_flag)

        return col_flag

    def get_my_traj(self,s0,act_set):
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
                    u = -1*self.my_car.accel
                # Cruise
                elif act == 1:
                    u = 0
                # Acel
                else:
                    u = 1*self.my_car.accel

                v = v + u*dt
                v = np.clip(v,self.my_car.min_v,self.my_car.max_v)
                x = x + v*dt
                traj[i,:,k+1] = np.array([x,y,v,yaw])

            #print("action\n",acti)
            #print("traj\n",traj[i,:,:])

        return traj

    def get_opp_traj(self,s0,act_set):
        num_traj = act_set.shape[0]
        traj = np.zeros((num_traj,s0.shape[0],self.horizon))
        traj[:,:,0] = s0

        for i in range(num_traj):
            acti = act_set[i,:]
            
            # Init
            opp_car_IsMerging = False
            opp_car_MergeCount = 0

            # Loop over time horizon
            for k in range(self.horizon-1):
                x, y, v, yaw = traj[i,:,k]
                ### HighWay Follower Dynamics ###
                act = acti[k]

                # Ego Vehicle Dynamics Prediction
                # Overwirte Merging Action
                if act != 3:
                    if opp_car_IsMerging and opp_car_MergeCount < self.est_opp_car.Merge_Steps.shape[0]:
                        act = 3

                #print("overwrite: ",k," act:\t",act)

                # Apply Action
                if act == 0:
                    u = -1*self.est_opp_car.accel
                elif act == 1:
                    u = 0
                elif act == 2:
                    u = 1*self.est_opp_car.accel
                else:
                    if opp_car_MergeCount == 0:
                        u = 0
                        y += self.est_opp_car.Merge_Steps[0]
                        y = np.clip(y,0,2.5)
                        opp_car_IsMerging = True
                        opp_car_MergeCount += 1
                    elif opp_car_MergeCount < self.est_opp_car.Merge_Steps.shape[0]:
                        u = 0
                        y += self.est_opp_car.Merge_Steps[0]
                        y = np.clip(y,0,2.5)
                        opp_car_MergeCount += 1
                    else:
                        u = 0
                
                v = v + u*dt
                v = np.clip(v,self.est_opp_car.min_v,self.est_opp_car.max_v)
                x = x + v*dt
                traj[i,:,k+1] = np.array([x,y,v,yaw])

        return traj


