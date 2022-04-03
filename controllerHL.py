# Human Leader Controller

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from controllerEF import ControllerEF
from car import Car


from visualize import *

import copy

dt = 0.1
MERGE_SAFE_DIST = 12 

### NOTE: TO DO Parameterize the Goal X ###
GOAL_X = 200 

# Helper #
IsVis = False 

class ControllerHL:
    def __init__(self,name,horizon):
        print("init HL controller")
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
        print("Set up HL controller")
        # Opp Car for their States
        self.my_car = my_car
        self.opp_car = opp_car

        # action set
        self.action_set = self.get_all_actions(my_car.actionspace)
        self.num_actions = self.action_set.shape[0]

        # action set of opponent
        self.opp_action_set = self.get_all_actions(opp_car.actionspace)
        self.num_opp_actions = self.opp_action_set.shape[0]

        self.lane_width = lane_width
        for i in range(self.horizon):
            self.discount[i] = pow(0.9,i)


    def setup_est_opp_control(self):
        self.est_opp_car = self.opp_car


    def select_action(self):
        opt_actions, opt_traj = self.select_opt_actions()
        
        self.action = opt_actions[0]

        # random policy
        #self.action = np.random.randint(3)

        return self.action

    def select_opt_actions(self, call_from_ego=False):
        Qfi = np.zeros(self.num_actions)

        # Obtain optimal ego follower trajectory
        act_ego_opt, traj_ego_opt = self.est_opp_car.controller.select_opt_actions(call_from_hum=True)
        print("est_ego_opt actions: ",act_ego_opt)

        # Update Est opp car Pos
        self.est_opp_car.s = copy.deepcopy(self.opp_car.s)

        # Predict my trajectories
        traj_hum = self.get_my_traj(self.my_car.s,self.action_set)

#        vis_traj(traj_hum,'r','dashed')

        for i in range(self.num_actions):
            traj_hum_i = traj_hum[i,:] 
            
            Qfi[i] = self.compute_reward(traj_ego_opt,traj_hum_i)
            
        # Visualize Qf Matrix
#        vis_Qmatrix(Qfi, self.action_set, np.array([1]), name = "human1 leader Qfi")
      
        # argmax Qf
        opt_traj_idx = np.argmax(Qfi)
        opt_actions = self.action_set[opt_traj_idx,:]
        opt_traj = traj_hum[opt_traj_idx,:,:]
        
        return opt_actions, opt_traj


    def compute_reward(self,traj_ego,traj_hum):
       
        s_pred = np.hstack((traj_ego.T,traj_hum.T))

        ##### Highway Leader #####
        # Init
        R_distance = np.zeros(s_pred.shape[0])
        R_collision = -10
        R = 0

        ###### Collision Reward ##### (entire horizon)
        R_collision = self.if_collision(s_pred)*R_collision

        ##### Distance Reward #####
        goal = GOAL_X
        
        # Normalized Distance Reward for each step, (min: -1, max: 0)
        R_distance = -abs(s_pred[:,4] - goal)/(goal)

        # Reward
        R = R_distance
        #R = np.minimum(R_distance, R_collision)

        # Discount factor for each step
        R = R*self.discount

        # Cumulative reward over entire horizon
        R = R.sum()
        
        return R


    def if_collision(self,s_pred):
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


    def get_all_actions(self, actionspace):
        return np.array(list(product(actionspace, repeat=self.horizon)))
    
    '''# Old Vesrion: entire horizon collision check #
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
    '''


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

