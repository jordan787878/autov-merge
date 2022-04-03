# Human Leader Controller

import numpy as np
import matplotlib.pyplot as plt
from controllerEF import ControllerEF
from car import Car


from visualize import *
from trajectory import *

import copy

dt = 0.1
MERGE_SAFE_DIST = 12 

### NOTE: TO DO Parameterize the Goal X ###
GOAL_X = 200 

# Helper #
IsVis = False 

class ControllerHL:
    def __init__(self,name,car_my,car_op):
        print("init HL controller")
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

        self.action = acts_opt[0]

        # random policy
        #self.action = np.random.randint(3)

        return self.action

    def select_opt_actions(self, call_from_ego=False):
        # Init Qli Matrix
        Qli = np.zeros(self.num_act_my)

        # Get optimal ego follower traj
        acts_ego_opt, traj_ego_opt = self.car_op.controller.select_opt_actions()
#        print("estimate ego follower opt. actions\t",acts_ego_opt)

        # Get human trajectory
        traj_hum = get_hum_traj(self.car_my.s,
                                self.act_set_my,
                                self.car_my.horizon,
                                self.car_my.dynamics)

        for i in range(self.num_act_my):
            traj_hum0 = traj_hum[i,:]
            Qli[i] = self.compute_reward(traj_ego_opt,traj_hum0)

        # argmax Qli
        traj_opt_idx = np.argmax(Qli)
        traj_opt = traj_hum[traj_opt_idx,:,:]
        acts_opt = self.act_set_my[traj_opt_idx,:]

        return acts_opt, traj_opt

        '''

        # Obtain optimal ego follower trajectory
        # act_ego_opt, traj_ego_opt = self.est_opp_car.controller.select_opt_actions(call_from_hum=True)
        act_ego_opt, traj_ego_opt = self.opp_car.controller.select_opt_actions(call_from_hum=True)
        print("est_ego_opt actions: ",act_ego_opt)

        # Update Est opp car Pos
#        self.est_opp_car.s = copy.deepcopy(self.opp_car.s)

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
        '''

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
    '''
    def get_all_actions(self):
        perm_actions = np.array(list(product(self.car_my.actionspace, repeat=self.horizon)))
        return perm_actions
    '''
    ''' 
    def get_all_actions(self, actionspace):
        return np.array(list(product(actionspace, repeat=self.horizon)))
    '''

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


