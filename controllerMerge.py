# Ego Leader Controller

import numpy as np
from itertools import product
from visualize import vis_Q
from controllerHF import ControllerHF

import copy
from visualize import *


dt = 0.1
MERGE_SAFE_DIST = 12

# For Merging Behavior Reward
GOAL_X = 200
MERGE_START_X = 50
MERGE_END_X = 175

# Helper #
IsVis = False 

class ControllerMerge:
    def __init__(self,name,horizon):
        self.name = name 
        self.horizon = horizon
        self.action = None
        self.action_set = None
        self.opp_action_set = None
        
        self.num_actions = None
        self.num_opp_actions = None

        # Add multiple opp cars
        self.my_car = None
        # Opp Cars are used to Obtain their states
        self.opp_car = None
        
        self.est_opp_cars = None

        ### Helper Data ###
        self.lane_width = None
        self.discount = np.zeros(self.horizon)

    def setup(self,my_car,opp_car,lane_width=2.5):
        # Opp Car for their States
        self.my_car = my_car
        self.opp_car = opp_car
        
        # Est cars for predicting their trajectories
        self.est_opp_cars = copy.deepcopy(self.opp_car)

        # Setupt opp_car controller (maybe)
        num_h_cars = len(opp_car)
        for i in range(num_h_cars):
                cont_hf_ego = ControllerHF('f_est_ego_'+str(i),self.horizon)
                self.est_opp_cars[i].controller = cont_hf_ego
                self.est_opp_cars[i].controller.setup(self.est_opp_cars[i], self.my_car, lane_width = self.lane_width)
        '''
        cont_hf1_ego = ControllerHF("follower1_est_ego", self.horizon)
        self.est_opp_cars[0].controller = cont_hf1_ego
        self.est_opp_cars[0].controller.setup(self.est_opp_cars[0], self.my_car,   lane_width=2.5)
        
        cont_hf2_ego = ControllerHF("follower2_est_ego", self.horizon)
        self.est_opp_cars[1].controller = cont_hf2_ego
        self.est_opp_cars[1].controller.setup(self.est_opp_cars[1], self.my_car,   lane_width=2.5)
        '''

        # action set (follower)
        self.action_set = self.get_all_actions(my_car.actionspace)
        self.num_actions = self.action_set.shape[0]

        # action set of opponent (leader)
        self.opp_action_set = self.get_all_actions(opp_car[0].actionspace)
        self.num_opp_actions = self.opp_action_set.shape[0]

        ### Helper Data ###
        self.lane_width = lane_width
        for i in range(self.horizon):
            self.discount[i] = pow(0.9,i)
        
    def select_action(self):

        # Ql1
        Ql1 = self.compute_Ql(0)
        Ql2 = self.compute_Ql(1)
        Ql3 = self.compute_Ql(2)
        #Ql2 = 0

        Ql = Ql1 + Ql2 + Ql3
        
        ###### Visualize #####
        if IsVis:
            vis_Qmatrix(Ql, self.action_set, np.array([1]), name = self.name)

        opt_traj_idx = np.argmax(Ql)
        opt_actions = self.action_set[opt_traj_idx,:]
        self.action = opt_actions[0]

        # Continue Mergine Step
        if self.my_car.IsMerging and self.my_car.MergeCount < self.my_car.Merge_Steps.shape[0]:
            print("controllerMerge overwrite to continue merging")
            self.action = 3

        return self.action

    def compute_Ql(self,opp_car_idx,debug = False):
        # follower opt actions and traj
        # print("est car pos",self.est_opp_cars[0].s)
        f_actions_opt, traj_f_opt = self.est_opp_cars[opp_car_idx].controller.select_opt_actions(call_from_ego=True)
        
        # Update Est opp cars pos
        self.est_opp_cars[opp_car_idx].s = copy.deepcopy(self.opp_car[opp_car_idx].s)
        
        traj_l = self.get_my_traj(self.my_car.s, self.action_set)

        Ql = np.zeros(self.num_actions) 

        for i in range(self.num_actions):
            traj_l_i = traj_l[i]
            Ql[i] = self.compute_returns(traj_l_i,traj_f_opt)

        return Ql

    def compute_returns(self,tra_l,tra_f):
        returns = 0

        # states predict: (time, (x,y,v,yaw,x_opp,y_opp,v_opp,yaw_opp))
        states_predict = np.hstack((tra_l.T,tra_f.T))

        returns = self.compute_reward(states_predict)

        return returns

    def compute_reward(self,s_pred):
        # NOTE if ego is leader, then highway is follower

        ##### Ego Leader #####
        R_distance = np.zeros(s_pred.shape[0])
        
        '''
        R_collision = -10*np.ones(self.horizon)
        R_collision = R_collision*(self.if_collision(s_pred))
        '''
        R_collision = 0
        # Collision Penalty over entire horizon
        if self.if_collision(s_pred) == 1:
            R_collision = -10

        goal = [GOAL_X,2.5]

        for k in range(s_pred.shape[0]):
            x_ego = s_pred[k,0]
            y_ego = s_pred[k,1]

            if x_ego <= MERGE_START_X :
                #R_distance[k] = -0.4 * abs(x_ego - goal[0]) - 10 * 2.5
                R_distance[k] = -abs(x_ego - goal[0])/goal[0] 
            elif x_ego > MERGE_START_X and x_ego <= MERGE_END_X:
                #R_distance[k] = -0.4 * abs(x_ego - goal[0]) - 10 * abs(y_ego - goal[1])
                R_distance[k] = -abs(x_ego - goal[0])/goal[0] + abs(y_ego)/goal[1]
            elif x_ego > MERGE_END_X and y_ego >= 2:
                #R_distance[k] = -0.4 * abs(x_ego - goal[0])
                R_distance[k] = -abs(x_ego - goal[0])/goal[0]

        R = np.minimum(R_distance, R_collision)

        # Discount factor for each step
        R = R*self.discount

        # Cumulative reward over entire horizon
        predict_reward = R.sum()

        return predict_reward

       
    def if_collision(self,s_pred):
        '''
        collision_flags = np.zeros(self.horizon)
        x_ego = s_pred[:,0]
        y_ego = s_pred[:,1]
        x_other = s_pred[:,4]
        y_other = s_pred[:,5]

        x_diff = abs(x_ego-x_other)
        y_diff = abs(y_ego-y_other)

        for i in range(0,self.horizon):
            if x_diff[i] >= MERGE_SAFE_DIST or y_diff[i] >= 2.4:
                collision_flags[i] = 0
            else:
                collision_flags[i] = 1
        return collision_flags
        '''
        x_ego = s_pred[:,0]
        y_ego = s_pred[:,1]
        x_other = s_pred[:,4]
        y_other = s_pred[:,5]

        x_diff = abs(x_ego-x_other)
        y_diff = abs(y_ego-y_other)

        if min(x_diff) >= MERGE_SAFE_DIST or min(y_diff) >= 2.4:
            return 0
        else:
            return 1

    ###########################################################################33
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



    def get_all_actions(self, actionspace):
        return np.array(list(product(actionspace, repeat=self.horizon)))
       

