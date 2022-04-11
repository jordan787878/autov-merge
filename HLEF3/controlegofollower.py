import numpy as np
from trajectory2 import * 
from visual2 import *

##############################
# For Merging Behavior Reward
GOAL_X = 200
MERGE_START_X = 25
MERGE_END_X = 175
# For Collision Detect
MERGE_SAFE_DIST = 10
##############################


class ControlEgoFollow:
    def __init__(self,car_my,car_op,name):
        print("init ego follower controller")
        self.name = name

        # Link Interacting Cars
        self.car_my = car_my
        self.car_op = car_op

        #####################################################3

        # Obtain horizon
        self.horizon = self.car_my.horizon

        # Init action
        self.action = None

        # Get action set
        self.act_set_my = self.car_my.get_all_ego_actions()
        self.act_set_op = self.car_op.get_all_actions()

        # Prepare num actions
        self.num_act_my = self.act_set_my.shape[0]
        self.num_act_op = self.act_set_op.shape[0]

        # Discount Vector
        self.discount = np.zeros(self.horizon)
        for i in range(self.horizon):
            self.discount[i] = pow(0.9,i)

    def select_action(self):
        acts_opt, traj_opt = self.select_opt_actions(False, None)

#        print("ego follower actual opt actions:\t",acts_opt)

        self.action = acts_opt[0]

        # Continue Mergine Step
        if self.car_my.IsMerging and self.car_my.MergeCount < self.car_my.merge_steps:
            self.action = 3

        return self.action

    
    def select_opt_actions(self, call_from_hum, actual_state):
#        if call_from_hum:
#            print("human lead -> ego folo.select_opt_actions()")
            #print("ego state:\t",self.car_my.s)
            #print("ego folo car op state:\t",self.car_op.s)
#        else:
#            print("ERROR !!!!!!!!!!!!!!!!!!")

        # Init Q Matrix
        Qfi = np.zeros(self.num_act_my)
        Qf  = np.zeros((self.num_act_my,
                        self.num_act_op))
        
        # Obtain Predicting Trajectories
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


        # Compute Q matrix
        for i in range(self.num_act_my):
            traj_ego0 = traj_ego[i,:,:]
            for j in range(self.num_act_op):
                traj_hum0 = traj_hum[j,:,:]
                Qf[i,j] = self.compute_reward(traj_ego0,traj_hum0,i)
            Qfi[i] = np.min(Qf[i,:])
        
        #print(Qfi)

        # Max Q
        traj_opt_idx = np.argmax(Qfi)
        traj_opt =     traj_ego[traj_opt_idx,:,:]
        acts_opt = self.act_set_my[traj_opt_idx,:]

#        if call_from_hum == False:
#            print("ego follow actual opt actions:\t",acts_opt)

        return acts_opt, traj_opt

    
    ######################################################################
    def compute_reward(self,traj_ego,traj_hum,act_idx):

        s_pred = np.hstack((traj_ego.T,traj_hum.T))

        ##### Ego Follower  #####
        # Init
        R_distance = np.zeros(s_pred.shape[0])
        R_collision = -100
        R = 0

        ###### Collision Reward ##### (entire horizon)
        #if self.if_collision(s_pred) == 1:
        #    R_collision = -10
        R_collision = self.if_collision(s_pred)*R_collision
        #print(act_idx,R_collision)

        ##### Distance and Merge Reward #####
        goal = [GOAL_X,2.5]
        w_yield = 1
        w_merge = 10
        for k in range(s_pred.shape[0]):
            x_ego = s_pred[k,0]
            y_ego = s_pred[k,1]
            x_hum = s_pred[k,4]
            
            if x_ego >= MERGE_START_X and x_ego <= MERGE_END_X and y_ego < 2.5:
                #R_distance[k] = -w_yield*(x_ego-x_hum) + w_merge*abs(y_ego)/goal[1]
                R_distance[k] = -abs(x_ego-goal[0])/goal[0] - w_merge * abs(y_ego-goal[1])/goal[1]
            else:
                R_distance[k] = -abs(x_ego-goal[0])/goal[0]

        #R = R_distance + R_collision
        R = np.minimum(R_distance,R_collision)

        # Discount factor for each step
        R = R*self.discount

        # Cumulative reward over entire horizon
        R = R.sum()

        return R

        '''
                       if x_ego <= MERGE_START_X :
                R_distance[k] = -abs(x_ego - goal[0])/goal[0]
            elif x_ego > MERGE_START_X and x_ego <= MERGE_END_X:
                #R_distance[k] = -abs(x_ego - goal[0])/goal[0] + 100*abs(y_ego)/goal[1]
#                R_distance[k] = -1/(abs(x_ego-MERGE_END_X)+1) + 1*abs(y_ego)/goal[1]
#                R_distance[k] = -abs(x_ego - goal[0])/goal[0] + 1*abs(y_ego)/goal[1]
                R_distance[k] = -w_yield*(x_ego - x_hum) + w_merg * abs(y_ego)/goal[1]
            elif x_ego > MERGE_END_X and y_ego >= 2:
                R_distance[k] = -abs(x_ego - goal[0])/goal[0]
        '''

    ##################################################################3
    def if_collision(self,s_pred):
        col_flag = np.zeros(self.horizon)

        x_ego = s_pred[:,0]
        y_ego = s_pred[:,1]
        x_other = s_pred[:,4]
        y_other = s_pred[:,5]

        x_diff = (x_ego-x_other)
        y_diff = (y_ego-y_other)

        for k in range(self.horizon):
            if abs(x_diff[k]) >= MERGE_SAFE_DIST or abs(y_diff[k]) > 2.0:
                col_flag[k] = 0
            else:
                col_flag[k] = 1

        return col_flag

