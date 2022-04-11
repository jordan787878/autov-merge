import numpy as np
from car2 import CarVer2
from trajectory2 import *
from visual2 import *

##############################
# For Merging Behavior Reward
GOAL_X = 200
# For Collision Detect
MERGE_SAFE_DIST = 10 
##############################


class ControlHumanFoll:
    def __init__(self,car_my,car_ego,name):
        print("init human follower controller")
        self.name = name

        # Link My Car
        self.car_my = car_my

        # Link opp car to car_ego
        self.car_op = car_ego

        ########################################################

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
        print(self.num_act_op)

        # Discount Vector
        self.discount = np.zeros(self.horizon)
        for i in range(self.horizon):
            self.discount[i] = pow(0.9,i)

    ##################################################
    def select_action(self):
#        print(self.name, " select action")
#        if self.name == "human1 follower":
#            print("\n----- human1 follower select action() -----")
#            print("opp car states:\t", self.car_op.s)

        acts_opt, traj_opt = self.select_opt_actions(False, None)
        
#        if self.name == "human1 follower":
#            print("hum1 follower opt actions:\t",acts_opt)
#            print("hum1 follower opt trajs:\n",traj_opt)

        self.action = acts_opt[0]

        # random policy
        #self.action = np.random.randint(3)

        return self.action

    ###################################################
    def select_opt_actions(self, call_from_ego, actual_state):
        # Init Qli Matrix
        Qf = np.zeros((self.num_act_my,self.num_act_op))
        Qfi = np.zeros(self.num_act_my)

        #############################################################
        # Get human trajectory
        traj_hum = get_hum_traj(self.car_my.s,
                                self.act_set_my,
                                self.car_my.horizon,
                                self.car_my.dynamics)

        # Get Ego Trajectory
#        if self.name == "human1 follower":
#            print("flag:\t",call_from_ego)

        if call_from_ego == True:
            self.car_op.s = actual_state
#            if self.name == "human1 follower":
#                print("pass in s_ego:\t",self.car_op.s)

        traj_ego = get_traj_ego(self.car_op.s,
                                self.act_set_op,
                                self.car_op.horizon,
                                self.car_op.merge_steps,
                                self.car_op.lane_width,
                                self.car_op.dynamics)

#        vis_traj(traj_hum,traj_ego,'r','dashed')

        #############################################################

        # Compute Qf Matrix
        for i in range(self.num_act_my):
            traj_hum0 = traj_hum[i,:]
            for j in range(self.num_act_op):
                traj_ego0 = traj_ego[j,:]
                Qf[i,j] = self.compute_reward(traj_ego0,traj_hum0)
            Qfi[i] = np.min(Qf[i,:])

        '''
        vis_Qmatrix(Qf, self.act_set_my, self.act_set_op, self.name + " reward Matrix"  )
        vis_Qmatrix(Qfi, self.act_set_my, np.array([1]), self.name + " reward Matrix")
        '''

        # Max Qf 
        traj_opt_idx = np.argmax(Qfi)
        traj_opt = traj_hum[traj_opt_idx,:,:]
        acts_opt = self.act_set_my[traj_opt_idx,:]

        return acts_opt, traj_opt

    #######################################################
    def compute_reward(self,traj_ego,traj_hum):

        s_pred = np.hstack((traj_ego.T,traj_hum.T))

        ##### Highway Leader #####
        # Init
        R_distance = np.zeros(s_pred.shape[0])
        R_collision = -100
        R = 0

        ###### Collision Reward ##### (entire horizon)
        R_collision = self.if_collision(s_pred)*R_collision

        ##### Distance Reward #####
        goal = GOAL_X

        # Normalized Distance Reward for each step, (min: -1, max: 0)
        R_distance = -abs(s_pred[:,4] - goal)/(goal)

        # Reward
        #debug R = R_distance
        #R = R_distance + R_collision
        R = np.minimum(R_distance, R_collision)

        # Discount factor for each step
        R = R*self.discount

        # Cumulative reward over entire horizon
        R = R.sum()

        return R

    #######################################################
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





