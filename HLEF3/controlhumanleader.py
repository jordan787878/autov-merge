import numpy as np
import copy
from car2 import CarVer2
from trajectory2 import *
from visual2 import *

##############################
# For Merging Behavior Reward
GOAL_X = 200
# For Collision Detect
MERGE_SAFE_DIST = 3
##############################


class ControlHumanLead:
    def __init__(self,car_my,car_op, name):
        print("init human leader controller")
        self.name = name

        # Link My Car
        self.car_my = car_my
        self.car_op = car_op

        # Check
        print("human lead control check link")
        print("human lead ego controller state:\t",self.car_op.s)
        ########################################################

        # Obtain horizon
        self.horizon = self.car_my.horizon

        # Init action
        self.action = None

        # Get action set
        self.act_set_my = self.car_my.get_all_actions()
        self.act_set_op = self.car_op.get_all_ego_actions()

        # Prepare num actions
        self.num_act_my = self.act_set_my.shape[0]
        self.num_act_op = self.act_set_op.shape[0]

        # Discount Vector
        self.discount = np.zeros(self.horizon)
        for i in range(self.horizon):
            self.discount[i] = pow(0.9,i)

    ##################################################
    def select_action(self):
        #print("human lead ego states: ",self.car_op.s)

        acts_opt, traj_opt = self.select_opt_actions(False, None)

#        print("human lead opt action: ",acts_opt)

        self.action = acts_opt[0]

        # random policy
        #self.action = np.random.randint(3)

        return self.action

    ###################################################
    def select_opt_actions(self, call_from_ego, actual_state):
        # Get human trajectory
        traj_hum = get_hum_traj(self.car_my.s,
                                self.act_set_my,
                                self.car_my.horizon,
                                self.car_my.dynamics)
        # Get Ego Traj
        traj_ego = get_traj_ego(self.car_op.s,
                                self.act_set_op,
                                self.car_op.horizon,
                                self.car_op.merge_steps,
                                self.car_op.lane_width,
                                self.car_op.dynamics)

        # Init Qli Matrix
        Ql = np.zeros((self.num_act_my,self.num_act_op))
        Qli = np.zeros(self.num_act_my)

        # Compute Ql Matrix
        for i in range(self.num_act_my):
            traj_hum0 = traj_hum[i,:]
            for j in range(self.num_act_op):
                traj_ego0 = traj_ego[j,:]
                Ql[i,j] = self.compute_reward(traj_ego0,traj_hum0)
            Qli[i] = np.min(Ql[i,:])


#        vis_Qmatrix(Qf, self.act_set_my, self.act_set_op, "human2 follower Qf")
#        vis_Qmatrix(Qfi, self.act_set_my, np.array([1]), "human2 follower Qfi")

        # Max Qf
        traj_opt_idx = np.argmax(Qli)
        traj_opt = traj_hum[traj_opt_idx,:,:]
        acts_opt = self.act_set_my[traj_opt_idx,:]


        acts_opt = self.act_set_my[-1];
        traj_opt = traj_hum[-1]

        return acts_opt, traj_opt

    #######################################################
    def compute_reward(self,traj_ego,traj_hum):

        s_pred = np.hstack((traj_ego.T,traj_hum.T))

        ##### Highway Leader #####
        # Init
        R_distance = np.zeros(s_pred.shape[0])
        R_collision = -1000
        R = 0

        ###### Collision Reward ##### (entire horizon)
        R_collision = self.if_collision(s_pred)*R_collision

        ##### Distance Reward #####
        goal = GOAL_X

        # Distance Reward for each step
        R_distance = -abs(s_pred[:,4] - goal) 

        # Reward
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
            if abs(x_diff[k]) >= MERGE_SAFE_DIST or abs(y_diff[k]) > 1.0:
                col_flag[k] = 0
            else:
                col_flag[k] = 1

        return col_flag





