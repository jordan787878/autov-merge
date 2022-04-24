import numpy as np
import copy
from car2 import CarVer2
from trajectory2 import *
from visual2 import *

##############################
# For Merging Behavior Reward
GOAL_X = 300
# For Collision Detect
MERGE_SAFE_DIST = 15
##############################


class ControlHumanLead:
    def __init__(self,car_my,car_op, name):
#        print("init human leader controller")
        self.name = name

        # Link My Car
        self.car_my = car_my
        self.car_op = car_op

        # Check
#        print("human lead control check link")
#        print("human lead ego controller state:\t",self.car_op.s)
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
        self.discount = np.zeros(self.horizon+1)
        for i in range(self.horizon+1):
            self.discount[i] = pow(0.9,i)

    ##################################################
    def select_action(self):
#        print("### human leader select actions ###")

        acts_opt, traj_opt = self.select_opt_actions(False, None)

#        print("human leader   action: ",acts_opt)

        self.action = acts_opt[0]

        # random policy
        #self.action = np.random.randint(3)

        return self.action

    ###################################################
    def select_opt_actions(self, call_from_ego, actual_state):
        if call_from_ego:
            self.car_op.s = copy.deepcopy(actual_state)

        # Get human trajectory
        traj_hum = get_hum_traj(self.car_my.s,self.act_set_my,self.car_my.horizon,self.car_my.dynamics)
        # Get Ego Traj
        traj_ego = get_traj_ego(self.car_op.s,self.act_set_op,self.car_op.horizon,
                                self.car_op.merge_steps,self.car_op.lane_width,self.car_op.dynamics)

        # Init Qf Matrix
        Qfi = self.compute_Qf(traj_ego, traj_hum, call_from_ego)
        traj_ego_opt_idx = np.argmax(Qfi)
        traj_ego_opt = traj_ego[traj_ego_opt_idx,:,:]
        acts_ego_opt = self.act_set_op[traj_ego_opt_idx,:]
#        print("ego   follower action: ", acts_ego_opt)
#        vis_hum_lead(Qfi, "ego follower Qfi", self.act_set_op)

        # Init Ql matrix
        Ql = np.zeros(traj_hum.shape[0])
        for i in range(traj_hum.shape[0]):
            Ql[i] = self.compute_reward(traj_ego_opt, traj_hum[i])

#            print(i,self.act_set_my[i],Ql[i])

        traj_opt_idx = np.argmax(Ql)
        traj_opt = traj_hum[traj_opt_idx,:,:]
        acts_opt = self.act_set_my[traj_opt_idx,:]
#        print(traj_opt_idx, acts_opt)
#        if call_from_ego == False:
#            vis_hum_lead(Ql,"human leader Ql",self.act_set_my)

        return acts_opt, traj_opt


    ##################################################
    def compute_Qf(self, traj_ego, traj_hum, call_from_ego):
        Qf = np.zeros((traj_ego.shape[0],traj_hum.shape[0]))
        Qfi = np.zeros(traj_ego.shape[0])

#        print(call_from_ego,traj_ego[0,0,:])
#        print(call_from_ego,traj_hum[0,1,:])
        for i in range(traj_ego.shape[0]):
            for j in range(traj_hum.shape[0]):
                Qf[i,j] = self.compute_ego_reward(traj_ego[i],traj_hum[j])
            Qfi[i] = np.min(Qf[i,:])
            #print(Qfi[i],end= ' ')
        #print(call_from_ego)
#            print(i,self.act_set_op[i],Qfi[i])
        
        ### Test ###
#        vis_ego_follow(Qfi,"ego follower Qfi",self.act_set_op)
        
        return Qfi

    def compute_ego_reward(self, traj_ego, traj_hum):
        # Init
        R_distance = np.zeros(traj_ego.shape[1])
        R_collision = -100
        R = 0

        ###### Collision Reward ##### (entire horizon)
        R_collision = self.if_collision(traj_ego,traj_hum)*R_collision

        ##### Distance Reward #####
        goal = GOAL_X

        # Normalized Distance Reward for each step, (min: -1, max: 0)
        
        #R_distance = -abs(traj_hum[0,:] - goal)/(goal)
        for k in range(traj_ego.shape[1]):
            x_ego = traj_ego[0,k]
            y_ego = traj_ego[1,k]
            if x_ego >= 25 and x_ego <= 175 and y_ego < 0.5:
                R_distance[k] = -abs(x_ego-goal)/goal -10*abs(y_ego-2.5)/2.5
            else:
                R_distance[k] = -abs(x_ego-goal)/goal
            if x_ego >= 100 and y_ego < 0.5:
                R_distance[k] = -100

        R = np.minimum(R_distance, R_collision)

        R = R*self.discount

        R = R.sum()

        return R
    
    #######################################################
    def compute_reward(self,traj_ego,traj_hum):
        # Init
        R_distance = np.zeros(traj_ego.shape[1])
        R_collision = -100
        R = 0

        ###### Collision Reward ##### (entire horizon)
        R_collision = self.if_collision(traj_ego,traj_hum)*R_collision

        ##### Distance Reward #####
        goal = GOAL_X

        # Normalized Distance Reward for each step, (min: -1, max: 0)
        R_distance = -abs(traj_hum[0,:] - goal)/(goal)

        x_ego = traj_ego[0,:]
        y_ego = traj_ego[1,:]
        x_other = traj_hum[0,:]
        y_other = traj_hum[1,:]

        x_diff = (x_ego-x_other)
        y_diff = (y_ego-y_other)
#        print(x_ego,' ', y_ego)
#        print(x_diff, ' ', y_diff, ' ', R_collision)
#        print("\n")

        R = np.minimum(R_distance, R_collision)

        R = R*self.discount

        R = R.sum()

        return R



    #######################################################
    def if_collision(self,traj_ego,traj_hum):
        col_flag = np.zeros(self.horizon+1)

        x_ego = traj_ego[0,:]
        y_ego = traj_ego[1,:]
        x_other = traj_hum[0,:]
        y_other = traj_hum[1,:]

        x_diff = (x_ego-x_other)
        y_diff = (y_ego-y_other)

        for k in range(1,self.horizon+1):
            if abs(x_diff[k]) <= MERGE_SAFE_DIST and abs(y_diff[k]) < self.car_my.lane_width:
                col_flag[k] = 1
            else:
                col_flag[k] = 0

        # TEST Collision Flog over Entire Horizon
        if col_flag.any() == 1:
            col_flag = np.ones(self.horizon+1)


        return col_flag





