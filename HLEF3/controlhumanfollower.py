import numpy as np
from car2 import CarVer2
from trajectory2 import *
from visual2 import *
from utility import *

##############################
# For Merging Behavior Reward
GOAL_X = 300
# For Collision Detect
MERGE_SAFE_DIST = 10 
##############################


class ControlHumanFoll:
    def __init__(self,car_my,car_ego,name):
#        print("init human follower controller")
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
#        print(self.num_act_op)

        # Discount Vector
        self.discount = np.zeros(self.horizon+1)
        for i in range(self.horizon+1):
            self.discount[i] = pow(0.9,i)

        # Goal X
        self.goal_x = GOAL_X

    ##################################################
    def select_action(self):
#        print("### human follower select actions ###")
    
        #print("human1 real state: ", self.car_my.s)

        # Update Parameters
        self.update_goal_x()


        acts_opt, traj_opt = self.select_opt_actions(False, None)

        #print("human1 real actions: ", acts_opt)
        
        self.action = acts_opt[0]
        
#        print("human1 real action: ", self.action)

#        print("human follower action: ",acts_opt)

        # random policy
        #self.action = np.random.randint(3)

        # 0410_Problem2 Hotfix (If ego merge behind me)
        #if self.car_op.s[1] > 0 and self.car_op.s[0] < self.car_my.s[0]:
        #    self.action = 2


        return self.action

    ###################################################
    def select_opt_actions(self, call_from_ego, actual_state):
        # Init Qli Matrix
        #Qf = np.zeros((self.num_act_my,self.num_act_op))
        #Qfi = np.zeros(self.num_act_my)
        
        # Get human trajectory
        traj_hum = get_hum_traj(self.car_my.s,
                                self.act_set_my,
                                self.car_my.horizon,
                                self.car_my.dynamics)

        traj_ego = get_traj_ego(self.car_op.s,
                                self.act_set_op,
                                self.car_op.horizon,
                                self.car_op.merge_steps,
                                self.car_op.lane_width,
                                self.car_op.dynamics)

        Qfi = Qf_hum(traj_ego, traj_hum, self.goal_x)
        traj_opt_idx = np.argmax(Qfi)
        traj_opt = traj_hum[traj_opt_idx,:,:]
        acts_opt = self.act_set_my[traj_opt_idx,:]

        return acts_opt, traj_opt

        # Old
        #R_colli = np.zeros((self.num_act_my,self.num_act_op))
        #R_col = np.zeros(self.num_act_my)

        # Compute Qf Matrix
        #for i in range(self.num_act_my):
        #    traj_hum0 = traj_hum[i,:]
        #    for j in range(self.num_act_op):
        #        traj_ego0 = traj_ego[j,:]
        #        Qf[i,j], R_debug = self.compute_reward(traj_ego0,traj_hum0)
        #        R_colli[i,j] = R_debug
        #    Qfi[i] = np.min(Qf[i,:])
        #    R_col[i] = np.min(R_colli[i,:])  

   
    def update_goal_x(self):
        x0 = self.car_my.s[0]
        if x0 >= (self.goal_x - 150):
            self.goal_x += 300

    #######################################################
    '''
    def compute_reward(self,traj_ego,traj_hum):
        ##### Highway Leader #####
        # Init
        R_distance = np.zeros(traj_ego.shape[1])
        R_collision = -100
        R = 0

        ###### Collision Reward ##### (entire horizon)
        R_collision = self.if_collision(traj_ego,traj_hum)*R_collision
        R_colli = R_collision.sum() 
        #print(R_collision)

        ##### Distance Reward #####
        goal = GOAL_X

        # Normalized Distance Reward for each step, (min: -1, max: 0)
        R_distance = -abs(traj_hum[0,:] - goal)/(goal)

        # Reward
        R = np.minimum(R_distance, R_collision)

        # Discount factor for each step
        R = R*self.discount

        # Cumulative reward over entire horizon
        R = R.sum()

        return R, R_colli 

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
    '''




