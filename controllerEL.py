# Ego Leader Controller

import numpy as np
from itertools import product

dt = 0.1
MIN_V = 5
MAX_V = 25

class ControllerEL:
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

        ### Helper Data ###
        self.lane_width = None

    def setup(self,my_car,opp_car,lane_width=2.5):
        # Opp Car for their States
        self.my_car = my_car
        self.opp_car = opp_car

        # action set (follower)
        self.action_set = self.get_all_actions(my_car.actionspace)
        self.num_actions = self.action_set.shape[0]

        # action set of opponent (leader)
        self.opp_action_set = self.get_all_actions(opp_car.actionspace)
        self.num_opp_actions = self.opp_action_set.shape[0]

        ### Helper Data ###
        self.lane_width = lane_width

    def select_action(self):

        print("call in EL controller, (MergeCount, IsMerging)", self.my_car.MergeCount, self.my_car.IsMerging)

        opt_follower_actions = self.opp_car.controller.select_opt_actions()

        opt_actions, opt_traj = self.select_opt_actions(debug=False)

        self.action = opt_actions[0]

        '''
        if self.my_car.MergeCount >= self.my_car.Merge_Steps.shape[0]:
            self.action = 1
            #self.action = np.random.randint(self.my_car.actionspace.shape[0]-1)
        else:
            if self.my_car.IsMerging:
        '''
        if self.my_car.IsMerging and self.my_car.MergeCount < self.my_car.Merge_Steps.shape[0]:
            print("EL overwrite to continue merging")
            self.action = 3

        return self.action

    def select_opt_actions(self,debug = False):
        f_actions_opt, traj_f_opt = self.opp_car.controller.select_opt_actions()
        
        traj_l = self.get_my_traj(self.my_car.s, self.action_set)
        Qf = np.zeros(self.num_actions) 
        for i in range(self.num_actions):
            traj_l_i = traj_l[i]
            Qf[i] = self.compute_returns(traj_l_i,traj_f_opt)

        opt_traj_idx = np.argmax(Qf)
        opt_actions = self.action_set[opt_traj_idx,:]
        opt_traj = traj_l[opt_traj_idx,:,:]

        return opt_actions, opt_traj

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
        R_collision = 0

        # Collision Penalty over entire horizon
        if self.if_collision(s_pred) == 1:
            R_collision = -120

        goal = [100,2.5]

        for k in range(s_pred.shape[0]):
            x_ego = s_pred[k,0]
            y_ego = s_pred[k,1]

            if x_ego <= 25 and y_ego <= 0.5:
                R_distance[k] = -0.4 * abs(x_ego - goal[0]) - 10 * 2.5
            elif x_ego > 25 and x_ego <= 75:
                R_distance[k] = -0.4 * abs(x_ego - goal[0]) - 10 * abs(y_ego - goal[1])
            elif x_ego > 80 and y_ego >= 2:
                R_distance[k] = -0.4 * abs(x_ego - goal[0])

        R = np.minimum(R_distance, R_collision)

        # Discount factor for each step
        discount = np.array([1,0.9,0.9*0.9,0.9*0.9*0.9])
        R = R*discount

        # Cumulative reward over entire horizon
        predict_reward = R.sum()

        return predict_reward


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
                    u = -1*self.opp_car.accel
                elif act == 1:
                    u = 0
                elif act == 2:
                    u = 1*self.opp_car.accel
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
                v = np.clip(v,MIN_V,MAX_V)
                x = x + v*dt
                traj[i,:,k+1] = np.array([x,y,v,yaw])

        return traj



    def get_all_actions(self, actionspace):
        return np.array(list(product(actionspace, repeat=self.horizon)))
       
    def if_collision(self,s_pred):
        x_ego = s_pred[:,0]
        y_ego = s_pred[:,1]
        x_other = s_pred[:,4]
        y_other = s_pred[:,5]

        x_diff = abs(x_ego-x_other)
        y_diff = abs(y_ego-y_other)

        if min(x_diff) >= 12 or min(y_diff) >= 2.4:
            return 0
        else:
            return 1

    ###########################################################################33
    ### helper function ###
