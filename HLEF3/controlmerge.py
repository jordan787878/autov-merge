import copy
from trajectory2 import *
from visual2 import *

#############################
MERGE_SAFE_DIST = 10
MERGE_START_X = 25
MERGE_END_X = 175
GOAL_X = 200
#############################


class ControlMerge:
    def __init__(self, car_my, car_h1, car_h2, name):
        print("init control merge")
        self.name = name

        # Link my Car
        self.car_my = car_my

        # Link to Human1 Car, Human2 Car
        self.car_h1_obs = copy.deepcopy(car_h1)
        self.car_h2_obs = copy.deepcopy(car_h2)

        # Store the absolute reference of Human1 Car for State Update
        self.car_h1_abs = car_h1
        self.car_h2_abs = car_h2

        # Check
#        print("merge car link check:\t",self.car_h1_obs.controller.name)
        print("merge car link check:\t",self.car_h1_obs.s)

        #############################################################
        # Obtain horizon
        self.horizon = self.car_my.horizon

        # Init action
        self.action = None

        # Get action set
        #self.act_set_my = self.car_my.get_all_actions()
        self.act_set_my = self.car_my.get_all_ego_actions()
        self.act_set_op = self.car_h1_obs.get_all_actions()

        # Prepare num actions
        self.num_act_my = self.act_set_my.shape[0]
        self.num_act_op = self.act_set_op.shape[0]

        # Check Action Set
        #for i in range(self.num_act_my):
        #    print(self.act_set_my[i])

        # Discount Vector
        self.discount = np.zeros(self.horizon)
        for i in range(self.horizon):
            self.discount[i] = pow(0.9,i)


    def select_action(self):
        print("ego selecting action")

        acts_opt = self.select_opt_actions()

        self.action = acts_opt[0]

        # Continue Mergine Step
        if self.car_my.IsMerging and self.car_my.MergeCount < self.car_my.merge_steps:
#            print("controllerMerge overwrite to continue merging")
            self.action = 3

        return self.action


    def select_opt_actions(self):

#        print("Estimate hum1 car states:\t", self.car_h1_obs.s)
        
        # Get Human1 Opt Action and Traj
        acts_opt_h1, traj_opt_h1 = self.car_h1_obs.controller.select_opt_actions(True,self.car_my.s)
#        print("Estimate hum1 leader opt actions: ", acts_opt_h1)
       
        # Get Human2 Opt Action and Traj
        acts_opt_h2, traj_opt_h2 = self.car_h2_obs.controller.select_opt_actions(True,self.car_my.s)
        #print("Estimate hum1 follower opt actions: ", acts_opt_h1)

        # Get Ego Trajectory
        traj_ego = get_traj_ego(self.car_my.s,
                                self.act_set_my,
                                self.car_my.horizon,
                                self.car_my.merge_steps,
                                self.car_my.lane_width,
                                self.car_my.dynamics)

#        vis_traj(traj_opt_h1, traj_ego)

        # MPC Cost
        MPC_Cost = self.mpc_cost(traj_ego,traj_opt_h1,traj_opt_h2)

#        vis_Qmatrix(-MPC_Cost,self.act_set_my, np.array([1]), self.name + " reward Matrix")

        acts_opt_idx = np.argmin(MPC_Cost)

        acts_opt = self.act_set_my[acts_opt_idx,:]

        # Update human1 car state
        #print('Esitmate hum1 leaderr states: ',self.car_h1_obs.s)
        self.car_h1_obs.s = copy.deepcopy(self.car_h1_abs.s)
        #print('Esitmate hum2 follower states: ',self.car_h2_obs.s)
        self.car_h2_obs.s = copy.deepcopy(self.car_h2_abs.s)


        return acts_opt


    def mpc_cost(self,traj_eg,traj_h1,traj_h2):
        # init
        cost_total = np.zeros(self.num_act_my)
        cost_coli_sum = np.zeros(self.num_act_my)
        cost_coli_hum1 = np.zeros(self.num_act_my)
        cost_coli_hum2 = np.zeros(self.num_act_my)
        cost_merge     = np.zeros(self.num_act_my)

        #######################################################################
        # Loop over Ego Actions
        for i in range(self.num_act_my):
            
            traj_ego_i = traj_eg[i,:,:]

            # Ego Merge Check
            if traj_ego_i[0,-1] >= MERGE_END_X and traj_ego_i[1,-1] < 0.4:
                cost_total[i] = 100
#                print("merge fail at act:\t",i)


            ###################################################################
            # Interation with Human 1 (Follower)
            s_bar = np.hstack((traj_ego_i.T,traj_h1.T))
            
            cost_coli = self.if_collision(s_bar) * 50

            cost_coli_hum1[i] = cost_coli.sum()

            ###################################################################
            # Interation with Human 2 (Follower)
            s_bar = np.hstack((traj_ego_i.T,traj_h2.T))

            cost_coli = self.if_collision(s_bar) * 50 

            cost_coli_hum2[i] = cost_coli.sum()

            ##################################################################
            ### Merge Cost ###
            cost_merge_vector = -1*(traj_ego_i[0,:] - GOAL_X)/(GOAL_X) \
                                -50*(traj_ego_i[1,:] - self.car_my.lane_width)/(self.car_my.lane_width)
            
            cost_merge[i] = cost_merge_vector.sum()
            ''' 
            print("act:",self.act_set_my[i])
            print("ego traj:\n", traj_ego_i[0:2,:])
            print("merge cost vector: ",cost_merge_vector)
            print("merge cost: {:3.2f} total cost: {:3.2f}".format(cost_merge[i],cost_total[i]))
            print("\n")
            '''
        
        #######################################################################
        ### 1. Collision Cost ###
        cost_coli_sum = cost_coli_hum1 + cost_coli_hum2
#        print("human1 collision cost:\t",cost_coli_hum1)
#        print("human2 collision cost:\t",cost_coli_hum2)

        ### Total Cost ###
        cost_total = 1*cost_coli_sum + 1*cost_merge # + 0*(-Qf_ego_hum1) + 0*(-Ql_ego_hum1) + 0*(-Ql_ego_hum2)

        ### Print Costs ###
        ''' 
        for i in range(self.num_act_my):
            print("act:",self.act_set_my[i]
            print("ego traj:\n", traj_eg[i,0:2,:])
            print("merge cost: {:3.2f} total cost: {:3.2f}".format(cost_merge[i],cost_total[i]))
            print("\n")
        '''
        #vis_ego_cost(cost_merge, cost_coli_sum, cost_total, 'ego cost', self.act_set_my) 
        ###################

        return cost_total


    def if_collision(self,s_pred):
        col_flag = np.zeros(self.horizon)

        x_ego = s_pred[:,0]
        y_ego = s_pred[:,1]
        x_other = s_pred[:,4]
        y_other = s_pred[:,5]

        x_diff = (x_ego-x_other)
        y_diff = (y_ego-y_other)

        for k in range(self.horizon):
            if abs(x_diff[k]) >= MERGE_SAFE_DIST or abs(y_diff[k]) > 2.4:
                col_flag[k] = 0
            else:
                col_flag[k] = 1

        return col_flag


