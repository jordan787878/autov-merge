import copy
from trajectory2 import *
from visual2 import *
from multi_var_pdf import *

#############################
MERGE_SAFE_DIST = 10
MERGE_START_X = 25
MERGE_CHECK_X = 175
MERGE_END_X = 300 
#############################

W_Merge_Enforce = np.inf
W_Merge_goalX = 1
W_Merge_goalY = 50
W_Collision   = 100

############################

class ControlMergeMulti:
    def __init__(self, car_my, car_h1, car_h2, car_h3, name):
#        print("init control merge")
        self.name = name

        # Link my Car
        self.car_my = car_my

        # Link to Human1 Car, Human2 Car
        self.car_h1_obs = copy.deepcopy(car_h1)
        self.car_h2_obs = copy.deepcopy(car_h2)
        self.car_h3_obs = copy.deepcopy(car_h3)

        # Store the absolute reference of Human Car for State Update
        self.car_h1_abs = car_h1
        self.car_h2_abs = car_h2
        self.car_h3_abs = car_h3

        #############################################################
        # Obtain horizon
        self.horizon = self.car_my.horizon

        # Init action
        self.action = None

        # Get action set
        self.act_set_my = self.car_my.get_all_ego_actions()
        self.act_set_op = self.car_h1_obs.get_all_actions()

        # Prepare num actions
        self.num_act_my = self.act_set_my.shape[0]
        self.num_act_op = self.act_set_op.shape[0]

        # Discount Vector
        self.discount = np.zeros(self.horizon+1)
        for i in range(self.horizon+1):
            self.discount[i] = pow(0.9,i)

        # Helper 
        self.count = 0
        self.max_count = 9

        # L/F Belief
        self.belief_leader = np.array([0.5,0.5,0.5])

        self.act_hum_F = None
        self.act_hum_L = None
        self.x_hum = None
        self.y_hum = None
        self.v_hum = None
        self.yaw_hum = None

        # 0426 Test
        self.merge_safe_dist = MERGE_SAFE_DIST

        # 0503
        self.goal_x = [300,300,300]


    def update_hum_goal_x(self):
        x1 = self.car_h1_obs.s[0]
        x2 = self.car_h2_obs.s[0]
        x3 = self.car_h3_obs.s[0]
        if x1 >= (self.goal_x[0] - 150):
            self.goal_x[0] += 300
        if x2 >= (self.goal_x[1] - 150):
            self.goal_x[1] += 300
        if x3 >= (self.goal_x[2] - 150):
            self.goal_x[2] += 300

        #print(self.goal_x)


    def select_action(self):
        self.update_hum_goal_x()

#        print("### ego select actions ###")
        
        self.select_opt_actions()

        # Continue Mergine Step
        if self.car_my.IsMerging and self.car_my.MergeCount < self.car_my.merge_steps:
            self.action = 3

        # 0428 Task 10
        self.count += 1
        if self.count > self.max_count:
            self.count = 0

        return self.action


    def select_opt_actions(self):
        # Get Ego Trajectory
        traj_ego = get_traj_ego(self.car_my.s,
                                self.act_set_my,
                                self.car_my.horizon,
                                self.car_my.merge_steps,
                                self.car_my.lane_width,
                                self.car_my.dynamics)

        # Get Human1 Opt Action and Traj
        acts_opt_h1_F, traj_opt_h1_F = self.get_human_follower_traj(traj_ego, self.car_my.s, self.car_h1_obs.s, 
                                                                    self.act_set_op, self.horizon, self.car_h1_obs.dynamics,0)
        acts_opt_h1_L, traj_opt_h1_L = self.get_human_leader_traj(traj_ego, self.car_my.s, self.car_h1_obs.s,
                                                                  self.act_set_op, self.horizon, self.car_h1_obs.dynamics,0)
        # Get Human2 Opt Action and Traj
        acts_opt_h2_F, traj_opt_h2_F = self.get_human_follower_traj(traj_ego, self.car_my.s, self.car_h2_obs.s,
                                                                    self.act_set_op, self.horizon, self.car_h2_obs.dynamics,1)
        acts_opt_h2_L, traj_opt_h2_L = self.get_human_leader_traj(traj_ego, self.car_my.s, self.car_h2_obs.s,
                                                                  self.act_set_op, self.horizon, self.car_h2_obs.dynamics,1)
        # Get Human3 Opt Action and Traj
        acts_opt_h3_F, traj_opt_h3_F = self.get_human_follower_traj(traj_ego, self.car_my.s, self.car_h3_obs.s,
                                                                    self.act_set_op, self.horizon, self.car_h3_obs.dynamics,2)
        acts_opt_h3_L, traj_opt_h3_L = self.get_human_leader_traj(traj_ego, self.car_my.s, self.car_h3_obs.s,
                                                                  self.act_set_op, self.horizon, self.car_h3_obs.dynamics,2)
        # Print
#        print("human1 F est action: ",acts_opt_h1_F)
#        print("human2 est opt action: ",acts_opt_h2)
#        print("human3 est opt action: ",acts_opt_h3)
        print("opt LL acts")
        print(acts_opt_h1_L)
        print(acts_opt_h2_L)


#        vis_traj(traj_opt_h1, traj_ego)

        #print(self.count)        
        if self.count % 1 == 0:
        # MPC Cost
            MPC_Cost = self.mpc_cost(traj_ego,traj_opt_h1_F,traj_opt_h1_L,
                                          traj_opt_h2_F,traj_opt_h2_L,
                                          traj_opt_h3_F,traj_opt_h3_L)
#           vis_Qmatrix(-MPC_Cost,self.act_set_my, np.array([1]), self.name + " reward Matrix")
            acts_opt_idx = np.argmin(MPC_Cost)
            acts_opt = self.act_set_my[acts_opt_idx,:]
            self.action = acts_opt[0]
            print("ego opt acts: ", acts_opt)
        
        # 0424
        # Update Leader Belief
        #self.belief_update(self.x_hum, self.y_hum, self.v_hum, self.yaw_hum, 
        #                   self.act_hum_F, self.act_hum_L,
        #                   self.car_h1_obs.s, self.car_h2_obs.s, self.car_h3_obs.s)

        # Store human action and prev state
        self.act_hum_F = []
        self.act_hum_L = []
        self.act_hum_F.append(acts_opt_h1_F[0])
        self.act_hum_L.append(acts_opt_h1_L[0])
        self.act_hum_F.append(acts_opt_h2_F[0])
        self.act_hum_L.append(acts_opt_h2_L[0])
        self.act_hum_F.append(acts_opt_h3_F[0])
        self.act_hum_L.append(acts_opt_h3_L[0])

        self.x_hum = []
        self.y_hum = []
        self.v_hum = []
        self.yaw_hum = []
        self.x_hum.append(self.car_h1_obs.s[0])
        self.y_hum.append(self.car_h1_obs.s[1])
        self.v_hum.append(self.car_h1_obs.s[2])
        self.yaw_hum.append(self.car_h1_obs.s[3])
        self.x_hum.append(self.car_h2_obs.s[0])
        self.y_hum.append(self.car_h2_obs.s[1])
        self.v_hum.append(self.car_h2_obs.s[2])
        self.yaw_hum.append(self.car_h2_obs.s[3])
        self.x_hum.append(self.car_h3_obs.s[0])
        self.y_hum.append(self.car_h3_obs.s[1])
        self.v_hum.append(self.car_h3_obs.s[2])
        self.yaw_hum.append(self.car_h3_obs.s[3])


        # Update human1 car state
        self.car_h1_obs.s = copy.deepcopy(self.car_h1_abs.s)
        self.car_h2_obs.s = copy.deepcopy(self.car_h2_abs.s)
        self.car_h3_obs.s = copy.deepcopy(self.car_h3_abs.s)
        #print('Esitmate hum1 leaderr states: ',self.car_h1_obs.s)
        #print('Esitmate hum2 follower states: ',self.car_h2_obs.s)
        #print("ego act: ", acts_opt)


    def compute_pos_belief(self, car_index, x, y, v, yaw, act_h_F, act_h_L, s0):
        # Get L/F action and Prev. State
        act_F = act_h_F[car_index]
        act_L = act_h_L[car_index]
        s_prev = np.array([x[car_index],v[car_index]])

        # Predict current L/F state
        s_F = get_hum_predict_state(s_prev, act_F, self.car_h1_obs.dynamics, tstep=0.1)
        s_L = get_hum_predict_state(s_prev, act_L, self.car_h1_obs.dynamics, tstep=0.1)
        s = np.array([s0[0],s0[2]])

        #print("current s:", s)
        #print("followe s:", s_F)
        #print("leader  s:", s_L)

        pdf_F, pdf_L = mvnpdf(s, s_F, s_L)

        # Get Prior Belief
        P_F = 1 - self.belief_leader[car_index]
        P_L = self.belief_leader[car_index]

        den = pdf_F*P_F + pdf_L*P_L
        belief_L = pdf_L*P_L/den
        belief_F = pdf_F*P_F/den
        self.belief_leader[car_index] = belief_L

        np.set_printoptions(precision=3)
        #print("prior proba: ",P_F, P_L)
        #print("hum belief (F/L): ",belief_F, belief_L)

    
    def belief_update(self, x, y, v, yaw, act_h_F, act_h_L, s_h1, s_h2, s_h3):
        if act_h_F:
            #print("start belief update")
            self.compute_pos_belief(0, x, y, v, yaw, act_h_F, act_h_L, s_h1)
            self.compute_pos_belief(1, x, y, v, yaw, act_h_F, act_h_L, s_h2)
            self.compute_pos_belief(2, x, y, v, yaw, act_h_F, act_h_L, s_h3)
            #print("belief leader: ",self.belief_leader)


    def mpc_cost(self,traj_eg,traj_h1_F,traj_h1_L,traj_h2_F,traj_h2_L,traj_h3_F,traj_h3_L):
        #print("mpc_cost()")
        #print("belief leader: ",self.belief_leader)
        
        # init
        cost_total = np.zeros(self.num_act_my)
        #cost_enforce = np.zeros(self.num_act_my)
        #cost_coli_sum = np.zeros(self.num_act_my)
        cost_merge     = np.zeros(self.num_act_my)

        #######################################################################
        tol = 0.1

        # Loop over Ego Actions
        for i in range(self.num_act_my):
            
            traj_ego_i = traj_eg[i,:,:]

            # Merge Fail Cost
            if traj_ego_i[0,-1] >= MERGE_CHECK_X and traj_ego_i[1,-1] == 0:
                cost_total[i] = np.inf

            # Collision Cost
            # Interation with Human 1
                # cost_coli_hum1[i,0], cost_coli_hum1[i,1] = self.MPC_cost_collision(traj_ego_i, traj_h1_F, traj_h1_L)
            colli_h1_F = np.any(self.if_collision(traj_ego_i,traj_h1_F,self.merge_safe_dist) == 1)
            colli_h1_L = np.any(self.if_collision(traj_ego_i,traj_h1_L,self.merge_safe_dist) == 1)
            if colli_h1_F and 1 - self.belief_leader[0] > tol:
                cost_total[i] = np.inf
            if colli_h1_L and self.belief_leader[0] > tol:
                cost_total[i] = np.inf

            # Interaction with Human 2 
                # cost_coli_hum2[i,0], cost_coli_hum2[i,1] = self.MPC_cost_collision(traj_ego_i, traj_h2_F, traj_h2_L)
            colli_h2_F = np.any(self.if_collision(traj_ego_i,traj_h2_F,self.merge_safe_dist) == 1)
            colli_h2_L = np.any(self.if_collision(traj_ego_i,traj_h2_L,self.merge_safe_dist) == 1)
            if colli_h2_F and 1 - self.belief_leader[1] > tol:
                cost_total[i] = np.inf
            if colli_h2_L and self.belief_leader[1] > tol:
                cost_total[i] = np.inf

            # Interaction with Human 3
                # cost_coli_hum3[i,0], cost_coli_hum3[i,1] = self.MPC_cost_collision(traj_ego_i, traj_h3_F, traj_h3_L)
            colli_h3_F = np.any(self.if_collision(traj_ego_i,traj_h3_F,self.merge_safe_dist) == 1)
            colli_h3_L = np.any(self.if_collision(traj_ego_i,traj_h3_L,self.merge_safe_dist) == 1)
            if colli_h3_F and 1 - self.belief_leader[2] > tol:
                cost_total[i] = np.inf
            if colli_h3_L and self.belief_leader[2] > tol:
                cost_total[i] = np.inf

            # Merge Cost 
            cost_merge_vector = -W_Merge_goalX*(traj_ego_i[0,:] - MERGE_END_X)/(MERGE_END_X) \
                                -W_Merge_goalY*(traj_ego_i[1,:] - self.car_my.lane_width)/(self.car_my.lane_width) 
            cost_merge[i] = cost_merge_vector.sum()
       
        # End For Loop
       
        ### 1. Collision Cost ###
        #cost_coli_sum = (1-self.belief_leader[0])*cost_coli_hum1[:,0] + (self.belief_leader[0])*cost_coli_hum1[:,1] + \
        #                (1-self.belief_leader[1])*cost_coli_hum2[:,0] + (self.belief_leader[1])*cost_coli_hum2[:,1] + \
        #                (1-self.belief_leader[2])*cost_coli_hum3[:,0] + (self.belief_leader[2])*cost_coli_hum3[:,1]

        ### Total Cost ###
        #cost_total = 1*cost_enforce + 1*cost_coli_sum + 1*cost_merge # + 0*(-Qf_ego_hum1) + 0*(-Ql_ego_hum1) + 0*(-Ql_ego_hum2)
        cost_total =  cost_total + cost_merge

        # debug
        #vis_ego_cost(np.zeros(self.num_act_my), cost_merge, np.zeros(self.num_act_my), cost_total, 'ego cost', self.act_set_my) 
        #for i in range(self.num_act_my):
        #    if cost_total[i] != np.inf:
        #        print(self.act_set_my[i],'\t',cost_total[i])

        return cost_total

    # MPC Helper Fcn
    def MPC_cost_collision(self, traj_ego, traj_hum_F, traj_hum_L):
        cost_coli_F = self.if_collision(traj_ego,traj_hum_F,self.merge_safe_dist) * W_Collision
        cost_coli_L = self.if_collision_HL(traj_ego,traj_hum_L,self.merge_safe_dist) * W_Collision
        return cost_coli_F.sum(), cost_coli_L.sum()


##### Leader Follower Game #####

    def get_human_follower_traj(self,traj_ego,s_ego,s_hum,acts_hum,horizon_hum,dynamics_hum,car_index):
        # Obtain goal
        goal = self.goal_x[car_index]

        # Check Traffic States
        #print(" ---- Estimate ---- ")
        #print("s_ego, s_hum: ", s_ego, s_hum)

        # Init Cost Matrix
        num_act_hum = acts_hum.shape[0]
        num_act_ego = self.num_act_my
        Qf = np.zeros((num_act_hum, num_act_ego))
        Qfi = np.zeros(num_act_hum)
        #print(num_act_hum)
        #print(num_act_ego)

        # Expand hum trajectory
        traj_hum = get_hum_traj(s_hum, acts_hum, horizon_hum, dynamics_hum)

        # Loop over possible (traj_hum, traj_ego)
        for i in range(num_act_hum):
            traj_hum0 = traj_hum[i,:]
            for j in range(num_act_ego):
                traj_ego0 = traj_ego[j,:]
                Qf[i,j] = self.compute_Q_hum(traj_ego0,traj_hum0,goal)
            Qfi[i] = np.min(Qf[i,:])

        # Get optimize traj
        traj_opt_idx = np.argmax(Qfi)
        traj_opt = traj_hum[traj_opt_idx,:,:]
        acts_opt = acts_hum[traj_opt_idx,:]

        return acts_opt, traj_opt

    '''
    def compute_Qf_hum(self, traj_ego, traj_hum):
        # Init
        R_distance = np.zeros(traj_ego.shape[1])
        R_collision = -100
        R = 0

        ###### Collision Reward ##### (entire horizon)
        R_collision = self.if_collision(traj_ego,traj_hum)*R_collision
        R_colli = R_collision.sum()

        ##### Distance Reward #####
        # Normalized Distance Reward for each step, (min: -1, max: 0)
        R_distance = -abs(traj_hum[0,:] - goal)/(goal)

        # Reward
        R = np.minimum(R_distance, R_collision)

        # Discount factor for each step
        R = R*self.discount

        # Cumulative reward over entire horizon
        R = R.sum()

        return R
    '''


    def get_human_leader_traj(self,traj_ego,s_ego,s_hum,acts_hum,horizon_hum,dynamics_hum,car_index):

        # Obtain goal
        goal = self.goal_x[car_index]

        # Init Cost Matrix
        num_act_hum = acts_hum.shape[0]
        num_act_ego = self.num_act_my

        # Expand hum trajectory
        traj_hum = get_hum_traj(s_hum, acts_hum, horizon_hum, dynamics_hum)

        # Get ego follower optimal trajectory
        Qfi = self.compute_Qf_ego(traj_ego, traj_hum)
        traj_ego_opt_idx = np.argmax(Qfi)
        traj_ego_opt = traj_ego[traj_ego_opt_idx,:,:]
        acts_ego_opt = self.act_set_my[traj_ego_opt_idx,:]

        # Overwrite Ego Actions After Merge
        y_ego = self.car_my.s[1]
        if y_ego > 0 and y_ego < 2.5:
            traj_ego_opt = traj_ego[-1,:,:]

        # Compute hum leader Cost Matrix
        Ql = np.zeros(num_act_hum)
        for i in range(num_act_hum):
            Ql[i] = self.compute_Q_hum(traj_ego_opt, traj_hum[i], goal)

        # Get optimal hum leader trajectory
        traj_opt_idx = np.argmax(Ql)
        traj_opt = traj_hum[traj_opt_idx,:,:]
        acts_opt = acts_hum[traj_opt_idx,:]
        
        return acts_opt, traj_opt


    def compute_Qf_ego(self, traj_ego, traj_hum):
        Qf = np.zeros((traj_ego.shape[0],traj_hum.shape[0]))
        Qfi = np.zeros(traj_ego.shape[0])
        
        for i in range(traj_ego.shape[0]):
            for j in range(traj_hum.shape[0]):
                Qf[i,j] = self.compute_ego_reward(traj_ego[i],traj_hum[j])
            Qfi[i] = np.min(Qf[i,:])

        return Qfi


    def compute_ego_reward(self, traj_ego, traj_hum):
        # Init
        R_distance = np.zeros(traj_ego.shape[1])
        R_collision = -100
        R = 0

        ###### Collision Reward ##### (entire horizon)
        R_collision = self.if_collision(traj_ego,traj_hum)*R_collision

        ##### Distance Reward #####
        for k in range(traj_ego.shape[1]):
            x_ego = traj_ego[0,k]
            y_ego = traj_ego[1,k]
            if x_ego >= MERGE_START_X and x_ego <= MERGE_END_X and y_ego < 0.1:
                R_distance[k] = -abs(x_ego-MERGE_END_X)/MERGE_END_X -10*abs(y_ego-2.5)/2.5
            else:
                R_distance[k] = -abs(x_ego-MERGE_END_X)/MERGE_END_X
            if x_ego >= MERGE_CHECK_X and y_ego < 0.1:
                R_distance[k] = -100
            
            '''
            if x_ego >= 25 and x_ego <= 175 and y_ego < 0.5:
                R_distance[k] = -abs(x_ego-goal)/goal -10*abs(y_ego-2.5)/2.5
            else:
                R_distance[k] = -abs(x_ego-goal)/goal
            if x_ego >= 100 and y_ego < 0.5:
                R_distance[k] = -100
            '''
        
        ##### Total Reward #####
        R = np.minimum(R_distance, R_collision)
        R = R*self.discount
        R = R.sum()

        return R


    def compute_Q_hum(self, traj_ego, traj_hum, goal):
        # Init
        R_distance = np.zeros(traj_ego.shape[1])
        R_collision = -100
        R = 0

        ###### Collision Reward ##### (entire horizon)
        R_collision = self.if_collision(traj_ego,traj_hum)*R_collision
        R_colli = R_collision.sum()

        ##### Distance Reward #####
        # Normalized Distance Reward for each step, (min: -1, max: 0)
        R_distance = -abs(traj_hum[0,:] - goal)/(goal)

        # Reward
        R = np.minimum(R_distance, R_collision)

        # Discount factor for each step
        R = R*self.discount

        # Cumulative reward over entire horizon
        R = R.sum()

        return R

    '''
    def compute_Ql_hum(self, traj_ego, traj_hum):
        # Init
        R_distance = np.zeros(traj_ego.shape[1])
        R_collision = -100
        R = 0

        ###### Collision Reward ##### (entire horizon)
        R_collision = self.if_collision(traj_ego,traj_hum)*R_collision
        R_colli = R_collision.sum()

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

        return R
    '''


##### Leader Follower Game #####

    def if_collision(self,traj_ego,traj_hum, merge_safe_dist=MERGE_SAFE_DIST):
        col_flag = np.zeros(self.horizon+1)

        x_ego = traj_ego[0,:]
        y_ego = traj_ego[1,:]
        x_other = traj_hum[0,:]
        y_other = traj_hum[1,:]

        x_diff = (x_ego-x_other)
        y_diff = (y_ego-y_other)

        for k in range(1,self.horizon+1):
            if abs(x_diff[k]) <= merge_safe_dist and abs(y_diff[k]) < self.car_my.lane_width:
                col_flag[k] = 1
            else:
                col_flag[k] = 0

        # Collision Flog over Entire Horizon
        if col_flag.any() == 1:
            col_flag = np.ones(self.horizon+1)

        return col_flag


    # Belief Helper Fcn
    def set_belief_leader(self, LF_labels):
        self.belief_leader = LF_labels
    
