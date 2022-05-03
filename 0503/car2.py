import numpy as np
from itertools import product


class CarVer2:
    def __init__(self, s, actionspace, ID, horizon, dynamics):
        self.s = s #s = [x,y,v,yaw]
        self.actionspace = actionspace
        self.id = ID
        self.horizon = horizon

        # Controller
        self.controller = None

        self.u = 0
        self.MergeCount = 0
        self.IsMerging = False
        self.merge_steps = 10
        self.car_action = 1
        self.dynamics = dynamics

        # Helper
        self.lane_width = None
        self.count = 0

        # Longi Controller in Same lane
        self.controller_longi = None

    def add_controller(self,c):
        self.controller = c

    def add_controller_longi(self,c):
        self.controller_longi = c

    def set_lane_width(self,l):
        self.lane_width = l

    def get_all_actions(self):
        perm_actions =  np.array(list(product(self.actionspace, repeat=self.horizon)))
        #print(perm_actions)
        return perm_actions

    def get_all_ego_actions(self):
        perm_actions =  np.array(list(product(self.actionspace, repeat=self.horizon)))

        num_acts = perm_actions.shape[0]
        for i in range(num_acts):
            for j in range(self.horizon-1):
                if perm_actions[i,j] == 3:
                    perm_actions[i,j+1:self.horizon] = 3
                    break

#        for i in range(num_acts):
#            print(i,perm_actions[i])
                

        #print(perm_actions)
        return perm_actions


    def update(self,dt):
        # Merging Controller
        if self.count % 1 == 0:
            self.car_action = self.controller.select_action()
            self.car_action = self.controller_longi.select_action(self.car_action)
        self.count += 1

        # Longi Controller in same lane
        # self.car_action = self.controller_longi.select_action(self.car_action)

        action = self.car_action

        x0, y0, v0, yaw0 = self.s

        # Get Dynamics
        v_min = self.dynamics['v_min']
        v_max = self.dynamics['v_max']
        accel = self.dynamics['accel']

        if action != 3 and self.id == "ego":
            if self.IsMerging and self.MergeCount < self.merge_steps:
                action = 3
        
        y = y0
        # Decl
        if action == 0:
            u = -1*accel
        # Cruise
        elif action == 1:
            u = 0
        # Acel
        elif action == 2:
            u = 1*accel
        else:
            if self.MergeCount == 0:
                u = 0
                y += self.lane_width/self.merge_steps
                self.IsMerging = True
                self.MergeCount += 1
#                print("***** MERGE *****")

            elif self.MergeCount < self.merge_steps:
                u = 0
                y += self.lane_width/self.merge_steps
                self.MergeCount += 1
#                print("***** MERGE *****")
            else:
                u = 0

        v = v0 + u*dt
        v = np.clip(v,v_min,v_max)
        x = x0 + v0*dt + 0.5*u*dt*dt
        y = np.clip(y,0,self.lane_width)
        yaw = yaw0

        self.s = np.array([x,y,v,yaw])
        


