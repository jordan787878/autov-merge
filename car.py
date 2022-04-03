import numpy as np
from itertools import product


class Car:
    def __init__(self, s0, actionspace, ID, horizon, v_constraint, accel):
        self.s = s0 #s = [x,y,v,yaw]
        self.actionspace = actionspace
        self.id = ID
        self.horizon = horizon

        # Controller
        self.controller = None
#        self.controller = controller
#        self.controller_longi = None

        self.u = 0
        self.MergeCount = 0
        self.IsMerging = False
        self.merge_steps = 10
        self.car_action = None
        
        self.dynamics = {"v_min":v_constraint[0],
                         "v_max":v_constraint[1],
                         "accel":accel,
                         }

        # Helper
        self.lane_width = None

    def add_controller(self,c):
        self.controller = c

    def set_lane_width(self,l):
        self.lane_width = l

    '''
    def add_control_longi(self,c):
        self.controller_longi = c
    '''

    def get_all_actions(self):
        perm_actions =  np.array(list(product(self.actionspace, repeat=self.horizon)))
        #print(perm_actions)
        return perm_actions

    def update(self,dt):
        self.car_action = self.controller.select_action()

        # Longi. Controll in case the behind car won't hit the leading car
#        if self.controller_longi:
#            self.car_action = self.controller_longi.overwrite()

        action = self.car_action

        x0, y0, v0, yaw0 = self.s
        #s = self.s
        #y = s[1]
        #yaw = s[3]

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

            elif self.MergeCount < self.merge_steps:
                u = 0
                y += self.lane_width/self.merge_steps
                self.MergeCount += 1
            else:
                u = 0

        v = v0 + u*dt
        v = np.clip(v,v_min,v_max)
        x = x0 + v0*dt
        y = np.clip(y,0,self.lane_width)
        yaw = yaw0

        self.s = np.array([x,y,v,yaw])
        


