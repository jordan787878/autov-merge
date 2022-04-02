import numpy as np

MERGE_VEL_INCREASE_FACTOR = 0

class Car:
    def __init__(self, s0, actionspace, controller, ID, v_constraint, accel):
        self.s = s0 #s = [x,y,v,yaw]
        self.actionspace = actionspace
        self.controller = controller

        self.controller_longi = None

        self.u = 0
        self.id = ID
        self.MergeCount = 0
        self.IsMerging = False
        self.Merge_Steps = np.linspace(2.5/10,2.5,10)
        self.car_action = None
        
        self.min_v = v_constraint[0]
        self.max_v = v_constraint[1]
        self.accel = accel

    def add_control_longi(self,c):
        self.controller_longi = c
    
    def update(self,dt):
        # Follower/Leader Action Decision considering Merge Car
        self.car_action = self.controller.select_action()

        # Longi. Controll in case the behind car won't hit the leading car
        if self.controller_longi:
            self.car_action = self.controller_longi.overwrite()

        action = self.car_action

        s = self.s

        y = s[1]
        yaw = s[3]

        if action != 3 and self.id == "ego":
            if self.IsMerging and self.MergeCount < self.Merge_Steps.shape[0]:
                action = 3

        # Decl
        if action == 0:
            u = -1*self.accel
        # Cruise
        elif action == 1:
            u = 0
        # Acel
        elif action == 2:
            u = 1*self.accel
        else:
            if self.MergeCount == 0:
                u = 0

                ### [17, -42, 5] issue ###
                u = MERGE_VEL_INCREASE_FACTOR*self.accel

                y += self.Merge_Steps[0]
                y = np.clip(y,0,2.5)
                self.IsMerging = True
                self.MergeCount += 1
            elif self.MergeCount < self.Merge_Steps.shape[0]:
                u = 0

                ### [17, -42, 5] issue ###
                u = MERGE_VEL_INCREASE_FACTOR*self.accel

                y += self.Merge_Steps[0]
                y = np.clip(y,0,2.5)
                self.MergeCount += 1
            else:
                u = 0

        v = s[2] + u*dt
        v = np.clip(v,self.min_v,self.max_v)
        x = s[0] + s[2]*dt

        self.s = np.array([x,y,v,yaw])
        


