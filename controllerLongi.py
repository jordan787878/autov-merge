# HignWay Longitudinal Controller

import numpy as np
from itertools import product
import matplotlib.pyplot as plt

class ControllerLongi:
    def __init__(self,name):
        self.name = name
        self.my_car = None
        self.opp_car = None

    def setup(self,my_car,opp_car):
        self.my_car = my_car
        self.opp_car = opp_car

    def overwrite(self):
        my_action = self.my_car.car_action
        
        x_my = self.my_car.s[0]
        x_lead = self.opp_car.s[0]
        #print("longi control",int(x_my),int(x_lead))

        if x_lead - x_my <= 15:
            my_action = 0
            #print(self.my_car.id, "longi overwrite", my_action)

        return my_action
        





