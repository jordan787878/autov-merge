from car import Car
from controllerHF import ControllerHF
from controllerHL import ControllerHL
#from controllerEL import ControllerEL
from controllerEF import ControllerEF
from controllerMerge import ControllerMerge
from controllerLongi import ControllerLongi
import numpy as np
import json

from process import ShowAnim

OUTPUT_DIR = 'output/task2/'
DATA_NAME = 'data2.txt'

LABEL = "0403_HL_EF_Test"
LANE_WIDTH = 2.5
NUM_CARS = 2
GOAL_X = 200

'''
V0 = 15
V_CONSTRAINT = [5,25]
ACCEL = 3
'''
#'''
V0 = 28 
V_CONSTRAINT = [22,33]
ACCEL = 3.5
#'''

def main():
    print("env setup")
    np.set_printoptions(precision=2)

    HORIZON = 3  
   
    aspace_highway = np.array([0,1,2])
    aspace_merge   = np.array([0,1,2,3])
    
    ###################################################################
    # Human 1: HighWay (Follower)
    x0_h1 = np.random.randint(0,60)
    x0_h1 = 12 
    s_h1 = np.array([x0_h1,2.5,V0,0])
    hum1 = Car(s0 = s_h1,
               actionspace = aspace_highway,
                 ID = "human1",
                 horizon = HORIZON,
                 v_constraint = V_CONSTRAINT, 
                 accel = ACCEL
                 )

    ##############################
    # Ego 1: (Follower)
    x0_eg = np.random.randint(0,60)
    x0_eg = 0 
    s = np.array([x0_eg,0,V0,0])
    ego1 = Car(s0 = s,
               actionspace = aspace_merge,
               ID = "ego",
               horizon = HORIZON,
               v_constraint = V_CONSTRAINT, 
               accel = ACCEL
               )

    ##### Setup Controllers #####
    # Human 1 Leader Controller
    control_hl = ControllerHL("hl1",hum1,ego1)
    hum1.add_controller(control_hl)
    # Ego   1 Follow Controller
    control_ef = ControllerEF("ef1",ego1,hum1)
    ego1.add_controller(control_ef)

    ##### Update Lane Width #####
    ego1.set_lane_width(LANE_WIDTH)
    hum1.set_lane_width(LANE_WIDTH)
    
    ### Env ###
    merge_done = False
    i = 1
    dt = 0.1
    Tstep = 60 
    t = np.linspace(0,Tstep,Tstep+1)
    states = []
    actions = []
    x0 = np.array([x0_eg,x0_h1])
    print("initial x0 (e,h1): ",x0)

    ### Car Dynamics Info ###
    # Uniform Car Dynamics now
    car_dynamics = ego1.dynamics

    
    ### Simulation ###
    while merge_done == False:
        if abs(ego1.s[1] - LANE_WIDTH) < 0.01 or ego1.s[0] > GOAL_X:
            merge_done = True

        print("\nstep: ",i)
        states.append(np.hstack((ego1.s,hum1.s)))
        actions.append(np.hstack((ego1.car_action,hum1.car_action)))
        x_log = "x (e,h1) {:1.1f} {:1.1f} ".format(ego1.s[0],hum1.s[0])
        print(x_log)

        ego1.update(dt)
        hum1.update(dt)
        i += 1

        a_log = "a (e,h1) {:3d} {:3d}".format(ego1.car_action, hum1.car_action)
        #a_log = "a (e,h1) {:3d} {:3d}".format(ego1.car_action, -1)
        print(a_log)

    states = np.array(states)
    actions = np.array(actions).reshape(-1,NUM_CARS)
    print("Simu Done")

    ### Save Data ###
    mydict = {"label":LABEL,
            "num_cars":NUM_CARS,
            "lane_width":LANE_WIDTH,
            "horizon":HORIZON,
            "states":states,
            "dt":dt,
            "x0":x0,
            "car_dynamics":car_dynamics,}
    
    
    save_path = OUTPUT_DIR + LABEL + '_' + str(x0)+'.npy'
    np.save(save_path, mydict)
    print("save to:\t", save_path)

    file_object = open(OUTPUT_DIR + DATA_NAME, 'a')
    file_object.write(LABEL + '_' + str(x0)+'.npy\n')
    file_object.close()

    ShowAnim(save_path)
    


def save_dict(di):
    filename = LABEL
    with open(filename, 'w') as f:
        f.write(json.dumps(di))



main()
