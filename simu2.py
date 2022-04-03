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

LABEL = "0402_HL_EF_Test"
LANE_WIDTH = 2.5
NUM_CARS = 2

V0 = 15
V_CONSTRAINT = [5,25]
ACCEL = 3

def main():
    print("env setup")
    np.set_printoptions(precision=2)

    horizon = 3 # if change this, make sure to change the discount array in HF controller in compute_reward() 
   
    aspace_highway = np.array([0,1,2])
    aspace_merge   = np.array([0,1,2,3])
    
    ###################################################################
    # NOTE: CARS
    # Human 1: HighWay (Follower)
    x0_h1 = np.random.randint(0,60)
    #x0_h1 = 0 

    s_h1 = np.array([x0_h1,2.5,V0,0])
    # Init Controllers
    cont_hf1 = ControllerHL("hl",horizon)
    cont_longi1 = ControllerLongi("longi")
    # Init Car Human1
    human1 = Car(s_h1,aspace_highway,cont_hf1,"human1",V_CONSTRAINT, ACCEL)
    ##############################
    # Ego 1: (Follower)
    x0_eg = np.random.randint(0,60)
    #x0_eg = x0_h1 + 18

    s = np.array([x0_eg,0,V0,0])
    cont = ControllerEF("ego_follower",horizon)
    ego1 = Car(s,aspace_merge,cont,"ego",V_CONSTRAINT, ACCEL)

    ############################################################################
    # NOTE: Setup Controller Dependency
    human1.controller.setup(human1, ego1,   lane_width=LANE_WIDTH)
    ego1.controller.setup  (ego1,   human1, lane_width=LANE_WIDTH)

    human1.controller.setup_est_opp_control()
    ego1.controller.setup_est_opp_control()
    ############################################################################

    ### Env ###
    merge_done = False
    i = 1
    dt = 0.1
    Tstep = 60 
    t = np.linspace(0,Tstep,Tstep+1)
    states = []
    actions = []
    #x0 = np.array([x0_eg,x0_h1,x0_h2,x0_h3])
    x0 = np.array([x0_eg,x0_h1])
    print("initial x0 (e,h1): ",x0)

    
    ### Simulation ###
    while merge_done == False:
        if abs(ego1.s[1] - 2.5) < 0.01 or ego1.s[0] > 150:
            merge_done = True

        print("\nstep: ",i)
        states.append(np.hstack((ego1.s,human1.s)))
        actions.append(np.hstack((ego1.car_action,human1.car_action)))
        x_log = "x (e,h1) {:1.1f} {:1.1f} ".format(ego1.s[0],human1.s[0])
        print(x_log)

        ego1.update(dt)
        human1.update(dt)
        i += 1

        a_log = "a (e,h1) {:3d} {:3d}".format(ego1.car_action, human1.car_action)
        print(a_log)
    states = np.array(states)
    actions = np.array(actions).reshape(-1,NUM_CARS)
    print("Simu Done")

    ### Save Data ###
    mydict = {"label":LABEL,
            "num_cars":NUM_CARS,
            "lane_width":LANE_WIDTH,
            "horizon":horizon,
            "states":states,
            "dt":dt,
            "x0":x0}
    
    save_path = 'output/'+LABEL + '_' + str(x0)+'.npy'
    np.save(save_path, mydict)
    print("save to:\t", save_path)

    ShowAnim(save_path)


def save_dict(di):
    filename = LABEL
    with open(filename, 'w') as f:
        f.write(json.dumps(di))



main()
