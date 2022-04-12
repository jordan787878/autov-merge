from controlhumanleader import ControlHumanLead
from controlhumanfollower import ControlHumanFoll
from controlmerge import ControlMerge
from controllongi import ControlLongi
from car2 import CarVer2

import numpy as np
from anim import ShowAnim

### Data Setup ###
OUTPUT_DIR = '../output/task3/'
DATA_NAME = 'data1.txt'
LABEL = "0410_Merge_Test"

### Setup ###
dt = 0.1
V0 = 24
HORIZON = 4 
LANE_WIDTH = 2.5
NUM_CARS = 3
GOAL_X = 200
dynamics = {"v_min":18,"v_max":30 ,"accel":3} # 18, 30, 3
aspace_highway = np.array([0,1,2])
aspace_merge   = np.array([0,1,2,3])

### Helper ###
EXTEND = 10

### Init X ###
x0_h2 = np.random.randint(0,30)
#x0_h1 = 0
x0_h1 = x0_h2 + np.random.randint(10,40)
#x0_h2 = 40 
x0_e =  np.random.randint(0,x0_h1+30)
#x0_e =  18 

def main():
    print("import")

    # Init Car
    s_h1 = np.array([x0_h1,LANE_WIDTH,V0,0])
    hum1 = CarVer2(s = s_h1,
                   actionspace = aspace_highway,
                   ID = "human1",
                   horizon = HORIZON,
                   dynamics = dynamics 
                   )

    s_h2 = np.array([x0_h2,LANE_WIDTH,V0,0])
    hum2 = CarVer2(s = s_h2,
                   actionspace = aspace_highway,
                   ID = "human2",
                   horizon = HORIZON,
                   dynamics = dynamics
                   )

    s_e = np.array([x0_e,0,V0,0])
    ego = CarVer2(s = s_e,
                  actionspace = aspace_merge,
                  ID = "ego",
                  horizon = HORIZON,
                  dynamics = dynamics
                 )

    # Set Lane Width
    hum1.set_lane_width(LANE_WIDTH)
    hum2.set_lane_width(LANE_WIDTH)
    ego.set_lane_width(LANE_WIDTH)


    # Init Human Leader/Follower Controller
#    cont_H1 = ControlHumanFoll(hum1,ego,name="human1 follower")
    cont_H1 = ControlHumanLead(hum1,ego,name="leader")
    hum1.add_controller(cont_H1)
    hum1.add_controller_longi(ControlLongi(hum1))

    # Init Human Follower Controller
    cont_H2 = ControlHumanFoll(hum2,ego,name="follower")
    hum2.add_controller(cont_H2)
    hum2.add_controller_longi(ControlLongi(hum2))

    # Init Ego Merge Controller
    cont_MG = ControlMerge(ego,hum1,hum2,name="merge")
    ego.add_controller(cont_MG)
    ego.add_controller_longi(ControlLongi(ego))

    # Setup Highway Leading
    if x0_h2 > x0_h1:
        hum1.controller_longi.set_car_lead(hum2)
    else:
        hum2.controller_longi.set_car_lead(hum1)


###########################################################################

    ### Env ###
    merge_done = False
    simu_count = 0
    i = 1
    dt = 0.1
    Tstep = 60
    t = np.linspace(0,Tstep,Tstep+1)
    states = []
    actions = []
    x0 = np.array([x0_e,x0_h1,x0_h2])
    print("initial x0 (e,h1): ",x0)

    ### Car Dynamics Info ###
    # Uniform Car Dynamics now
    car_dynamics = ego.dynamics

    ### Simulation ###
    while merge_done == False or simu_count <= EXTEND:
        if abs(ego.s[1] - LANE_WIDTH) < 0.01 or ego.s[0] > GOAL_X:
            merge_done = True
        if merge_done:
            print("extend count")
            simu_count += 1

        print("\n===== Step: ",i," =====")
        states.append(np.hstack((ego.s,hum1.s,hum2.s)))
        actions.append(np.hstack((ego.car_action,hum1.car_action,hum2.car_action)))
        x_log = "x (e,h1,h2) {:3.1f} {:3.1f} {:3.1f}".format(ego.s[0],hum1.s[0],hum2.s[0])
        print(x_log)
        v_log = "v (e,h1,h2) {:3.1f} {:3.1f} {:3.1f}".format(ego.s[2],hum1.s[2],hum2.s[2])
        print(v_log)
        y_log = "y (e,h1,h2) {:3.1f} {:3.1f} {:3.1f}".format(ego.s[1],hum1.s[1],hum2.s[1])
        print(y_log)


        hum1.update(dt)
        hum2.update(dt)
        ego.update(dt)
        i += 1

        a_log = "=== action (e,h1,h2) {:3d}{:3d}{:3d} ===".format(ego.car_action, hum1.car_action,hum2.car_action)
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
            "actions":actions,
            "dt":dt,
            "x0":x0,
            "car_dynamics":car_dynamics,}


    save_path = OUTPUT_DIR + LABEL + '_' + str(x0)+'.npy'
    np.save(save_path, mydict)
    print("save to:\t", save_path)

#    file_object = open(OUTPUT_DIR + DATA_NAME, 'a')
#    file_object.write(LABEL + '_' + str(x0)+'.npy\n')
#    file_object.close()

    ShowAnim(save_path)



def save_dict(di):
    filename = LABEL
    with open(filename, 'w') as f:
        f.write(json.dumps(di))



########################################################
main()
