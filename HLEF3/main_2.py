from controlhumanleader import ControlHumanLead
from controlhumanfollower import ControlHumanFoll
#from controlmerge import ControlMerge
from controlmergemulti import ControlMergeMulti
from controllongi import ControlLongi
from car2 import CarVer2
from post_process import *

import numpy as np
from anim import ShowAnim

#NUM_SIMU = 105
VERBOSE = True
MIN_X_DIFF = 5

### Data Setup ###
OUTPUT_DIR = '../output/task11/'
DATA_NAME = 'data1.txt'
LABEL = "0505_Task_FL"

### Setup ###
dt = 0.1
V0 = 24
HORIZON = 4 
LANE_WIDTH = 2.5
NUM_CARS = 4
STOP_X = 350
dynamics = {"v_min":18,"v_max":30 ,"accel":3} # 18, 30, 3
aspace_highway = np.array([0,1,2])
aspace_merge   = np.array([0,1,2,3])

### Helper ###
EXTEND = 10
X0_H1 = np.array([-30])
X0_H2 = np.arange(-20,15)


def simulation(x01, x02):
    ### Init X ###
    x0_h1 = x01
    x0_h2 = x02 
    x0_h3 = 500
    x0_e = 0

    # Random Choose Highway Label
    LF_labels = np.random.randint(2, size = 3)
    LF_labels = np.array([0,1,0])
    #print("L/F Labels: ",LF_labels)

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
    
    s_h3 = np.array([x0_h3,LANE_WIDTH,V0,0])
    hum3 = CarVer2(s = s_h3,
                   actionspace = aspace_highway,
                   ID = "human3",
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
    hum3.set_lane_width(LANE_WIDTH)
    ego.set_lane_width(LANE_WIDTH)

    # Init Human Leader/Follower Controller
    if LF_labels[0] == 0:
        cont_H1 = ControlHumanFoll(hum1,ego,name="follower")
    else:
        cont_H1 = ControlHumanLead(hum1,ego,name="leader")
    hum1.add_controller(cont_H1)
    hum1.add_controller_longi(ControlLongi(hum1))

    if LF_labels[1] == 0:
        cont_H2 = ControlHumanFoll(hum2,ego,name="follower")
    else:
        cont_H2 = ControlHumanLead(hum2,ego,name="leader")
    hum2.add_controller(cont_H2)
    hum2.add_controller_longi(ControlLongi(hum2))

    if LF_labels[2] == 0:
        cont_H3 = ControlHumanFoll(hum3,ego,name="follower")
    else:
        cont_H3 = ControlHumanLead(hum3,ego,name="leader")
    hum3.add_controller(cont_H3)
    hum3.add_controller_longi(ControlLongi(hum3))

    # Init Ego Merge Controller
    cont_MG = ControlMergeMulti(ego,hum1,hum2,hum3,name="merge")
    ego.add_controller(cont_MG)
    ego.add_controller_longi(ControlLongi(ego))
    # Set Leader/Follower Belief Labels # Temp
    ego.controller.set_belief_leader(LF_labels)
    #ego.controller.set_belief_leader(np.array([0.5,1,1]))

    # Link Longi Controller (x_hum1 < x_hum2 < x_hum3)
    hum1.controller_longi.set_car_lead(hum2)
    hum2.controller_longi.set_car_lead(hum3)

###########################################################################

    ### Env ###
    simu_done = False
    merge_success = False
    simu_count = 0
    i = 1
    dt = 0.1
    Tstep = 60
    t = np.linspace(0,Tstep,Tstep+1)
    states = []
    actions = []
    x0 = np.array([x0_e,x0_h1,x0_h2,x0_h3])
    #print("initial x0 (e,h1,h2,h3): ",x0)

    ### Car Dynamics Info ###
    # Uniform Car Dynamics now
    car_dynamics = ego.dynamics

    ### Simulation ###
    while simu_done == False or simu_count <= EXTEND:
        if abs(ego.s[1] - LANE_WIDTH) < 0.01 or ego.s[0] > STOP_X:
            simu_done = True
            if abs(ego.s[1] - LANE_WIDTH) < 0.01:
                merge_success = True
        if simu_done:
            #print("extend count")
            simu_count += 1

        if VERBOSE:
            print("LF roles: ", LF_labels)
            print("\n===== Step: ",i," =====")
        states.append(np.hstack((ego.s,hum1.s,hum2.s,hum3.s)))
        actions.append(np.hstack((ego.car_action,hum1.car_action,hum2.car_action,hum3.car_action)))
        x_log = "x (e,h1,h2,h3) {:3.1f} {:3.1f} {:3.1f} {:3.1f}".format(ego.s[0],hum1.s[0],hum2.s[0],hum3.s[0])
        v_log = "v (e,h1,h2,h3) {:3.1f} {:3.1f} {:3.1f} {:3.1f}".format(ego.s[2],hum1.s[2],hum2.s[2],hum3.s[2])
        y_log = "y (e,h1,h2,h3) {:3.1f} {:3.1f} {:3.1f} {:3.1f}".format(ego.s[1],hum1.s[1],hum2.s[1],hum3.s[1])

        hum1.update(dt)
        hum2.update(dt)
        hum3.update(dt)
        ego.update(dt)
        i += 1

        a_log = "=== action (e,h1,h2,h3) {:3d}{:3d}{:3d}{:3d} ===".format(ego.car_action, hum1.car_action,hum2.car_action, hum3.car_action)

        if VERBOSE:
            print(x_log)
            print(y_log)
            print(v_log)
            print(a_log)

    states = np.array(states)
    actions = np.array(actions).reshape(-1,NUM_CARS)
    #print("Simu Done")
    #print("Merge: ",simu_done)
 
    
    ### Save Data ###
    mydict = {"label":LABEL,
            "num_cars":NUM_CARS,
            "lane_width":LANE_WIDTH,
            "horizon":HORIZON,
            "states":states,
            "actions":actions,
            "lf_roles":LF_labels,
            "dt":dt,
            "x0":x0,
            "car_dynamics":car_dynamics,}


    # Obtain Min X diff after Merge
    min_x_diff_merge = obtain_min_xdiff(states)
    CLOSE_MERGE = False
    if np.any(min_x_diff_merge < MIN_X_DIFF):
        CLOSE_MERGE = True

    save_path = OUTPUT_DIR + LABEL + '_' + str(LF_labels) + '_' + str(x0)+'.npy'
    np.save(save_path, mydict)
    print("save to: ", save_path, '\t', merge_success, CLOSE_MERGE, min_x_diff_merge, ego.controller.belief_leader)

    file_object = open(OUTPUT_DIR + DATA_NAME, 'a')
    file_object.write(save_path+'\t' + str(merge_success) + 
                      ' ' + str(CLOSE_MERGE) + ' ' + str(min_x_diff_merge) + 
                      ' ' + str(ego.controller.belief_leader) + '\n')
    #file_object.close()

    #ShowAnim(save_path)



def save_dict(di):
    filename = LABEL
    with open(filename, 'w') as f:
        f.write(json.dumps(di))


def main():
    for i in range(X0_H1.shape[0]):
        for j in range(X0_H2.shape[0]):
            print(X0_H1[i], X0_H2[j])
            simulation(X0_H1[i], X0_H2[j])
                    
    #for i in range(NUM_SIMU):
    #    print(i)
    #    simulation(-50,X0_H2[i])


########################################################
main()
