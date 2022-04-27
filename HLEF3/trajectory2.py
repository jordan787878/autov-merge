import numpy as np

### Global ###
dt = 1.0 

def get_traj_ego(s0,act_set,horizon,merge_steps,lane_width,dynamics,tstep=dt):

#    print("get traj ego()")

    num_traj = act_set.shape[0]
    accel = dynamics['accel']
    v_min = dynamics['v_min']
    v_max = dynamics['v_max']

    # Init Traj
    traj = np.zeros((num_traj,s0.shape[0],horizon+1))
    traj[:,:,0] = s0

    for i in range(num_traj):
        acts_i = act_set[i,:]

        # Init
        IsMerging = False
        MergeCount = 0

        # Loop over Time Horizon
        for k in range(horizon):
            x0, y0, v0, yaw0 = traj[i,:,k]
            act = acts_i[k]

            # Overwirte Merging Action
            if act != 3:
                if IsMerging and MergeCount < merge_steps:
                    act = 3
                    #print("overwrite: ",k," act:\t",act)

            # Apply Action
            # outputs: u, y
            y = y0
            if act == 0:
                u = -1*accel
            elif act == 1:
                u = 0
            elif act == 2:
                u =  1*accel
            else:
                if MergeCount == 0:
                    u = 0
                    y = y0 + lane_width/merge_steps
                    
                    car_IsMerging = True
                    MergeCount += 1
                
                elif MergeCount < merge_steps:
                    u = 0
                    y = y0 + lane_width/merge_steps
                    
                    MergeCount += 1
                
                else:
                    u = 0
            
            # Integrate to New States
            # inputs: u
            v = v0 + u*dt
            v = np.clip(v,v_min,v_max)

            # 0425 Velocity Constraints saturate Acceleration
            if v0 >= v_max and u > 0:
                u = 0
            if v0 <= v_min and u < 0:
                u = 0

            x = x0 + v0*dt + 0.5*u*dt*dt #0410
            y = np.clip(y,0,lane_width)
            yaw = yaw0
            traj[i,:,k+1] = np.array([x,y,v,yaw])

    return traj



def get_hum_traj(s0,act_set,horizon,dynamics,tstep=dt):
#    print("get hum traj()")

    num_traj = act_set.shape[0]
    accel = dynamics['accel']
    v_min = dynamics['v_min']
    v_max = dynamics['v_max']

    # Init Traj
    traj = np.zeros((num_traj,s0.shape[0],horizon+1))
    traj[:,:,0] = s0

    for i in range(num_traj):
        acts_i = act_set[i,:]
        for k in range(horizon):
            # Get current states
            x0, y0, v0, yaw0 = traj[i,:,k]
            # Get current action
            act = acts_i[k]
            
            # Apply Action
            if act == 0:
                u = -1*accel
            elif act == 1:
                u = 0
            else:
                u =  1*accel

            # Integral: update states
            v = v0 + u*dt
            v = np.clip(v,v_min,v_max)
            #x = x + v*dt
            if v0 >= v_max or v0 <= v_min: #0411
                u = 0
            x = x0 + v0*dt + 0.5*u*dt*dt #0410
            traj[i,:,k+1] = np.array([x,y0,v,yaw0])

    return traj


def get_hum_predict_state(s0,act,dynamics,tstep=dt):
    accel = dynamics['accel']
    v_min = dynamics['v_min']
    v_max = dynamics['v_max']

    x0, y0, v0, yaw0 = s0
    
    # Apply Action
    if act == 0:
        u = -1*accel
    elif act == 1:
        u = 0
    else:
        u =  1*accel

    # Integral: update states
    v = v0 + u*tstep
    v = np.clip(v,v_min,v_max)
    if v0 >= v_max or v0 <= v_min: #0411
        u = 0
    x = x0 + v0*tstep# + 0.5*u*tstep*tstep #0410
    y = y0
    yaw = yaw0

    s = np.array([x,y,v,yaw])

    return s


def test():
    act_ego = np.array([[0,0,0,3],[0,0,0,3]])
    dynamics = {"v_min":18,"v_max":30 ,"accel":3}
    s_ego = np.array([15,0,30,0])
    traj_ego = get_traj_ego(s_ego,act_ego,4,10,2,dynamics)
    
    act_hum = np.array([[1,1,1,1],[1,1,1,1]])
    s_hum = np.array([0,2.5,30,0])
    traj_hum = get_hum_traj(s_hum,act_hum,4,dynamics) 
 
    print(traj_ego[0])
    print(traj_hum[0])

def main():
    test()

if __name__ == "__main__":
    main()

