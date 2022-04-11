import numpy as np

### Global ###
dt = 1.0 

def get_traj_ego(s0,act_set,horizon,merge_steps,lane_width,dynamics):

#    print("get traj ego()")

    num_traj = act_set.shape[0]
    accel = dynamics['accel']
    v_min = dynamics['v_min']
    v_max = dynamics['v_max']

    # Init Traj
    traj = np.zeros((num_traj,s0.shape[0],horizon))
    traj[:,:,0] = s0

    for i in range(num_traj):
        acts_i = act_set[i,:]

        # Init
        IsMerging = False
        MergeCount = 0

        # Loop over Time Horizon
        for k in range(horizon-1):
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
            #x = x0 + v0*dt
            # 0410
            x = x0 + v0*dt + 0.5*u*dt*dt
            y = np.clip(y,0,lane_width)
            yaw = yaw0
            traj[i,:,k+1] = np.array([x,y,v,yaw])

    return traj



def get_hum_traj(s0,act_set,horizon,dynamics):
#    print("get hum traj()")

    num_traj = act_set.shape[0]
    accel = dynamics['accel']
    v_min = dynamics['v_min']
    v_max = dynamics['v_max']

    # Init Traj
    traj = np.zeros((num_traj,s0.shape[0],horizon))
    traj[:,:,0] = s0

    for i in range(num_traj):
        acts_i = act_set[i,:]
        for k in range(horizon-1):
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
            # 0410
            x = x0 + v0*dt + 0.5*u*dt*dt
            traj[i,:,k+1] = np.array([x,y0,v,yaw0])

    return traj
