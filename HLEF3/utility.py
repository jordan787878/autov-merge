import numpy as np

def check_collision(traj_ego, traj_hum, horizon, x_margin, y_margin):
    x_ego = traj_ego[0,:]
    y_ego = traj_ego[1,:]
    x_other = traj_hum[0,:]
    y_other = traj_hum[1,:]

    x_diff = (x_ego-x_other)
    y_diff = (y_ego-y_other)

    x_diff = x_diff[:horizon]
    y_diff = y_diff[:horizon]
    #print(x_diff)
    #print(y_diff)

    index_close = np.where(abs(y_diff) < y_margin)

    if np.any(index_close):
        if np.min(abs(x_diff[index_close])) < x_margin:
            return True

    return False


def check_collision_HL(traj_ego, traj_hum, horizon, x_margin, y_margin):
    x_ego = traj_ego[0,:]
    y_ego = traj_ego[1,:]
    x_other = traj_hum[0,:]
    y_other = traj_hum[1,:]

    x_diff = (x_ego-x_other)
    y_diff = (y_ego-y_other)

    x_diff = x_diff[:horizon]
    y_diff = y_diff[:horizon]
    #print(x_diff)
    #print(y_diff)

    index_close = np.where(abs(y_diff) < y_margin)

    if np.any(index_close):
        if np.min(abs(x_diff[index_close])) < x_margin and x_diff[0] > 0:
            return True

    return False


def Qf_hum(traj_ego, traj_hum, goal):
    Qf = np.zeros((traj_hum.shape[0],traj_ego.shape[0]))
    Qfi = np.zeros(traj_hum.shape[0])
    for i in range(traj_hum.shape[0]):
        for j in range(traj_ego.shape[0]):
            Qf[i,j] = hum_follow_reward(traj_ego[j], traj_hum[i], goal)
        Qfi[i] = np.min(Qf[i,:])
    return Qfi

def hum_follow_reward(traj_ego, traj_hum, goal):
    # Init
    horizon = traj_ego.shape[1]
    discount = np.power(0.9, np.arange(horizon))
    R_distance = np.zeros(traj_ego.shape[1])
    R = 0

    ###### Collision Reward ##### (entire horizon)
    if check_collision(traj_ego, traj_hum, horizon, x_margin = 10, y_margin = 2.5):
        R_collision = -100
    else:
        R_collision = 0

    ##### Distance Reward #####
    # Normalized Distance Reward for each step, (min: -1, max: 0)
    R_distance = -abs(traj_hum[0,:] - goal)/(goal)
    R = np.minimum(R_distance, R_collision)
    R = R*discount
    R = R.sum()

    '''
    for k in range(horizon):
        r = np.minimum(R_distance[k], R_collision)*pow(0.9,k)
        R += r
    '''

    return R


def Qf_ego(traj_ego, traj_hum):
    Qf = np.zeros((traj_ego.shape[0],traj_hum.shape[0]))
    Qfi = np.zeros(traj_ego.shape[0])
    for i in range(traj_ego.shape[0]):
        for j in range(traj_hum.shape[0]):
            Qf[i,j] = ego_follow_reward(traj_ego[i],traj_hum[j])
        Qfi[i] = np.min(Qf[i,:])
    return Qfi

def ego_follow_reward(traj_ego, traj_hum):
    # Temp
    MERGE_START_X = 25
    MERGE_CHECK_X = 175
    MERGE_END_X = 300

    # Init
    horizon = traj_ego.shape[1]
    R_distance = np.zeros(horizon)
    R = 0

    ###### Collision Reward ##### (entire horizon)
    if check_collision(traj_ego, traj_hum, horizon, x_margin = 10, y_margin = 2.5):
        R_collision = -100
    else:
        R_collision = 0

    ##### Distance Reward #####
    # Normalized Distance Reward for each step, (min: -1, max: 0)
    for k in range(horizon):
        x_ego = traj_ego[0,k]
        y_ego = traj_ego[1,k]
        if x_ego >= MERGE_START_X and x_ego <= MERGE_END_X and y_ego < 0.1:
            R_distance[k] = 0*-abs(x_ego-MERGE_END_X)/MERGE_END_X -10*abs(y_ego-2.5)/2.5
        else:
            R_distance[k] = -abs(x_ego-MERGE_END_X)/MERGE_END_X
        if x_ego >= MERGE_CHECK_X and y_ego < 0.1:
            R_distance[k] = -100
        r = np.minimum(R_distance[k], R_collision)*pow(0.9,k)
        R += r

    return R


def Ql_hum(traj_ego, traj_hum, goal):
    Ql = np.zeros(traj_hum.shape[0])
    for i in range(traj_hum.shape[0]):
        Ql[i] = hum_lead_reward(traj_ego, traj_hum[i], goal)
    return Ql


def hum_lead_reward(traj_ego,traj_hum, goal):
    # Init
    horizon = traj_ego.shape[1]
    discount = np.power(0.9, np.arange(horizon))
    R_distance = np.zeros(traj_ego.shape[1])
    R = 0

    ###### Collision Reward ##### (entire horizon)
    if check_collision_HL(traj_ego, traj_hum, horizon, x_margin = 10, y_margin = 2.5):
        R_collision = -100
    else:
        R_collision = 0

    ##### Distance Reward #####
    # Normalized Distance Reward for each step, (min: -1, max: 0)
    R_distance = -abs(traj_hum[0,:] - goal)/(goal)
    R = np.minimum(R_distance, R_collision)
    R = R*discount
    R = R.sum()
    '''
    for k in range(horizon):
        r = np.minimum(R_distance[k], R_collision)*pow(0.9,k)
        R += r
    '''
    return R


def store_prev_states(acts_opt_h1_F,acts_opt_h1_L,acts_opt_h2_F,acts_opt_h2_L,acts_opt_h3_F,acts_opt_h3_L,h1_s,h2_s,h3_s):
    act_hum_F = []
    act_hum_L = []
    act_hum_F.append(acts_opt_h1_F[0])
    act_hum_L.append(acts_opt_h1_L[0])
    act_hum_F.append(acts_opt_h2_F[0])
    act_hum_L.append(acts_opt_h2_L[0])
    act_hum_F.append(acts_opt_h3_F[0])
    act_hum_L.append(acts_opt_h3_L[0])

    x_hum = []
    y_hum = []
    v_hum = []
    yaw_hum = []
    x_hum.append(h1_s[0])
    y_hum.append(h1_s[1])
    v_hum.append(h1_s[2])
    yaw_hum.append(h1_s[3])
    x_hum.append(h2_s[0])
    y_hum.append(h2_s[1])
    v_hum.append(h2_s[2])
    yaw_hum.append(h2_s[3])
    x_hum.append(h3_s[0])
    y_hum.append(h3_s[1])
    v_hum.append(h3_s[2])
    yaw_hum.append(h3_s[3])

    return act_hum_F, act_hum_L, x_hum, y_hum, v_hum, yaw_hum




def main():
    traj_ego = np.array([[0,22.5,42,60,78],
                         [0,0,0,0,0]])
    #traj_ego = np.array([[0,24,48,72,96],
    #                     [0,0.25,0.5,0.75,1]])

    traj_hum = np.array([[0,25.5,54,84,114],
                         [2.5,2.5,2.5,2.5,2.5]])
    print(check_collision(traj_ego, traj_hum, 5, x_margin=10, y_margin=2.5))

if __name__ == "__main__":
    main()


