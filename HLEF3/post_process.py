import numpy as np

def obtain_min_xdiff(states):
    num_hum_cars = int(states.shape[1]/4)-1
    x_ego = states[:,0]
    y_ego = states[:,1]
    time = np.arange(states.shape[0])

    if np.any(y_ego > 0):
        # Get Ego Merge Time
        time_merge = np.argwhere(y_ego >= 0.5)[0]
        time_merge = time_merge[0]

        # Init Min X Diff. after merge
        min_x_diff_merge = np.zeros(num_hum_cars)

        for i in range(num_hum_cars):
            x_hum = states[:,4*(i+1)]
            x_diff = abs(x_ego-x_hum)
            min_x_diff_merge[i] = np.min(x_diff[time_merge:])

        return min_x_diff_merge

    else:
        return 0

