import numpy as np
import matplotlib.pyplot as plt

def vis_Q(Q,title,xticks):
    num_Q = Q.shape[0]
    x = np.arange(0,num_Q) + 1
    x_max = np.argmax(Q)
    Q_max = np.max(Q)
    text_max = str(xticks[x_max])

    fig, ax = plt.subplots()
    ax.plot(x,Q)
    ax.scatter(x_max,Q_max)
    ax.annotate(text_max, (x_max, Q_max) )
    
    plt.title(title)
    labels = np.array([str(xx) for xx in xticks])
    plt.xticks(x, labels, rotation = 'vertical')
    plt.margins(0.1)
    # Adjust more bottom space
    plt.subplots_adjust(bottom = 0.15)
    # Show and close figure auto. 1 sec
    plt.show(block=False)
    plt.pause(0.7)
    plt.close()

def vis_HF_reward(s):
    return 0

def vis_Qmatrix(Q,acts_i, acts_j, name):
    num_my_act = acts_i.shape[0]
    num_opp_act = acts_j.shape[0]

    Q = np.array(Q).reshape(num_my_act,num_opp_act)
    
    fig, ax = plt.subplots()
    ax.imshow(Q, cmap = "hot", aspect = "auto")
    
    y_ticks = np.arange(num_my_act)
    ax.set_yticks(y_ticks)
    labels = np.array([str(y) for y in acts_i])
    ax.set_yticklabels(labels)

    x_ticks = np.arange(num_opp_act)
    ax.set_xticks(x_ticks)
    
    #labels = np.array([str(y) for y in acts_i])
    #ax.set_yticklabels(labels)
    plt.xlabel('actions j')
    plt.ylabel('action i')
    ax.grid(which='both')
    plt.title(name)
    plt.show()

def vis_collision_reward(x_diff,y_diff,act_hf,act_el,flag):
    horizon = x_diff.shape[0]
    t = np.arange(horizon)
    print(act_hf,"\t",act_el)
    if flag == 1:
        print("will collide in horizon")
    else:
        print("Safe")
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.plot(t,x_diff,color='blue')
    ax2.plot(t,y_diff,color='blue')
    ax1.plot([0,3],[12,12],color='red',linestyle='dashed')
    ax2.plot([0,3],[-2.4,-2.4],color='red',linestyle='dashed')
    '''
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    '''
    if np.array_equal(act_hf, np.array([2,2,2])):
        plt.show()

def main():
    Q = np.array([1,2,3])
    vis_Q(Q)

#main()
