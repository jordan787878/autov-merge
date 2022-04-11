import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm

def vis_traj(tra1,tra2):
    fig, ax = plt.subplots()
    if tra1.shape[0] == 4:
        tra1 = tra1.reshape(1,4,4)

    x0, y0 = tra1[0,0:2,0]
    horizon = tra1.shape[2]
    ax.scatter(x0,y0,color='b')
    for i in range(tra1.shape[0]):
        trai = tra1[i,0:2,:]
        ax.plot(trai[0,:],trai[1,:],color='b')
        ax.scatter(trai[0,horizon-1],trai[1,horizon-1])

    x0, y0 = tra2[0,0:2,0]
    horizon = tra2.shape[2]
    ax.scatter(x0,y0,color='r')
    for i in range(tra2.shape[0]):
        trai = tra2[i,0:2,:]
        ax.plot(trai[0,:],trai[1,:],color='r')
        ax.scatter(trai[0,horizon-1],trai[1,horizon-1])
        
    ax.grid(which='major')
    plt.show()


def plot_Q(Q,title,xticks):
    num_Q = Q.shape[0]
    x = np.arange(0,num_Q) + 1
    x_max = np.argmax(Q)
    Q_max = np.max(Q)
    text_max = str(xticks[x_max])

    fig, ax = plt.subplots(figsize=(15,30))
    ax.plot(x,Q)
    ax.scatter(x_max,Q_max)
    ax.annotate(text_max, (x_max, Q_max) )
    ax.set_aspect('auto')
    
    plt.title(title)
    labels = np.array([str(xx) for xx in xticks])
    plt.xticks(x, labels, rotation = 'vertical',fontsize = 5)
    fig.tight_layout()
    # Adjust more bottom space
    plt.subplots_adjust(bottom = 0.15)
    # Show and close figure auto. 1 sec
    plt.show()
    #plt.show(block=False)
    #plt.pause(0.7)
    #plt.close()

def vis_ego_cost(Q_merge,Q_coli,Q_total,title,xticks):
    num_Q = Q_merge.shape[0]
    x = np.arange(0,num_Q) + 1
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(x,Q_merge,'r')
    ax.plot(x,Q_coli,'g')
    ax.plot(x,Q_total,'b')
    x_min = np.argmin(Q_total)
    cost_min = np.min(Q_total)
    opt_act = xticks[x_min]
    ax.scatter(x_min,cost_min)

    ax.set_aspect('auto')
    plt.title(title+str(opt_act))
    labels = np.array([str(xx) for xx in xticks])
    plt.xticks(x, labels, rotation = 'vertical',fontsize = 5)
    

    fig.tight_layout()
    # Adjust more bottom space
    plt.subplots_adjust(bottom = 0.15)
    # Show and close figure auto. 1 sec
    plt.show()
    #plt.show(block=False)
    #plt.pause(0.7)
    #plt.close()


def vis_HF_reward(s):
    return 0

def vis_Qmatrix(Q,acts_i, acts_j, name):
    num_my_act = acts_i.shape[0]
    num_opp_act = acts_j.shape[0]

    Q = np.array(Q).reshape(num_my_act,num_opp_act)
    print(Q.shape)

    fig, ax = plt.subplots()
    ax.imshow(Q, cmap = "hot", aspect = "auto")
    
    y_ticks = np.arange(num_my_act)
    ax.set_yticks(y_ticks)
    labels = np.array([str(y) for y in acts_i])
    ax.set_yticklabels(labels, fontsize = 3)

    x_ticks = np.arange(num_opp_act)
    labels = np.array([str(xx) for xx in x_ticks])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels, fontsize = 3)
    
    #labels = np.array([str(y) for y in acts_i])
    #ax.set_yticklabels(labels)
    plt.xlabel('actions j')
    plt.ylabel('action i')
    #ax.grid(which='both')
    plt.tight_layout()
    if num_opp_act <= 1:
        opt_act = acts_i[np.argmax(Q)]
        plt.title(name + str(opt_act))
    else:
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
