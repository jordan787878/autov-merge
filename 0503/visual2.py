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

def vis_tra1_tra2(tra1,tra2):
    print(tra1.shape)
    print(tra2.shape)

    # init figure
    fig, ax = plt.subplots()

    color = ['black','r','g','b','y']
    for k in range(tra1.shape[2]):
        traj_k = tra1[:,:,k]
        ax.scatter(traj_k[0,0],traj_k[0,1]+0.1,c=color[k])
        ax.scatter(traj_k[-1,0],traj_k[-1,1]-0.1,c=color[k])

    for k in range(tra2.shape[2]):
        traj_k = tra2[:,:,k]
        ax.scatter(traj_k[0,0],traj_k[0,1],c=color[k])
        ax.scatter(traj_k[-1,0],traj_k[-1,1],c=color[k])
        ax.scatter(traj_k[11,0],traj_k[11,1],c=color[k])
        ax.scatter(traj_k[12,0],traj_k[12,1],c=color[k])
        ax.scatter(traj_k[27,0],traj_k[27,1],c=color[k])
        ax.scatter(traj_k[48,0],traj_k[48,1],c=color[k])
        ax.scatter(traj_k[122,0],traj_k[122,1],c=color[k])
        ax.scatter(traj_k[176,0],traj_k[176,1],c=color[k])
        #traj_k = tra2[:,:,k:k+2]
        #ax.plot(traj_k[0,0,:],traj_k[0,1,:],c=color[k],marker='o')
        #ax.plot(traj_k[-1,0,:],traj_k[-1,1,:],c=color[k],marker='o')
        #ax.plot(traj_k[11,0,:],traj_k[11,1,:],c=color[k],marker='o')
        #ax.plot(traj_k[12,0,:],traj_k[12,1,:],c=color[k],marker='o')
        #ax.plot(traj_k[27,0,:],traj_k[27,1,:],c=color[k],marker='o')
        #ax.plot(traj_k[45,0,:],traj_k[48,1,:],c=color[k],marker='o')
        #ax.plot(traj_k[122,0,:],traj_k[122,1,:],c=color[k],marker='o')
        #ax.plot(traj_k[176,0,:],traj_k[176,1,:],c=color[k],marker='o')


    plt.xticks(np.arange(150),fontsize = 4)
    plt.grid()


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

def vis_ego_follow(Qfi,title,xticks):
    scale = 5 

    num_Q = Qfi.shape[0]
    x = np.arange(0,num_Q)
    x = scale*x
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(x,Qfi,'r')
    
    x_max = np.argmax(Qfi)
    cost_max = np.max(Qfi)
    opt_act = xticks[x_max]
    ax.scatter(scale*x_max,cost_max)

    ax.set_aspect('auto')
    plt.title(title+str(opt_act))

    colors = ['r','g','b','y']
    for i in range(xticks.shape[0]):
        act0 = xticks[i]
        act0_x = [scale*i,scale*i,scale*i,scale*i]
        act0_y = [-700,-680,-660,-640]
        ax.scatter(act0_x,act0_y,c=[colors[act0[0]],colors[act0[1]],colors[act0[2]],colors[act0[3]]])

    fig.tight_layout()
    plt.tick_params(labelbottom = False, bottom = False)
    plt.show()

def vis_hum_lead(Qfi,title,xticks):
    scale = 5

    num_Q = Qfi.shape[0]
    x = np.arange(0,num_Q)
    x = scale*x

    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,10))
    ax1.plot(x,Qfi,'r')

    x_max = np.argmax(Qfi)
    cost_max = np.max(Qfi)
    opt_act = xticks[x_max]
    ax1.scatter(scale*x_max,cost_max)

    #ax1.set_aspect('auto')
    plt.suptitle(title+str(opt_act))

    colors = ['r','g','b','y']
    for i in range(xticks.shape[0]):
        act0 = xticks[i]
        act0_x = [scale*i,scale*i,scale*i,scale*i]
        act0_y = [0,1,2,3]
        ax2.scatter(act0_x,act0_y,c=[colors[act0[0]],colors[act0[1]],colors[act0[2]],colors[act0[3]]])

    fig.tight_layout()
    plt.tick_params(labelbottom = False, bottom = False)
    plt.show()




def vis_ego_cost(Q_enforce, Q_merge,Q_coli,Q_total,title,xticks):
    scale = 5 

    num_Q = Q_merge.shape[0]
    x = np.arange(0,num_Q)
    x = scale*x
    
    fig, (ax,ax1,ax2,ax3,ax4) = plt.subplots(5,1,figsize=(10,10))
    ax.plot(x,Q_enforce,'r')
    ax1.plot(x,Q_merge,'g')
    ax2.plot(x,Q_coli,'b')
    ax3.plot(x,Q_total,'black')

    ax.set_ylabel('enforce')
    ax1.set_ylabel('merge goal')
    ax2.set_ylabel('colli')
    ax3.set_ylabel('total')

    ax.set_xticks(x)
    ax1.set_xticks(x)
    ax2.set_xticks(x)
    ax3.set_xticks(x)

    ax.grid(which='both')
    ax1.grid(which='both')
    ax2.grid(which='both')
    ax3.grid(which='both')
   
    ax.set_xticklabels([])
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    #ax.set_xticks(color='white')

    
    x_min = np.argmin(Q_total)
    cost_min = np.min(Q_total)
    opt_act = xticks[x_min]
    ax3.scatter(scale*x_min,cost_min)

    ax.set_aspect('auto')
    plt.suptitle(str(opt_act))

    colors = ['r','g','b','y']
    for i in range(xticks.shape[0]):
        act0 = xticks[i]
        act0_x = [scale*i,scale*i,scale*i,scale*i]
        act0_y = [-70,-50,-30,-10]
        ax4.scatter(act0_x,act0_y,c=[colors[act0[0]],colors[act0[1]],colors[act0[2]],colors[act0[3]]])

    #labels = np.array([str(xx) for xx in xticks])
    #plt.xticks(x, labels, rotation = 'vertical',fontsize = 5)

    fig.tight_layout()
    # Adjust more bottom space
    plt.tick_params(labelbottom = False, bottom = False)
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

def vis_actions(acts):
    colors = ['r','g','b','y']
    num_cars = acts.shape[1]
    time = acts.shape[0]
    fig, axs = plt.subplots(num_cars, 1, figsize=(7,7))
    for k in range(time):
        for i in range(num_cars):
            axs[i].scatter(k,acts[k,i],c=colors[acts[k,i]])

    plt.suptitle('actions')
    plt.show()

def vis_xdiff(states):
    num_hum_cars = int(states.shape[1]/4)-1
    x_ego = states[:,0]
    y_ego = states[:,1]
    time = np.arange(states.shape[0])

    # Get Ego Merge Time
    time_merge = np.argwhere(y_ego >= 1.0)[0]
    time_merge = time_merge[0]

    # Init Min X Diff. after merge
    min_x_diff_merge = np.zeros(num_hum_cars)

    fig, axs = plt.subplots(num_hum_cars, 1, figsize = (7,7))
    for i in range(num_hum_cars):
        x_hum = states[:,4*(i+1)]
        y_hum = states[:,4*(i+1)+1]
        x_diff = abs(x_ego-x_hum)
        y_diff = abs(y_ego-y_hum)
        axs[i].plot(time,x_diff)
        axs[i].plot(time,y_diff)
        #axs[i].plot(time[0:time_merge],x_diff[0:time_merge],'b')
        #axs[i].plot(time[time_merge-1:],x_diff[time_merge-1:],'r')
        #axs[i].plot(time,0*time + safe_x_diff,'g')
        axs[i].set_ylim([0,7])
        #min_x_diff_merge[i] = np.min(x_diff[time_merge-1:])

    #print(min_x_diff_merge)

    plt.suptitle('delta x between ego and human vehicles')
    plt.show()
        

def main():
    Q = np.array([1,2,3])
    vis_Q(Q)

#main()
