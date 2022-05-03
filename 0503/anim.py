import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.pyplot import cm
import matplotlib.colors as clr
import matplotlib.colors as colors
import matplotlib.animation as animation
import time
from visual2 import *

#LOAD_PATH = input("filename: ") 
ANIM_PATH = 'animations/1.mp4'

# use colormap
colormap = cm.viridis

def scatter_color(v):
    normv = (v-18)/(30-18)
    color = colormap(normv)
    return colors.rgb2hex(color)

def animate(i,scs,sc_lfs,d,acts):
    try:
        #print(i)
        d0 = d[i,:]
        actions = acts[i,:]
    except:
        print("simultion ends")
        sys.exit()

    for i in range(len(scs)):
        scs[i].set_offsets(np.hstack((d0[4*i],d0[4*i+1])))
        scs[i].set_color(scatter_color(d0[4*i+2]))
        if i != 0:
            sc_lfs[i-1].set_offsets(np.hstack((d0[4*i],d0[4*i+1])))
    

    time.sleep(0.001)
    
    return i

def ShowAnim(X):
    mydict = np.load(X,allow_pickle=True)
    data = mydict.item().get('states')

    # debug
    #print(data.shape)

    vis_xdiff(data)

    actions = mydict.item().get('actions')
    vis_actions(actions)

    #print(actions)
    lane_width = mydict.item().get('lane_width')
    num_cars = mydict.item().get('num_cars')
    lf_roles = mydict.item().get('lf_roles')
    x0_text = str(mydict.item().get('x0'))
    fig = plt.figure()
    fig.set_figheight(10)
    fig.set_figwidth(20)
    ax = plt.axes()
    lf_text = ['Follower','Leader']
    lf_title = [lf_text[x] for x in lf_roles]
    plt.title(str(lf_title),fontsize = 25)

#   print(data)

    ax.set_xlim([-100,400])
    ax.set_ylim([0-0.5*lane_width,2.5+0.5*lane_width])
    ax.set_xticks(np.linspace(-100,400,101))
    scs = []
    sc_lfs = []
    colors_lf = ['b','r']

    for i in range(num_cars):
        act_car_i_k = actions[0,0]
        if i != 0:
            sc_lf = ax.scatter(data[0,4*i], data[0,4*i+1], c = colors_lf[lf_roles[i-1]], s = 80, alpha = 0.5)
            sc_lfs.append(sc_lf)
            #sc = ax.scatter(data[0,4*i], data[0,4*i+1], c = colors[act_car_i_k], s = 200, edgecolors='black')
            sc = ax.scatter(data[0,4*i], data[0,4*i+1], c = scatter_color(data[0,4*i+2]), s = 40, edgecolors='black')
        else:
            #sc = ax.scatter(data[0,4*i], data[0,4*i+1], c = colors[act_car_i_k], s = 400, edgecolors='black')
            sc = ax.scatter(data[0,4*i], data[0,4*i+1], c = scatter_color(data[0,4*i+2]), s = 80, edgecolors='black')
        scs.append(sc)

    ax.plot(ax.get_xlim(),[lane_width/2,lane_width/2],color='black',linewidth=1.5)
    ax.set_aspect(3)
    ax.xaxis.grid(True)
    plt.xticks(fontsize = 5.5)
    plt.yticks(fontsize = 5.5)
    plt.tight_layout()
    anim = animation.FuncAnimation(fig,animate,interval=1,fargs=(scs,sc_lfs,data,actions))
#    anim.save('animations/'+str(lf_title) + str(x0_text)+'.mp4', fps=8, extra_args=['-vcodec', 'libx264'])
    plt.show()



if __name__ == "__main__":
    input_data = input("filename: ")
    ShowAnim(input_data)

