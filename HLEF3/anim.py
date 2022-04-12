import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.animation as animation
import time

#LOAD_PATH = input("filename: ") 

def animate(i,scs,d,acts,colors):
    try:
        #print(i)
        d0 = d[i,:]
        actions = acts[i,:]
    except:
        print("simultion ends")
        sys.exit()

    for i in range(len(scs)):
        scs[i].set_offsets(np.hstack((d0[4*i],d0[4*i+1])))
        scs[i].set_color(colors[actions[i]])
    
    time.sleep(0.01)
    
    return i

def ShowAnim(X):
    mydict = np.load(X,allow_pickle=True)
    data = mydict.item().get('states')
    actions = mydict.item().get('actions')
    #print(actions)
    lane_width = mydict.item().get('lane_width')
    num_cars = mydict.item().get('num_cars')
    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(30)
    ax = plt.axes()

#   print(data)

    ax.set_xlim([0,200])
    ax.set_ylim([0-0.5*lane_width,2.5+0.5*lane_width])
    ax.set_xticks(np.linspace(0,200,21))
    scs = []
    colors = ['r','g','b','y']

    for i in range(num_cars):
        act_car_i_k = actions[0,0]
        sc = ax.scatter(data[0,4*i], data[0,4*i+1], c = colors[act_car_i_k], s = 200)
        scs.append(sc)

    ax.plot(ax.get_xlim(),[lane_width/2,lane_width/2],color='black',linewidth=3)
    ax.set_aspect(4)
    ax.xaxis.grid(True)
    anim = animation.FuncAnimation(fig,animate,interval=2,fargs=(scs,data,actions,colors))
    plt.show()


if __name__ == "__main__":
    input_data = input("filename: ")
    ShowAnim(input_data)

