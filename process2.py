import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.animation as animation
import time
#import seaborn as sns
from matplotlib.pyplot import cm

DATA_DIR = ''

DATA1 = 'data1.txt'
DATA2 = 'data2.txt'

TITLE = '0403 HL EF Test'

SAVE_FIG = False

INPUT_LIST = [
              '0402_HL_EF_Test_[0 0].npy',
              '0402_HL_EF_Test_[1 0].npy',
              '0402_HL_EF_Test_[2 0].npy',
              '0402_HL_EF_Test_[3 0].npy',
              '0402_HL_EF_Test_[4 0].npy',
              '0402_HL_EF_Test_[5 0].npy',
              '0402_HL_EF_Test_[6 0].npy',
              '0402_HL_EF_Test_[7 0].npy',
              '0402_HL_EF_Test_[8 0].npy',
              '0402_HL_EF_Test_[9 0].npy',
              '0402_HL_EF_Test_[10  0].npy',
              '0402_HL_EF_Test_[11  0].npy',
              '0402_HL_EF_Test_[12  0].npy',
              '0402_HL_EF_Test_[13  0].npy',
              '0402_HL_EF_Test_[14  0].npy',
             # '0402_HL_EF_Test_[15  0].npy',
             # '0402_HL_EF_Test_[16  0].npy',
             # '0402_HL_EF_Test_[17  0].npy',
             # '0402_HL_EF_Test_[18  0].npy',
              ]

# Random
#'''
INPUT_LIST = [              
              '0402_HL_EF_Test_[14 11].npy',
             # '0402_HL_EF_Test_[26 44].npy',
              '0402_HL_EF_Test_[34 33].npy',
              '0402_HL_EF_Test_[57 46].npy',
             # '0402_HL_EF_Test_[11 19].npy',
             # '0402_HL_EF_Test_[30 17].npy',
              '0402_HL_EF_Test_[44 26].npy',
              '0402_HL_EF_Test_[12 12].npy',
              ]
#'''
INPUT_LIST = [
             '0402_3HF_EL_Test_[16  7 32 49].npy',
             ]


def CompareTraj(X,fig,axs):
#    fig, axs = plt.subplots()
    
    list_dict = []
    num_data = len(X)
    for i in range(num_data):
        mydict = np.load(DATA_DIR+X[i], allow_pickle=True)
        list_dict.append(mydict)
        if i == 0:
            car_dynamics = mydict.item().get('car_dynamics')
            v_min = car_dynamics['v_min']
            v_max = car_dynamics['v_max']
            accel = car_dynamics['accel']
            car_info = '\nv range:(' + str(v_min) +','+ str(v_max) + ') accel: ' + str(accel)
    
    # Create Color Code
    color = cm.rainbow(np.linspace(0, 1, num_data))
    #color = sns.color_palette(n_colors=num_data)
    print(color)

    k = 0
    for dic in list_dict:
        data = dic.item().get('states')
        merge_time = np.where(data[:,1] == 2.5)
        #print(merge_time)
        num_cars = dic.item().get('num_cars')
        for i in range(num_cars):
            x = data[:,4*i]
            y = data[:,4*i+1]
            axs.plot(x,y,color=color[k],linestyle='dashed',alpha = 1.0, linewidth=1.5)
            axs.scatter(data[0,4*i],data[0,4*i+1],color=color[k],s=150)
            axs.scatter(data[merge_time[0],4*i],data[merge_time[0],4*i+1],color=color[k],s=150,marker='X')
        k = k+1

    fig.set_figheight(15)
    fig.set_figwidth(30)
    axs.set_aspect(15)
    axs.xaxis.grid(True)
    axs.set_facecolor("black")
    axs.set_xticks(np.linspace(0,200,41))

    return car_info
#    plt.title('Ego Follower vs Highway Leader Merge Trajectories')
#    plt.xlabel('x')
#    plt.ylabel('y')
    '''
    plt.savefig('src/figs/0402_EF_HL_Trajetctories_random.png',
                pad_inches = 1,
                bbox_inches ="tight",
               )
    '''
#    plt.show()

def CreatePlots(X1,X2):
    fig, (ax1, ax2) = plt.subplots(2,1)
    car_info = CompareTraj(X1,fig,ax1)
    car_info = CompareTraj(X2,fig,ax2)

    title_txt = TITLE + car_info
    plt.suptitle(title_txt)
    plt.xlabel('x')
    plt.ylabel('y')
    if SAVE_FIG:
        plt.savefig('src/figs/'+title_txt+'.png', pad_inches = 1, bbox_inches ="tight")
    else:
        plt.show()

def Readtxt(d):
    my_file = open(DATA_DIR+d, "r")

    # reading the file
    data = my_file.read()

    # replacing end of line('/n') with ' ' and
    # splitting the text it further when '.' is seen.
    #data_into_list = data.replace('\n', ' ').split(".")
    data_into_list = data.split('\n')
    data_into_list.pop()

    # printing the data
    return data_into_list


if __name__ == "__main__":

    DATA_DIR = input("data dir? ")

    input_data1 = Readtxt(DATA1)

    input_data2 = Readtxt(DATA2)
    
    CreatePlots(input_data1,input_data2)

