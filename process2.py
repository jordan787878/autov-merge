import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.animation as animation
import time
#import seaborn as sns
from matplotlib.pyplot import cm

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

def CompareTraj(X):
    list_dict = []
    num_data = len(X)
    for i in range(num_data):
        mydict = np.load('output/'+X[i], allow_pickle=True)
        list_dict.append(mydict)
   
    color = cm.rainbow(np.linspace(0, 1, num_data))
    #color = sns.color_palette(n_colors=num_data)
    print(color)

    k = 0
    fig, axs = plt.subplots()
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
    axs.set_xticks(np.linspace(0,150,31))
    plt.title('Ego Follower vs Highway Leader Merge Trajectories')
    plt.xlabel('x')
    plt.ylabel('y')
   # '''
    plt.savefig('src/figs/0402_EF_HL_Trajetctories_random.png',
                pad_inches = 1,
                bbox_inches ="tight",
               )
   # '''
   # plt.show()

if __name__ == "__main__":
    input_data = INPUT_LIST 
    #print(input_data)
    CompareTraj(input_data)

