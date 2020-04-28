# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:18:14 2018

@author: jlcarvi
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
"""
******** GRAPH RESULTS OF CLASSIFICATION OF 5 WiFi ANs *************
"""

def graphAccuracyTOTAL_WiFi_MF():
    #Plot accuracy of ensemble learning classifiers vs number of WiFi APs
    # data to plot
    #column 0=5WiFi, column1=6WiFi ...
  
    accuracy=np.array([[88.3],#CART
                  [94.1],#SVM
                  [94.8],#MLP
                  [95.6],#KNN
                  [93.6],#NB
                  [96.1],#SV
                  [96.8]]) #Conditional
    
    

    classifiersLabel=['CART','SVM','MLP','KNN','NB','SV','COND']

    n_groups = accuracy.shape[1] #number of columns



    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.17
    opacity = 0.80

    plt.bar(index, accuracy[0,:], bar_width,alpha=opacity, label='CART')
    plt.bar(index + bar_width, accuracy[1,:], bar_width, alpha=opacity, color='r', label='SVM')
    plt.bar(index + 2* bar_width, accuracy[2,:], bar_width, alpha=opacity,color='y', label='MLP')
    plt.bar(index + 3* bar_width, accuracy[3,:], bar_width, alpha=opacity, color='m', label='KNN')
    plt.bar(index + 4* bar_width, accuracy[4,:], bar_width,alpha=opacity, color='g', label='NB')
    plt.bar(index + 5* bar_width, accuracy[5,:],bar_width,alpha=opacity, color='orange', label='SV')
    plt.bar(index + 6* bar_width, accuracy[6,:],bar_width, alpha=opacity, color='cyan', label='Cond')
    
    
    
    plt.xlabel('Classifiers', size=12)
    plt.ylabel('Correctly classified instances (%)',size=12)
    
  
    
    ax.set_xticks(index+ 3*bar_width)
    ax.set_xticklabels((''),size=12)
    
    
    startY=80  #which number to start in Y axis
    plt.ylim(startY,100)
    plt.yticks( np.arange(startY,100,5),size=12)
    
    
    for i in index:
        value=0
        for bar in accuracy[:,i]:
            plt.text(i+value*bar_width-0.07,y=bar-1,s=classifiersLabel[value]+'   '+str(bar)+'%' ,size=12,rotation=90)
            value=value+1
    
    #plt.legend()
     
    plt.tight_layout()
    plt.gca().yaxis.grid(True)
    plt.show()
    fig.savefig('models/WiFiTotal/AccuracyTOTALWiFi_MFRigthPart.pdf')




#*********** RUN ********
graphAccuracyTOTAL_WiFi_MF()  



