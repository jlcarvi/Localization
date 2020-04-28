# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:18:14 2018

@author: jlcarvi
"""
import numpy as np
import matplotlib.pyplot as plt
"""
******** GRAPH RESULTS OF CLASSIFICATION OF 5 WiFi ANs *************
"""
def graphAccuracyIndividualWiFi():
    #Plot accuracy of individual classifiers vs number of WIFi APs
    # data to plot
    #column 0=5WiFi, column1=6WiFi ...
    accuracy=np.array([[71.5,73.2,76,74.8,72.7],#CART
                  [76.6,78.9,82.1,80.5,82.3],#SVM
                  [84.8,82.7,87.1,88.6,84],#MLP
                  [81.6,84.4,86.9,82.3,85.6],#KNN
                  [85.9,81.6,85.9,81.9,81.1]])#NB
                


    classifiersLabel=['CART','SVM','MLP','KNN','NB']

    n_groups = accuracy.shape[1] #number of columns



    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.17
    opacity = 0.8

 
    plt.bar(index, accuracy[0,:], bar_width,alpha=opacity, label='CART')
    plt.bar(index + bar_width, accuracy[1,:], bar_width, alpha=opacity, color='r', label='SVM')
    plt.bar(index + 2* bar_width, accuracy[2,:], bar_width, alpha=opacity,color='y', label='MLP')
    plt.bar(index + 3* bar_width, accuracy[3,:], bar_width, alpha=opacity, color='m', label='KNN')
    plt.bar(index + 4* bar_width, accuracy[4,:], bar_width,alpha=opacity, color='g', label='NB')
    
    
    plt.xlabel('Number of Wi-Fi RSS attributes', size=12)
    plt.ylabel('Correctly classified instances (%)',size=12)
    
    ax.set_xticks(index+ 3*bar_width)
    ax.set_xticklabels(('5','6','7','8','9'),size=12)
    
    
    startY=50  #which number to start in Y axis
    plt.ylim(startY,100)
    plt.yticks( np.arange(startY,100,10),size=12)
    
    
    for i in index:
        value=0
        for bar in accuracy[:,i]:
            plt.text(i+value*bar_width-0.04,y=bar-4,s=classifiersLabel[value]+'   '+str(bar)+'%' ,size=11,rotation=90)
            #plt.text(i+value*bar_width-0.04,y=startY+2,s=classifiersLabel[value],size=11,rotation=90,va='bottom')

            value=value+1
    
   # plt.legend()
     
    plt.tight_layout()
    plt.gca().yaxis.grid(True)
    plt.show()
    fig.savefig('models/WiFiTotal/AccuracyIndividualWiFi.pdf')

def graphAccuracyEnsembleWiFi():
    #Plot accuracy of ensemble learning classifiers vs number of WiFi APs
    # data to plot
    #column 0=5WiFi, column1=6WiFi ...
    accuracy=np.array([[84.5,86.7,88.3,84.5,84.5],#SV #using all indiviual classifiers but entropy
                       [96.2,94.1,93.2,94.1,94]])#CONDITIONAL Using all individual classifiers but entropy



    classifiersLabel=['SV','COND']

    n_groups = accuracy.shape[1] #number of columns



    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.4
    opacity = 0.80

 
    plt.bar(index, accuracy[0,:], bar_width,alpha=opacity, color='orange', label='SV')
    plt.bar(index + bar_width, accuracy[1,:], bar_width, alpha=opacity, color='cyan', label='Cond')
    
    
    
    plt.xlabel('Number of Wi-Fi RSS attributes', size=12)
    plt.ylabel('Correctly classified instances (%)',size=12)
    
    ax.set_xticks(index+ bar_width/2)
    ax.set_xticklabels(('5','6','7','8','9'),size=12)
    
    
    startY=70  #which number to start in Y axis
    plt.ylim(startY,100)
    plt.yticks( np.arange(startY,100,5),size=12)
    
    
    for i in index:
        value=0
        for bar in accuracy[:,i]:
            plt.text(i+value*bar_width-0.07,y=bar-4,s=classifiersLabel[value]+'   '+str(bar)+'%' ,size=12,rotation=90)
            value=value+1
    
    #plt.legend()
     
    plt.tight_layout()
    plt.gca().yaxis.grid(True)
    plt.show()
    fig.savefig('models/WiFiTotal/AccuracyEnsembleWiFi.pdf')


def graphAccuracyIndividual_WiFi_MF_L():
    #Plot accuracy of ensemble learning classifiers vs number of WiFi APs
    # data to plot
    #column 0=9WiFi, column1=9WiFi+MF column1=9WiFi+MF+Light...
    accuracy=np.array([[72.7,75,76.9],#CART
                  [82.3,81.5,86.3],#SVM
                  [84,86.3,83.7],#MLP
                  [85.6,84.8,84.3],#KNN
                  [81.1,79,90.3]])#NB
                


    classifiersLabel=['CART','SVM','MLP','KNN','NB']



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
    
    
    
    plt.xlabel('Number of Wi-Fi RSS attributes', size=12)
    plt.ylabel('Correctly classified instances (%)',size=12)
    
    ax.set_xticks(index+ 2*bar_width)
    ax.set_xticklabels(('Wi-Fi','Wi-Fi+MF','Wi-Fi+MF+Light'),size=12)
    
    
    startY=50  #which number to start in Y axis
    plt.ylim(startY,100)
    plt.yticks( np.arange(startY,100,10),size=12)
    
    
    for i in index:
        value=0
        for bar in accuracy[:,i]:
            plt.text(i+value*bar_width-0.07,y=bar-4,s=classifiersLabel[value]+'   '+str(bar)+'%' ,size=12,rotation=90)
            value=value+1
    
    #plt.legend()
     
    plt.tight_layout()
    plt.gca().yaxis.grid(True)
    plt.show()
    fig.savefig('models/WiFiTotal/AccuracyIndWiFi_MF_L.pdf')

def graphAccuracyEnsemble_WiFi_MF_L():
    #Plot accuracy of ensemble learning classifiers vs number of WiFi APs
    # data to plot
    #column 0=5WiFi, column1=6WiFi ...
    accuracy=np.array([[84.5,85.6,87.7],#SV #using all indiviual classifiers but entropy
                       [94,95.6,96.6]])#CONDITIONAL Using all individual classifiers but entropy



    classifiersLabel=['SV','COND']

    n_groups = accuracy.shape[1] #number of columns



    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.80

 
    plt.bar(index, accuracy[0,:], bar_width,alpha=opacity, color='orange', label='SV')
    plt.bar(index + bar_width, accuracy[1,:], bar_width, alpha=opacity, color='cyan', label='Cond')
    
    
    
    plt.xlabel('Number of Wi-Fi RSS attributes', size=12)
    plt.ylabel('Correctly classified instances (%)',size=12)
    
    ax.set_xticks(index+ bar_width/2)
    ax.set_xticklabels(('Wi-Fi','Wi-Fi+MF','Wi-Fi+MF+Ligth'),size=12)
    
    
    startY=70  #which number to start in Y axis
    plt.ylim(startY,100)
    plt.yticks( np.arange(startY,100,5),size=12)
    
    
    for i in index:
        value=0
        for bar in accuracy[:,i]:
            plt.text(i+value*bar_width-0.07,y=bar-4,s=classifiersLabel[value]+'   '+str(bar)+'%' ,size=12,rotation=90)
            value=value+1
    
    #plt.legend()
     
    plt.tight_layout()
    plt.gca().yaxis.grid(True)
    plt.show()
    fig.savefig('models/WiFiTotal/AccuracyEnsembleWiFi_MF_L.pdf')



def graphAccuracyTOTAL_WiFi_MF_L(): #Scenario 1
    #Plot accuracy of ensemble learning classifiers vs number of WiFi APs
    # data to plot
    #column 0=5WiFi, column1=6WiFi ...
  
    accuracy=np.array([[76.9],#CART
                  [86.3],#SVM
                  [83.7],#MLP
                  [84.3],#KNN
                  [90.3],#NB
                  [87.7],#SV
                  [96.6],#Conditional
                  [79.6]]) #Bagging
    
    

    classifiersLabel=['CART','SVM','MLP','KNN','NB','SV','COND','BAGGING']

    n_groups = accuracy.shape[1] #number of columns



    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.15
    opacity = 0.80

    plt.bar(index, accuracy[0,:], bar_width,alpha=opacity, label='CART')
    plt.bar(index + bar_width, accuracy[1,:], bar_width, alpha=opacity, color='r', label='SVM')
    plt.bar(index + 2* bar_width, accuracy[2,:], bar_width, alpha=opacity,color='y', label='MLP')
    plt.bar(index + 3* bar_width, accuracy[3,:], bar_width, alpha=opacity, color='m', label='KNN')
    plt.bar(index + 4* bar_width, accuracy[4,:], bar_width,alpha=opacity, color='g', label='NB')
    plt.bar(index + 5* bar_width, accuracy[5,:],bar_width,alpha=opacity, color='orange', label='SV')
    plt.bar(index + 6* bar_width, accuracy[6,:],bar_width, alpha=opacity, color='cyan', label='Cond')
    plt.bar(index + 7* bar_width, accuracy[7,:],bar_width, alpha=opacity, color='magenta', label='Bagging')
    
    
    
    plt.xlabel('Classifiers', size=12)
    plt.ylabel('Correctly classified instances (%)',size=12)
    
  
    
    ax.set_xticks(index+ 3*bar_width)
    ax.set_xticklabels((''),size=12)
    
    
    startY=50  #which number to start in Y axis
    plt.ylim(startY,100)
    plt.yticks( np.arange(startY,100,5),size=12)
    
    
    for i in index:
        value=0
        for bar in accuracy[:,i]:
            plt.text(i+value*bar_width-0.07,y=bar-4,s=classifiersLabel[value]+'   '+str(bar)+'%' ,size=12,rotation=90)
            value=value+1
    
    #plt.legend()
     
    plt.tight_layout()
    plt.gca().yaxis.grid(True)
    plt.show()
    fig.savefig('models/WiFiTotal/AccuracyTOTALWiFi_MF_L.pdf')



def graphAccuracyTOTAL_WiFi_MF_R(): #Scenario 2
    #Plot accuracy of ensemble learning classifiers vs number of WiFi APs
    # data to plot
    #column 0=5WiFi, column1=6WiFi ...
  
    accuracy=np.array([[88.3],#CART
                  [94.1],#SVM
                  [94.8],#MLP
                  [95.6],#KNN
                  [93.6],#NB
                  [96.1],#SV
                  [96.8],#Conditional
                  [84.2]]) #Bagging
    
    

    classifiersLabel=['CART','SVM','MLP','KNN','NB','SV','COND','BAGGING']

    n_groups = accuracy.shape[1] #number of columns



    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.15
    opacity = 0.80

    plt.bar(index, accuracy[0,:], bar_width,alpha=opacity, label='CART')
    plt.bar(index + bar_width, accuracy[1,:], bar_width, alpha=opacity, color='r', label='SVM')
    plt.bar(index + 2* bar_width, accuracy[2,:], bar_width, alpha=opacity,color='y', label='MLP')
    plt.bar(index + 3* bar_width, accuracy[3,:], bar_width, alpha=opacity, color='m', label='KNN')
    plt.bar(index + 4* bar_width, accuracy[4,:], bar_width,alpha=opacity, color='g', label='NB')
    plt.bar(index + 5* bar_width, accuracy[5,:],bar_width,alpha=opacity, color='orange', label='SV')
    plt.bar(index + 6* bar_width, accuracy[6,:],bar_width, alpha=opacity, color='cyan', label='Cond')
    plt.bar(index + 7* bar_width, accuracy[7,:],bar_width, alpha=opacity, color='magenta', label='Bagging')
    
    
    
    plt.xlabel('Classifiers', size=12)
    plt.ylabel('Correctly classified instances (%)',size=12)
    
  
    
    ax.set_xticks(index+ 3*bar_width)
    ax.set_xticklabels((''),size=12)
    
    
    startY=50  #which number to start in Y axis
    plt.ylim(startY,100)
    plt.yticks( np.arange(startY,100,5),size=12)
    
    
    for i in index:
        value=0
        for bar in accuracy[:,i]:
            plt.text(i+value*bar_width-0.07,y=bar-4,s=classifiersLabel[value]+'   '+str(bar)+'%' ,size=12,rotation=90)
            value=value+1
    
    #plt.legend()
     
    plt.tight_layout()
    plt.gca().yaxis.grid(True)
    plt.show()
    fig.savefig('models/WiFiTotal/AccuracyTOTALWiFi_MF_R.pdf')





#*********** RUN ********
#graphAccuracyIndividualWiFi()
#graphAccuracyEnsembleWiFi()
#graphAccuracyIndividual_WiFi_MF_L()
#graphAccuracyEnsemble_WiFi_MF_L()
#graphAccuracyTOTAL_WiFi_MF_L()  
graphAccuracyTOTAL_WiFi_MF_R()  



