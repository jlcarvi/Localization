# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 10:34:25 2018

@author: jlcarvi
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


#generate a normalized matrix with respect to the ground truth class
#this is a recall matrix 
#rate of missed values
def plot_confusion_matrix(cm,classes, normalize=False,title='Recall Confusion Matrix',xlabel='Predicted label',ylabel='True label',cmap=plt.cm.Blues):
    fig, ax = plt.subplots()
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        #print 'Normalized Confusion Matrix'
    #else:
        #print 'Confusion Matrix without Normalization'
    #print cm
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    fmt='.2f' if normalize else 'd'
    thresh=cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt),horizontalalignment='center',color='white' if cm[i,j]>thresh else 'black')
        
    plt.tight_layout()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)   
    return fig              


#generate a normalized matrix with repect to the predicted results
#this is the precision matrix
#rate of caught values    
def plot_confusion_matrix_Precision(cm,classes, normalize=False,title='Precision Confusion Matrix',xlabel='Predicted label',ylabel='True label',cmap=plt.cm.Blues):
    fig, ax = plt.subplots()
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=0)[:,np.newaxis]
       # print 'Normalized Confusion Matrix'
   # else:
    #    print 'Confusion Matrix without Normalization'
   # print cm
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    fmt='.2f' if normalize else 'd'
    thresh=cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt),horizontalalignment='center',color='white' if cm[i,j]>thresh else 'black')
        
    plt.tight_layout()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)   
    return fig
            