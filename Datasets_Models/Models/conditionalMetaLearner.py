# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:06:34 2018

@author: jlcarvi
"""
import numpy as np

class clf_conditionalMetaLearner:
     def __init__(self, estimators):
         self.estimators=estimators
         #self.individualPred=[]
         
     #*********** TO PREDICT A SET OF INSTANCES*****
     #Predict a set of instances
     #input: np matrix, each row of the matrix is an instance to classify 
     #output: list of classified labels and the respective probablility distribution 
     def predict(self,x_text):
         predictionClass=[]
         predictionProb=[]
         for instance in x_text:
             zone, prob=self.predictInd(instance.reshape(1,-1))  #reshape(1,-1) reshape in just 1 row
             predictionClass.append(zone)
             predictionProb.append(prob)
            
         return predictionClass, predictionProb
         
     #*********** TO PREDICT INDIVIDUALLY*****
     #Return the predicted zone and the probabily distribution of the prediction
     def predictInd(self,instance):   
         #Individual prediction
         individualPred=[]
         for est in self.estimators:
             indPrediction=est.predict(instance) #return the predicted class
             individualPred.append((indPrediction[0],est))
            
         numbZones= est.cmRecall.shape[0] #number of classes in the matrix
         PC_Z=[]
         
         for zone in range(numbZones):
             PC_Z.append(1.)           
             for indPred in individualPred:
                #indPred[1] is the model. Delete '-1'if zone start in 1
                 PC_Z[zone]=PC_Z[zone]*indPred[1].cmRecall[zone][indPred[0]-1] 
                
         #normalize PC_Z 
         PC_Z=PC_Z/sum(PC_Z)
         return  np.argmax(PC_Z)+1, PC_Z
     
            
             
