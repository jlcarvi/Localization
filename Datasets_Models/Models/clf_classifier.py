# -*- coding: utf-8 -*-
"""
Created on Thu Aug 09 15:36:38 2018

@author: jlcarvi
"""

from sklearn.externals import joblib
import numpy as np

class clf_classifier:
    def __init__(self, modelFile,confusionMatrixFile):
        #upload model
        self.model = joblib.load(modelFile)
        self.confusionMatrix=np.load(confusionMatrixFile)
        #print 'Confusion ****',self.confusionMatrix
        #Generate minimun Confusion matrix by adding 1 to all elements in the original matrix
        self.confusionMatrix=self.confusionMatrix+1.0
        #build the RECALL conditional matrix
        self.cmRecall=self.confusionMatrix/self.confusionMatrix.sum(axis=1,keepdims=True)
        self.cmPrecision=self.confusionMatrix/self.confusionMatrix.sum(axis=0,keepdims=True) 
        
              
    def predict(self,x_text):
        prediction=self.model.predict(x_text)
        return prediction
        
    def predict_prob(self,x_text):
        prediction_proba=self.model.predict_proba(x_text)
        return prediction_proba
    
   
"""
Obj=clf_classifier('classifiers/clf_gini.sav','classifiers/cmsv.dat')
test_individual=np.array([[12,7,6,44,5,21,-17,30,-12]])
p=Obj.predict(test_individual)
print 'Prediction: ',p
"""