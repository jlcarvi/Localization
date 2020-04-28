# -*- coding: utf-8 -*-
"""
Created on Tue Aug 07 14:50:54 2018

@author: jlcar
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier


from plot_classification_report import plot_classification_report
from plot_confusion_matrix import plot_confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix_Precision
from clf_classifier import clf_classifier
from conditionalMetaLearner import clf_conditionalMetaLearner 


"""
*************************************************************************************
*********************** UPLOAD THE DATASET ******************************************
*************************************************************************************
"""
""" ************************Use same dataset for training and testing***********

df=pd.read_csv('datasets/train.csv')
y_data=df['zone']
x_data=df.drop(['zone','temp','light','presure','humidity','gravityX',
                'gravityY','gravityZ','magX','magY','magZ','gravityMagnitude',
                'magMagnitude','latitude','longitude'],axis=1) #axis=1 because it is a column

#split data into training and test datasets
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.3,random_state=100)


"""

"""  **************DATASET ATTRIBUTES ************
zone,temp,light,	presure,	humidity,	gravityX,	gravityY,	gravityZ,	magX,	magY,	magZ,	
gravityMagnitude,	magMagnitude,	magEarthY,	magEarthZ,	latitude,	longitude,	rss0,	rss1,
rss2,	rss3,	rss4,	rss5,	rss6,	rss7,	rss8,	rss9

"""



""" ************************Use different training and test dataset***********
"""
dfTrain=pd.read_csv('datasets/train.csv')
dfTest=pd.read_csv('datasets/test.csv')
y_train=dfTrain['zone']
y_test=dfTest['zone']

"""
******************* Experiment Wifi 5 AN

x_train=dfTrain.drop(['zone','light','temp','presure','humidity','gravityX',
                'gravityY','gravityZ','magX','magY','magZ','gravityMagnitude',
                'magMagnitude','magEarthY',	'magEarthZ','latitude','longitude',
                'rss5','rss6','rss7','rss8','rss9'],axis=1) #axis=1 because it is a column


x_test=dfTest.drop(['zone','light','temp','presure','humidity','gravityX',
                'gravityY','gravityZ','magX','magY','magZ','gravityMagnitude',
                'magMagnitude','magEarthY',	'magEarthZ','latitude','longitude',
                'rss5','rss6','rss7','rss8','rss9'],axis=1) #axis=1 because it is a column

"""


"""
******************* Experiment Wifi 6 AN

x_train=dfTrain.drop(['zone','light','temp','presure','humidity','gravityX',
                'gravityY','gravityZ','magX','magY','magZ','gravityMagnitude',
                'magMagnitude','magEarthY',	'magEarthZ','latitude','longitude',
                'rss6','rss7','rss8','rss9'],axis=1) #axis=1 because it is a column


x_test=dfTest.drop(['zone','light','temp','presure','humidity','gravityX',
                'gravityY','gravityZ','magX','magY','magZ','gravityMagnitude',
                'magMagnitude','magEarthY',	'magEarthZ','latitude','longitude',
                'rss6','rss7','rss8','rss9'],axis=1) #axis=1 because it is a column

"""


"""
******************* Experiment Wifi 7 AN

x_train=dfTrain.drop(['zone','light','temp','presure','humidity','gravityX',
                'gravityY','gravityZ','magX','magY','magZ','gravityMagnitude',
                'magMagnitude','magEarthY',	'magEarthZ','latitude','longitude',
                'rss7','rss8','rss9'],axis=1) #axis=1 because it is a column


x_test=dfTest.drop(['zone','light','temp','presure','humidity','gravityX',
                'gravityY','gravityZ','magX','magY','magZ','gravityMagnitude',
                'magMagnitude','magEarthY',	'magEarthZ','latitude','longitude',
                'rss7','rss8','rss9'],axis=1) #axis=1 because it is a column
"""


"""
******************* Experiment Wifi 8 AN

x_train=dfTrain.drop(['zone','light','temp','presure','humidity','gravityX',
                'gravityY','gravityZ','magX','magY','magZ','gravityMagnitude',
                'magMagnitude','magEarthY',	'magEarthZ','latitude','longitude',
                'rss8','rss9'],axis=1) #axis=1 because it is a column


x_test=dfTest.drop(['zone','light','temp','presure','humidity','gravityX',
                'gravityY','gravityZ','magX','magY','magZ','gravityMagnitude',
                'magMagnitude','magEarthY',	'magEarthZ','latitude','longitude',
                'rss8','rss9'],axis=1) #axis=1 because it is a column
"""


"""
******************* Experiment Wifi 9 AN

x_train=dfTrain.drop(['zone','light','temp','presure','humidity','gravityX',
                'gravityY','gravityZ','magX','magY','magZ','gravityMagnitude',
                'magMagnitude','magEarthY',	'magEarthZ','latitude','longitude',
                'rss9'],axis=1) #axis=1 because it is a column


x_test=dfTest.drop(['zone','light','temp','presure','humidity','gravityX',
                'gravityY','gravityZ','magX','magY','magZ','gravityMagnitude',
                'magMagnitude','magEarthY',	'magEarthZ','latitude','longitude',
                'rss9'],axis=1) #axis=1 because it is a column


"""
"""
******************* Experiment Wifi 9 AN +MF

x_train=dfTrain.drop(['zone','light','temp','presure','humidity','gravityX',
                'gravityY','gravityZ','magX','magY','magZ','gravityMagnitude',
                'magMagnitude','latitude','longitude',
                'rss9'],axis=1) #axis=1 because it is a column


x_test=dfTest.drop(['zone','light','temp','presure','humidity','gravityX',
                'gravityY','gravityZ','magX','magY','magZ','gravityMagnitude',
                'magMagnitude','latitude','longitude',
                'rss9'],axis=1) #axis=1 because it is a column

"""

"""
******************* Experiment Wifi 9 AN +MF +L
"""
x_train=dfTrain.drop(['zone','temp','presure','humidity','gravityX',
                'gravityY','gravityZ','magX','magY','magZ','gravityMagnitude',
                'magMagnitude','latitude','longitude',
                'rss9'],axis=1) #axis=1 because it is a column


x_test=dfTest.drop(['zone','temp','presure','humidity','gravityX',
                'gravityY','gravityZ','magX','magY','magZ','gravityMagnitude',
                'magMagnitude','latitude','longitude',
                'rss9'],axis=1) #axis=1 because it is a column




"""
**************************************************************************************
******************** BUILD CLASSIFICATION MODELS *************************************
**************************************************************************************
"""


#********************DECISION TREE WITH GINI*******************************
clf_gini = DecisionTreeClassifier(criterion = "gini",random_state=15325)
clf_gini.fit(x_train, y_train)


#********************DECISION TREE WITH ENTROPY*******************************
clf_en = DecisionTreeClassifier(criterion = "entropy",random_state=15325)
clf_en.fit(x_train, y_train)


#********************SUPPORT VECTOR MACHINE*******************************
svmclf=svm.SVC(kernel='poly',probability=True,random_state=15325) #kernel=poly, degree=3 default,
svmclf.fit(x_train, y_train)

#********************************* MLP ***********************************
mlpclf=MLPClassifier(solver='adam',random_state=15325)  #solver='lbfgs'-->0.7583 solver='sgd'-->0.79, solver='adam'-->0.79
mlpclf.fit(x_train, y_train)

#********************************* KNN ***********************************
KNNclf=neighbors.KNeighborsClassifier(n_neighbors=3) #n_neighbors=5 default
KNNclf.fit(x_train, y_train)

#********************************* NAIVE BAYES ***********************************
NBclf=GaussianNB()
NBclf.fit(x_train, y_train)

#********************************* Soft Voting ***********************************
SoftVotingclf=VotingClassifier(estimators=[('dt',clf_gini),('svm',svmclf),('mlp',mlpclf),('knn',KNNclf),('nb',NBclf)],voting='soft')
SoftVotingclf.fit(x_train, y_train)

#********************************* Bagging ***********************************
Baggingclf=BaggingClassifier()
Baggingclf.fit(x_train, y_train)



"""
***************************************************************************************
******************* CHECK CLASSIFICATION PERFORMANCE **********************************
***************************************************************************************
"""
y_predict_gini=clf_gini.predict(x_test) #decisioj=n tree gini
y_predict_en=clf_en.predict(x_test)     #decision tree entropy
y_predict_svm=svmclf.predict(x_test)    #Support vector Machine
y_predict_mlp=mlpclf.predict(x_test)    #MLP
y_predict_knn=KNNclf.predict(x_test)    #KNN 
y_predict_nb=NBclf.predict(x_test)    #Naive Bayes 
y_predict_sv=SoftVotingclf.predict(x_test)    #soft Voting
y_predict_ba=Baggingclf.predict(x_test)    #Bagging

 


#***************Print the performance report of the classifier
labels=[1, 2,3,4,5,6,7,8,9]
target_names=['1','2','3','4','5','6','7','8','9']
modelDirectory='models/WiFi_MF_L/'

print '******* Performance classification with GINI ***********'
report_gini=classification_report(y_test,y_predict_gini,labels=labels,target_names=target_names)
print report_gini

print 'Accuracy gini: ', accuracy_score(y_test,y_predict_gini)
cmgini=confusion_matrix(y_test,y_predict_gini,labels=labels)


#*******save model in a file
filename = modelDirectory+'clf_gini.sav'
joblib.dump(clf_gini , filename) #save Decision tree gini
#save confusion matrix
cmgini.dump(modelDirectory+'cmgini.dat') #save in a file the confusion matrix


print 'Confusion Matrix ' 
print cmgini


print '************ Performance classification with entropy************'
report_en=classification_report(y_test,y_predict_en,labels=labels,target_names=target_names)
print report_en
print 'Accuracy entropy: ', accuracy_score(y_test,y_predict_en)
cmen=confusion_matrix(y_test,y_predict_en,labels=labels)



#save model in a file
filename = modelDirectory+'clf_en.sav'
joblib.dump(clf_en , filename) #save Decision tree gini
#save confusion matrix
cmen.dump(modelDirectory+'cmen.dat') #save in a file the confusion matrix
print 'Confusion Matrix '
print cmen


print '*********** Performance classification SVM ****************'
report_svm=classification_report(y_test,y_predict_svm,labels=labels,target_names=target_names)
print report_svm
print 'Accuracy svm: ', accuracy_score(y_test,y_predict_svm)
cmsvm=confusion_matrix(y_test,y_predict_svm,labels=labels)



#save model in a file
filename = modelDirectory+'clf_svm.sav'
joblib.dump(svmclf , filename) #save Decision tree gini
#save confusion matrix
cmsvm.dump(modelDirectory+'cmsvm.dat') #save in a file the confusion matrix
print 'Confusion Matrix '
print cmsvm



print '*************** Performance classification MLP***********'
report_mlp=classification_report(y_test,y_predict_mlp,labels=labels,target_names=target_names)
print report_mlp
print 'Accuracy mlp: ', accuracy_score(y_test,y_predict_mlp)
cmmlp=confusion_matrix(y_test,y_predict_mlp,labels=labels)



#save model in a file
filename = modelDirectory+'clf_mlp.sav'
joblib.dump(mlpclf , filename) #save Decision tree gini
#save confusion matrix
cmmlp.dump(modelDirectory+'cmmlp.dat') #save in a file the confusion matrix
print 'Confusion Matrix ' 
print cmmlp



print '************* Performance classification KNN**********************'
report_knn=classification_report(y_test,y_predict_knn,labels=labels,target_names=target_names)
print report_knn
print 'Accuracy KNN: ', accuracy_score(y_test,y_predict_knn)
cmknn=confusion_matrix(y_test,y_predict_knn,labels=labels)



#save model in a file
filename = modelDirectory+'clf_knn.sav'
joblib.dump(KNNclf , filename) #save Decision tree gini
#save confusion matrix
cmknn.dump(modelDirectory+'cmknn.dat') #save in a file the confusion matrix
print 'Confusion Matrix ' 
print cmknn


print '******************** Performance classification Naive Bayes *************'
report_nb=classification_report(y_test,y_predict_nb,labels=labels,target_names=target_names)
print report_nb
print 'Accuracy NB: ', accuracy_score(y_test,y_predict_nb)
cmnb=confusion_matrix(y_test,y_predict_nb,labels=labels)



#save model in a file
filename = modelDirectory+'clf_nb.sav'
joblib.dump(NBclf, filename) #save Decision tree gini
#save confusion matrix
cmnb.dump(modelDirectory+'cmnb.dat') #save in a file the confusion matrix
print 'Confusion Matrix ' 
print cmnb





print '******************** Performance classification Soft Voting **********************'
report_sv=classification_report(y_test,y_predict_sv,labels=labels,target_names=target_names)
print report_sv
print 'Accuracy SV: ', accuracy_score(y_test,y_predict_sv)
cmsv=confusion_matrix(y_test,y_predict_sv,labels=labels)



#save model in a file
filename = modelDirectory+'clf_sv.sav'
joblib.dump(SoftVotingclf, filename) #save Decision tree gini
#save confusion matrix
cmsv.dump(modelDirectory+'cmsv.dat') #save in a file the confusion matrix
print 'Confusion Matrix'
print cmsv


print '******************** Performance classification Bagging **********************'
report_ba=classification_report(y_test,y_predict_ba,labels=labels,target_names=target_names)
print report_sv
print 'Accuracy Bagging: ', accuracy_score(y_test,y_predict_ba)
cmba=confusion_matrix(y_test,y_predict_ba,labels=labels)



#save model in a file
filename = modelDirectory+'clf_ba.sav'
joblib.dump(Baggingclf, filename) #save Decision tree gini
#save confusion matrix
cmba.dump(modelDirectory+'cmba.dat') #save in a file the confusion matrix
print 'Confusion Matrix'
print cmba







print '******************** Performance classification Conditional**********************'

cl1=clf_classifier(modelDirectory+'clf_gini.sav',modelDirectory+'cmgini.dat') #CART GINI 
cl2=clf_classifier(modelDirectory+'clf_en.sav',modelDirectory+'cmen.dat')   #CART ENTROPY
cl3=clf_classifier(modelDirectory+'clf_knn.sav',modelDirectory+'cmknn.dat') #KNN
cl4=clf_classifier(modelDirectory+'clf_nb.sav',modelDirectory+'cmnb.dat') #NB
cl5=clf_classifier(modelDirectory+'clf_svm.sav',modelDirectory+'cmsvm.dat')#SVM
cl6=clf_classifier(modelDirectory+'clf_mlp.sav',modelDirectory+'cmmlp.dat')#MLP
cl7=clf_classifier(modelDirectory+'clf_sv.sav',modelDirectory+'cmsv.dat')#SV



estimators=[cl1,cl3,cl4,cl5,cl6]
Obj=clf_conditionalMetaLearner(estimators)
y_predictClass,y_predictProb=Obj.predict(np.matrix(x_test))


#Accuracy
print '********** classification Performance Conditional ensemble learner ***********'
report_conditional=classification_report(y_test,y_predictClass,labels=labels,target_names=target_names)
print report_conditional
print 'Accuracy conditional: ',accuracy_score(y_test,y_predictClass)  #grounf truth, predicted
print 'Confusion matrix: '
cmConditional=confusion_matrix(y_test,y_predictClass,labels=labels)
print cmConditional
 




#********** CONDITIONAL CLASSIFIER PERFORMANCE PLOTS**********

#***********CONDITIONAL ************
fig=plot_confusion_matrix(cmConditional,classes=target_names,normalize=False,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmConditional.pdf', bbox_inches='tight')
fig=plot_confusion_matrix(cmConditional,classes=target_names,normalize=True,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmRecallConditional.pdf', bbox_inches='tight')
fig=plot_confusion_matrix_Precision(cmConditional,classes=target_names,normalize=True,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmPrecisionConditional.pdf', bbox_inches='tight')



# *****************************CART GINI *******************
fig=plot_confusion_matrix(cmgini,classes=target_names,normalize=False,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmGini.pdf', bbox_inches='tight')
fig=plot_confusion_matrix(cmgini,classes=target_names,normalize=True,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmRecallGini.pdf', bbox_inches='tight')

fig=plot_confusion_matrix_Precision(cmgini,classes=target_names,normalize=True,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmPrecisionGini.pdf', bbox_inches='tight')

# *****************************CART EMTROPY *******************
fig=plot_confusion_matrix(cmen,classes=target_names,normalize=False,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmEn.pdf', bbox_inches='tight')
fig=plot_confusion_matrix(cmen,classes=target_names,normalize=True,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmRecallEn.pdf', bbox_inches='tight')

fig=plot_confusion_matrix_Precision(cmen,classes=target_names,normalize=True,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmPrecisionEn.pdf', bbox_inches='tight')

# *****************************SVM *******************
fig=plot_confusion_matrix(cmsvm,classes=target_names,normalize=False,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmSVM.pdf', bbox_inches='tight')
fig=plot_confusion_matrix(cmsvm,classes=target_names,normalize=True,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmRecallSVM.pdf', bbox_inches='tight')
fig=plot_confusion_matrix_Precision(cmsvm,classes=target_names,normalize=True,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmPrecisionSVM.pdf', bbox_inches='tight')

# *****************************MLP *******************

fig=plot_confusion_matrix(cmmlp,classes=target_names,normalize=False,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmMLP.pdf', bbox_inches='tight')
fig=plot_confusion_matrix(cmmlp,classes=target_names,normalize=True,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmRecallMLP.pdf', bbox_inches='tight')
fig=plot_confusion_matrix_Precision(cmmlp,classes=target_names,normalize=True,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmPrecisionMLP.pdf', bbox_inches='tight')


#************************* ************KNN******************
fig=plot_confusion_matrix(cmknn,classes=target_names,normalize=False,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmKNN.pdf', bbox_inches='tight')
fig=plot_confusion_matrix(cmknn,classes=target_names,normalize=True,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmRecallKNN.pdf', bbox_inches='tight')
fig=plot_confusion_matrix_Precision(cmknn,classes=target_names,normalize=True,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmPrecisionKNN.pdf', bbox_inches='tight')

#*****************************NB ************
fig=plot_confusion_matrix(cmnb,classes=target_names,normalize=False,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmNB.pdf', bbox_inches='tight')
fig=plot_confusion_matrix(cmnb,classes=target_names,normalize=True,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmRecallNB.pdf', bbox_inches='tight')
fig=plot_confusion_matrix_Precision(cmnb,classes=target_names,normalize=True,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmPrecisionNB.pdf', bbox_inches='tight')

#***********SV ************
fig=plot_confusion_matrix(cmsv,classes=target_names,normalize=False,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmSV.pdf', bbox_inches='tight')
fig=plot_confusion_matrix(cmsv,classes=target_names,normalize=True,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmRecallSV.pdf', bbox_inches='tight')
fig=plot_confusion_matrix_Precision(cmsv,classes=target_names,normalize=True,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmPrecisionSV.pdf', bbox_inches='tight')


#***********BAGGING ************
fig=plot_confusion_matrix(cmba,classes=target_names,normalize=False,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmba.pdf', bbox_inches='tight')
fig=plot_confusion_matrix(cmba,classes=target_names,normalize=True,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmRecallBA.pdf', bbox_inches='tight')
fig=plot_confusion_matrix_Precision(cmsv,classes=target_names,normalize=True,title='',
                      ylabel='Ground truth zone',xlabel='Predicted zone')
fig.savefig(modelDirectory+'cmPrecisionBA.pdf', bbox_inches='tight')




