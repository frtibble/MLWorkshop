import matplotlib
import matplotlib
matplotlib.use('TkAgg')
import sklearn.metrics as m
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix as cm

def azureml_main(dataframe1 = None):
    dataframe1 = dataframe1.dropna()
    # pick the label out of the dataframe and the positive label and plot the RoC curve
    r1 = m.roc_curve(dataframe1['HasDiabetes'], dataframe1['Scored Probabilities'], pos_label= 1)
    plt.plot(r1[0], r1[1], 'r-', label='Logistic Regression RoC curve')
    plt.grid('on')
    plt.legend(loc='best')
    plt.savefig('roc.png')
    #Derive test statistics to show the accuracy of the model and output on POrt1
    cmarray =  cm(dataframe1['HasDiabetes'], dataframe1['Scored Labels'])
    TrueNeg, FalsePos = cmarray[0]
    FalseNeg, TruePos = cmarray[1]
    TruePos = float(TruePos)
    FalseNeg = float(FalseNeg)
    FalsePos= float(FalsePos)
    TrueNeg= float(TrueNeg)
    
    Accuracy = (TruePos + TrueNeg)/(TruePos+ TrueNeg + FalsePos + FalseNeg)
    Recall = TruePos/(TruePos + FalseNeg)
    Precision = TruePos/(TruePos+ FalsePos)
    F1Score = 2 * (Precision * Recall)/(Precision + Recall)

    data = {'Description': ['True Positives','False Negatives','False Positives','True Negatives','Accuracy','Precision','Recall','F1 Score'],'Score': [TruePos,FalseNeg,FalsePos,TrueNeg,Accuracy,Precision,Recall,F1Score]}
    dataframe1 =pd.DataFrame(data,columns=['Description','Score'])
    return dataframe1,