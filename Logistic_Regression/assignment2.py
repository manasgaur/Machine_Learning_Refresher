
'''
Identifying the relationship between Risk and Flu. Modeling it using logistic Regression using 1 variable
'''
import os
import pandas as pd
from matplotlib import pyplot as plt
from random import seed, randrange
from math import sqrt, exp
import numpy as np
from scipy.stats import chi2_contingency
import time

def plot_train_error(train_error_list):
    plt.plot(train_error_list)
    plt.xlabel('epochs')
    plt.ylabel('cost function value')
    plt.show()
    return

def analyzing_features(df):
    '''

    :param df:
    :return: displays the correlation value between the features. plot the cummulative sum of the feature values to
            assess their trend
    '''
    print df.corr(method='pearson')
    df = df.cumsum()
    plt.figure(0)
    df.plot()
    plt.xlabel('Number of Data Points')
    plt.ylabel('Feature Values')
    plt.show()
    return

def predict(row, coeff):
    yhat = coeff[0]
    for i in range(len(row)-1):
        yhat += coeff[i+1]*row[i]
    return 1.0/(1.0+exp(-yhat))

def metric(actual, predicted):
    correct = 0
    tp=0; fp=0
    tn=0; fn=0
    precision=0.0; recall = 0.0; fscore=0.0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct+=1
        if actual[i] == 1 and predicted[i] == 1:
            tp+=1
        if actual[i] == 0 and predicted[i] == 1:
            fp+=1
        if actual[i] == 1 and predicted[i] == 0:
            fn+=1
        if actual[i] == 0 and predicted[i] == 0:
            tn+=1
    print tp,fp,tn,fn
    precision = tp/float(tp+fp)
    recall = tp/float(tp+fn)
    fscore = 2*((precision*recall)/float(precision+recall))
    return (precision, recall,fscore)
#(correct / float(len(actual)) * 100.0)
def splitCV(dataset, nfolds):
    split_dataset = []
    temp_dataset = list(dataset)
    fold_ratio = int(len(dataset)/nfolds)
    for idx in range(nfolds):
        fld = []
        while len(fld) < fold_ratio:
            indx = randrange(len(temp_dataset))
            fld.append(temp_dataset.pop(indx))
        split_dataset.append(fld)
    return split_dataset

def sgd(traindata,l_rate, epochs):
    train_error_list=[]
    coefs = [0.0 for i in range(len(traindata[0]))]
    for epoch in range(epochs):
        sum_error=0
        for row in traindata:
            yhat = predict(row,coefs)
            error = row[-1]-yhat
            sum_error += error**2
            coefs[0] = coefs[0] + l_rate * error * yhat * (1.0 - yhat)
            for i in range(len(row)-1):
                coefs[i+1] = coefs[i+1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
        train_error_list.append(sum_error)
    return (coefs,train_error_list)

def logisticRegression(trainset, testset, l_rate, nepochs):
    all_predictions = []
    coefficients, train_error_list = sgd(trainset, l_rate, nepochs)
    for row in testset:
        yhat = predict(row, coefficients)
        yhat = float(str(round(yhat, 2)))
        #print yhat
        all_predictions.append(yhat)
    return all_predictions, train_error_list

def model(dataset, n_folds, l_rate, n_epochs):
    folds = splitCV(dataset, n_folds)
    precision_lst=[]
    recall_lst=[]
    fscore_lst=[]
    print n_folds
    for fld in folds:
        train = list(folds)
        train.remove(fld)
        train = sum(train,[])
        test = list()
        for row in fld:
            temp_row = list(row)
            test.append(temp_row)
            temp_row[-1] = None
        predict_result, train_error_list = logisticRegression(train,test,l_rate,n_epochs)
        threshold = 0.2
        predict_result = [1.0 if v > threshold else 0.0 for v in predict_result]
        observed = [row[-1] for row in fld]
        p,r, Fscore = metric(observed,predict_result)
        precision_lst.append(p)
        recall_lst.append(r)
        fscore_lst.append(Fscore)
    plot_train_error(train_error_list)
    exit()
    return (precision_lst, recall_lst, fscore_lst)

def normlize_dataframe(df):
    '''

    :param df:
    :return: normalized dataframe after Min-max feature scaling (normalization)
    '''
    df_norm = (df - df.min()) / (df.max()-df.min())
    return df_norm



if __name__ == '__main__':

    dir_path = os.path.dirname(__file__)
    rel_path = '../Data/fluML.csv'
    full_path = os.path.join(dir_path,rel_path)
    data = pd.read_csv(full_path)
    working_frame  = pd.DataFrame(columns=['Risk', 'Flu'])
    working_frame['Risk'] = data['Risk'].copy(deep=True)
    working_frame['Flu'] = data['Flu'].copy(deep=True)
    #print working_frame.head()
    working_frame = working_frame[np.isfinite(working_frame['Flu'])]
    working_frame = working_frame[np.isfinite(working_frame['Risk'])]
    working_frame = normlize_dataframe(working_frame)
    working_list  = working_frame.values.tolist()
    n_folds = np.arange(2,10,1)
    learning_rate = 0.0001
    n_epochs = 5000
    precision_list=[]
    recall_list =[]
    for i in range(len(n_folds)):
        precision_list, recall_list, fscore_result= model(working_frame.values.tolist(), n_folds[i], learning_rate, n_epochs)
        print "nfolds :", n_folds[i]
        print ('Mean precision', max(precision_list))#sum(precision_list)/float(len(precision_list)))
        print ('Mean recall', recall_list[precision_list.index(max(precision_list))])#sum(recall_list)/float(len(recall_list)))
        print ('Mean f-score ', fscore_result[precision_list.index(max(precision_list))])
        print ('index : ', precision_list.index(max(precision_list)))#sum(fscore_result)/float(len(fscore_result)))
    #analyzing_features(working_frame)