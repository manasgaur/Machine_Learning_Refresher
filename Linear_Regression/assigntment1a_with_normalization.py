import os
import pandas as pd
from matplotlib import pyplot as plt
from random import seed, randrange
from math import sqrt
import numpy as np
import time

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

def learning_error_plot(l_error):
    '''

    :param l_error:
    :return: training error curve of the model
    '''
    plt.figure(1)
    plt.plot(np.log(l_error))
    plt.xlabel('epochs')
    plt.ylabel('training error of the model')
    plt.show()
    return

def prediction_error_plot(learning_rate, model_score_lst):
    '''

    :param learning_rate:
    :param model_score_lst:
    :return: prediction error plot, learning rate v/s model prediction error value
    '''
    plt.figure(2)
    plt.plot(learning_rate, model_score_lst)
    plt.xlabel('varying learning rate')
    plt.ylabel('prediction error of the model on the test data')
    plt.show()
    return


def regression_line_plot(feat,observed, ypred):
    '''
    params : feat (features value list)
    params : observed (observed risk : Risk)
    params : ypred ( predicted risk)
    result : regression line plot
    '''
    plt.figure(3)
    plt.plot(observed,color='m', marker="o")
    plt.plot(ypred,color="k")
    plt.xlabel('Data points')
    plt.ylabel('Observed Risk (Scatter), Predicted Risk regression line')
    plt.show()
    return

def RMSE(observed, predicted):
    '''

    :param observed:
    :param predicted:
    :return: returns RMSE and Averaged RMSE

    '''
    total_error = 0.0
    error_list=[]
    for idx in range(len(observed)):
        predn_error = predicted[idx] - observed[idx]
        total_error  = total_error + (predn_error ** 2)
        error_list.append(total_error)
       # print error_list
    return (float(sqrt(float(total_error)/len(observed))),error_list)


def predict(coeff, row):
    '''

    :param coeff:
    :param row:
    :return: the predicted risk of the model is returned (ypred)

    '''
    ypred = coeff[0]
    for idx in range(len(row)-1):
        ypred = ypred + coeff[idx+1]*row[idx]
    return ypred

def splitting_for_cross_validation(data, nfolds):
    '''

    :param data:
    :param nfolds:
    :return: this function splits the data in to training and testing dataset.
    '''
    data_copy = list(data)
    splitted_data = []
    folds_length = int(len(data)/nfolds)
    for i in range(nfolds):
        each_fold = list()
        while len(each_fold) < folds_length:
            indx = randrange(len(data_copy))
            each_fold.append(data_copy.pop(indx))
        splitted_data.append(each_fold)
    return splitted_data

def coefficients_by_sgd(traindata, learning_rate, numepochs):
    '''

    :param traindata:
    :param learning_rate:
    :param numepochs:
    :return: coefficients of stochastic gradient descent (the theta values). These theta values are during the learning
             phase and hence are changing over the epochs
             error_record : this variable  records the changing error while the model is training. It should decrease or
             increase but should converge to a point (flattening of the learning curve) .
    '''
    coefficients = [0.0 for idx in range(len(traindata[0]))]
    error_record=[]
    for epoch in range(numepochs):
        error=[]
        for row in traindata:
            ypred = predict(coefficients,row)
            error = ypred - row[-1]
            coefficients[0] = coefficients[0] - learning_rate * error
            for i in range(1, len(row)-1):
                coefficients[i] = coefficients[i]- learning_rate*error*row[i]
        error_record.append(error)
    return (coefficients, error_record)

def linear_regression(traindata, testdata, learning_rate, numepochs):
    '''

    :param traindata:
    :param testdata:
    :param learning_rate:
    :param numepochs:
    :return:  all the prediction made over the test data ( all_predictions : list)
            coefficients : these are the theta values for one fold ratio. Suppose I trained the model over two folds
                            [8:2 and 9:1], then there will be coefficients generated for both.
            error_recording : training error recording. this is just done to link the functions to the main, so as to generate
            the training error curve as the outcome of the program
    '''
    all_predictions = []
    coefficients, error_recording = coefficients_by_sgd(traindata, learning_rate, numepochs)
    for row in testdata:
        ypred = predict(coefficients,row)
        all_predictions.append(ypred)
    return (all_predictions, coefficients, error_recording)

def LR_evaluation(data,nfolds,learning_rate, numepochs):
    '''

    :param data:
    :param nfolds:
    :param learning_rate:
    :param numepochs:
    :return: The functions returns :
             Root means square error,
             predicted risk (defined by the variable : final_predict_result)
             coefficients (the theta values of the features),
             residuals (the error difference or deviation),
             learning_error ( this is the training error of the model when it is learning)
    '''
    total_folds = splitting_for_cross_validation(data,nfolds=nfolds)
    result = []
    final_predict_result=[]
    coeffs=[]
    residuals=[]
    learning_error=[]
    for each_fold in total_folds:
        train = list(total_folds)
        train.remove(each_fold)
        test = list()
        train = sum(train,[])
        for row in each_fold:
            temp_row = list(row)
            test.append(temp_row)
            temp_row[-1] = None
        predict_result, coeffs, learning_error = linear_regression(train, test, learning_rate, numepochs)
        observed  = [row[-1] for row in each_fold]
        RMSE_score, error_lst = RMSE(observed, predict_result)
        final_predict_result.extend(predict_result)
        residuals.extend(error_lst)
        result.append(RMSE_score)
    return (result,final_predict_result, coeffs,residuals, learning_error)

def covariance_between_feature_target(df_list):
    '''

    :param df_list:
    :return: calculate the covariance value between the feature and target variable
    '''
    cov = 0.0
    feature=df_list['KnowlTrans'].tolist()
    mean_feature = np.mean(feature)
    target=df_list['Risk'].tolist()
    mean_target = np.mean(target)
    for idx in range(len(feature)):
        cov += (feature[idx]-mean_feature)*(target[idx]-mean_target)
    return cov

''' Normalization is done using Min-Max feature scaling'''
def MinMax_calculation(df_list):
    '''

    :param df_list:
    :return: generate a min-max list for min-max scaling of the data set (normalization)
    '''
    min_max = []
    for idx in range(len(df_list[0])):
        column_val = [row_val[idx] for row_val in df_list]
        min_val = min(column_val)
        max_val = max(column_val)
        min_max.append([min_val, max_val])
    return min_max

def normalization(df_list):
    min_max = MinMax_calculation(df_list)
    for each_row in df_list:
        for idx in range(len(each_row)):
            each_row[idx] = (each_row[idx]-min_max[idx][0]) / (min_max[idx][1]-min_max[idx][0])



if __name__ == '__main__':

    dir_path = os.path.dirname(__file__)
    rel_path = 'Machine_learning/fluML.csv'
    full_path = os.path.join(dir_path,rel_path)
    data = pd.read_csv(full_path)

    print "Number of observation", data.shape[0], "and", "Number of features", data.shape[1]
    missing_values  = data.isnull().values.ravel().sum()
    print "percentage of missing values", float(missing_values)/data.shape[0]*100
    '''
    KnowlTrans = Knowledge about how the flu is transmitted (score in logits)
    Risk = Perceived risk of contracting influenza (score in logits)
    '''
    working_frame = pd.DataFrame(columns=['KnowlTrans', 'Risk'])
    working_frame['KnowlTrans'] = data['KnowlTrans'].copy(deep=True)
    working_frame['Risk'] = data['Risk'].copy(deep=True)

    # you can see the cummulative plot of the data by un-commenting subsequent statement
    analyzing_features(working_frame)

    # 1. Approach to removing NaN  --> converting them to 0 and removing all the observations (row) with value 0 either under KnowlTrans or Risk
    working_frame = working_frame.fillna(0)
    working_frame = working_frame[(working_frame.T !=0).any()]
    working_list  = working_frame.values.tolist()

    print "Covariance score between KnowlTrans and Risk", covariance_between_feature_target(working_frame)
    print "performing the normalization of the dataset"
    normalization(working_list)

    # parameters; Need to create a function for parameter tuning as well
    # defining the grid
    nfolds = np.arange(2,10,1)
    learning_rate = np.arange(0.001,0.009, 0.001)
    numepochs = np.arange(2000,10000, 1000)
    model_score_lst=[]
    learning_error=[]
    final_predict_result=[]
    model_score_for_each_learning_rate=[]

    # training and prediction cycle starts
    # you can remove this outer loop if you are manually passing the fold value. As a part of experiment, I have created this outer loop
    for k in range(len(nfolds)):
        print nfolds[k]
        # this list is made empty purposely so that you can check on which fold the model performed well
        model_score_for_each_learning_rate=[]
        for i in range(len(learning_rate)):
            model_score_lst=[]
            for j in range(len(numepochs)):
                model_score, final_predict_result, coeffs, residuals, learning_error = LR_evaluation(working_list,nfolds[k],learning_rate[i],numepochs[j])
                avg_model_score = np.sum(model_score)/float(len(model_score))
                print avg_model_score
                model_score_lst.append(np.sum(model_score)/float(len(model_score)))
            model_score_for_each_learning_rate.append(np.sum(model_score_lst)/float(len(model_score_lst)))

    print "Note : Model learning and Prediction is done on Normalized Dataset"
    learning_error_plot(learning_error)
    time.sleep(1)
    prediction_error_plot(learning_rate, model_score_for_each_learning_rate)
    time.sleep(1)
    regression_line_plot(working_frame['KnowlTrans'].tolist(),working_frame['Risk'].tolist(),final_predict_result)
