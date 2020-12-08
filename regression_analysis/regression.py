import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import statistics
import json

def main():
    # kfold = 2
    np.random.seed(r) # set random seed
    ## import data ##
    data = pd.read_csv("Clean_Data/final_data.csv")
    states = data['State'].unique()
    
    possible_features = data.columns[5:]
    possible_labels = ['Mix_Score', 'Depression_Score', 'Anxiety_Score']
    
    final = {}

    for state in states:
        for label in possible_labels:
            for feature in possible_features:

                state_data = data.copy().loc[data['State'] == state].reset_index(drop=True)
                X = state_data.loc[:,feature] # feature columns
                y = state_data.loc[:,label] # label column
                
                # if state_data.shape[0] <= len(X.columns):
                #     continue
                # elif kfold > state_data.shape[0]:
                #     kfold = state_data.shape[0]
                # kf = kfold_validation(X, kfold)
                # test_errors = []
                # for train_index, test_index in kf.split(y):
                # split += 1
                # train_data, test_data = X.loc[train_index], X.loc[test_index]
                # train_label, test_label = y.loc[train_index], y.loc[test_index]

                ## train test split ##
                train_data, test_data, train_label, test_label = train_test_split(
                    X, y, test_size= 0.2)

                ## Regression ##
                regr = linear_model.LinearRegression()
                # regr = linear_model.LogisticRegression()
                train_data_reshaped = train_data.to_numpy().reshape(-1, 1)
                test_data_reshaped = test_data.to_numpy().reshape(-1, 1)

                lm = regr.fit(train_data_reshaped, train_label) # train on training data
                score = regr.score(train_data_reshaped, train_label)
                # y_pred = regr.predict(test_data_reshaped) # test on testing data
                
                if not label in final:
                    final[label] = {}
                if state in final[label]:
                    if score > final[label][state][1]:
                        final[label][state] = [feature, score]
                else:
                    final[label][state] = [feature, score]

                if state_data.shape[0] == 0:
                    continue
    
    with open("final.json", "w") as outfile:  
        json.dump(final, outfile) 

if __name__ == "__main__":
    ## initialize variables ## 
    r = 2010
    main()
