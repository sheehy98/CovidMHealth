import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import os
import matplotlib.pyplot as plt

def main():
    np.random.seed(r) # set random seed
    ## import data ##
    data = pd.read_csv("clean_data/final_data.csv")
    # TODO, adjust this to fit the actual final data csv
    X = data.iloc[:,:] # feature columns
    y = data.iloc[:,:] # label column

    ## train test split ##
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size= 0.2)

    ## Regression ##
    regr = linear_model.LinearRegression()
    # regr = linear_model.LogisticRegression()
    regr.fit(X_train, y_train) # train on training data
    y_pred = regr.predict(X_test) # test on testing data

    ## Analyze Results ##
    print('Coefficients: \n', regr.coef_) # The coefficients
    print('Mean squared error: %.2f'
        % mean_squared_error(y_test, y_pred)) # The mean squared error
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
        % r2_score(y_test, y_pred))

    ## Plot outputs ##
    # plt.scatter(X_test, y_test,  color='black')
    # plt.plot(X_test, y_pred, color='blue', linewidth=3)
    # plt.xticks(())
    # plt.yticks(())
    # plt.savefig("regression_analysis/charts/regression.png")
    # plt.show()

if __name__ == "__main__":
    ## initialize variables ## 
    r = 2010

    main()
