#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: Importez vos modules ici
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# TODO: Définissez vos fonctions ici
# def train_test(path: str="./data/winequality-white.csv") -> tuple:
#     df = pd.read_csv(path, sep=";", header=0)
#     y = df["quality"]
#     X = df.drop(columns=["quality"])

#     return train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.1)

def train_test():
    df = pd.read_csv('./data/winequality-white.csv', sep= ';')
    x = df.drop(columns='quality').to_numpy()
    y = df['quality'].to_numpy()
    return train_test_split(x, y)

def train_and_eval_model(model, X_train: list, X_test: list, y_train: list, y_test: list) -> list:
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    make_plot(np.arange(len(y_test)), y_test, pred, model.__class__.__name__)

    return pred

def forestReg(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    return prediction

def linearReg(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    return prediction

def make_plot(x, target, prediction, model):
    fig = plt.figure()
    plt.plot(x, target, label="Target values")
    plt.plot(x, prediction, label="Predicted values")
    plt.legend()
    plt.title(f"{model} predictions analysis")
    plt.xlabel("Number of samples")
    plt.ylabel("Quality")

    # fig.savefig(f"./{model_name}.png")
    
    
    # fig = plt.figure()
    # plt.plot(x, y_test, label= 'Target')
    # plt.plot(x, prediction, label= 'Prediction')
    # plt.legend()
    # plt.xlabel('Sample #')
    # plt.ylabel('Quality')

    # plt.title(model)
    plt.show()


if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    x_train, x_test, y_train, y_test = train_test()
    random_forest_pred = forestReg(RandomForestRegressor(), x_train, x_test, y_train, y_test)
    # random_forest_pred = train_and_eval_model(RandomForestRegressor(),  x_train, x_test, y_train, y_test)
    make_plot(np.arange(len(y_test)), y_train, random_forest_pred, "Random Forest prediction")
    linear_regression_pred = linearReg(LinearRegression(), x_train, x_test, y_train, y_test)
    make_plot(np.arange(len(y_test)), y_train, linear_regression_pred, "Linear Regression prediction")
