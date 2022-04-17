import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import yfinance as yf
import math
import os.path
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from math import sqrt


# BITCOIN #


cryptocurrencies = ['BTC-USD']
bitcoin_df = yf.download(cryptocurrencies, start='2019-01-01',
                end='2022-03-01', parse_dates = ['Month'])

bitcoin_df["Volume"] = bitcoin_df["Volume"].astype("float")
bitcoin_df.info()

bitcoin_df = bitcoin_df.drop(columns = ['Adj Close'])

bitcoin_features = bitcoin_df
features = ['Open', 'High', 'Low', 'Volume']
X_bitcoin = bitcoin_features[features]
y_bitcoin = bitcoin_features['Close']

X_train, X_test, y_train, y_test = train_test_split(X_bitcoin, y_bitcoin, test_size = 0.30, random_state = 42)


def rmse(x, y): return math.sqrt(((x-y)**2).mean())
def print_score(m):
    res = [rmse(m.predict(X_train), y_train),
           rmse(m.predict(X_test), y_test),
                m.score(X_train, y_train),
                m.score(X_test, y_test)]
    print(f"RMSE Training Set: {res[0]} \nRMSE validation set: {res[1]} \nR2 validation set: {res[3]}")

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

predictions = lm.predict(X_test)
predictions

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))




def reg_accuracy(y_true, y_pre):
    return_var = []
    from math import sqrt
    rmse = sqrt(metrics.mean_squared_error(y_true, y_pre))
    return_var.append(rmse)
    print ("RMSE: ", rmse)
    r2 = metrics.r2_score(y_true, y_pre)
    return_var.append(r2)
    print("R2: ", r2)
    mae = metrics.mean_absolute_error(y_true, y_pre)
    return_var.append(mae)
    print("MAE: " , mae)

    if 0 in y_true:
        print("MAPE erróneo")
        return_var.append(0)

    else:
        mape = round(np.mean(np.abs((y_true - y_pre) / y_true))*100,4)
        print('MAPE :', mape)
        print('======================')
        print('Model Accuracy(%) :', 100 - mape)
        print('======================')
        return_var.append(mape)
        return_var.append(100-mape)
    return return_var


# CARDANO #

cryptocurrencies = ['BTC-USD']
cardano_df = yf.download(cryptocurrencies, start='2019-01-01',
                end='2022-03-01', parse_dates = ['Month'])

cardano_df["Volume"] = cardano_df["Volume"].astype("float")
cardano_df.info()

cardano_df = cardano_df.drop(columns = ['Adj Close'])

cardano_features = cardano_df
features = ['Open', 'High', 'Low', 'Volume']
X_cardano = cardano_features[features]
y_cardano = cardano_features['Close']

X_train, X_test, y_train, y_test = train_test_split(X_cardano, y_cardano, test_size = 0.30, random_state = 42)



from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

predictions = lm.predict(X_test)
predictions

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))



# ETHEREUM #


cryptocurrencies = ['BTC-USD']
ethereum_df = yf.download(cryptocurrencies, start='2019-01-01',
                end='2022-03-01', parse_dates = ['Month'])

ethereum_df["Volume"] = ethereum_df["Volume"].astype("float")
ethereum_df.info()

ethereum_df = ethereum_df.drop(columns = ['Adj Close'])

ethereum_features = ethereum_df
features = ['Open', 'High', 'Low', 'Volume']
X_ethereum = ethereum_features[features]
y_ethereum = ethereum_features['Close']

X_train, X_test, y_train, y_test = train_test_split(X_ethereum, y_ethereum, test_size = 0.30, random_state = 42)


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

predictions = lm.predict(X_test)
predictions

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
