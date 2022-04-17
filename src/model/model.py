import numpy as np
import pickle
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# Set random seed
np.random.seed(42)

from sklearn import metrics, tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV


X = bitcoin_df[['Open', 'High', 'Low', 'Volume']]

y = bitcoin_df['Close']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42)

lr = LinearRegression(n_jobs=-1) 
lr.fit(X_train, y_train)

predictions = lm.predict(X_test)
predictions
lr.fit(X, y)


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

modelo_defecto  = RandomForestRegressor(random_state = 42)
modelo_defecto.fit(X_train, y_train)

print_score(modelo_defecto)

rf = RandomForestRegressor(random_state=42)
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, scoring = 'neg_mean_squared_error',
                               cv= 3, verbose = 2, random_state = 42, n_jobs =-1,
                               return_train_score = False)

rf_random.fit(X_train, y_train)

my_model = grid_search.best_params_ 
my_model