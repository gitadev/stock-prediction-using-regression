import math
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from pandas import Series, DataFrame
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from matplotlib import style
from datetime import datetime, timedelta
import pickle

start = datetime(2010, 1, 1)
end = datetime(2019, 1, 11)
# 2018 2, 15 to 2019, 1, 11 <- Actual Price

df = web.DataReader("NVDA", 'yahoo', start, end)

style.use('ggplot')

df = df[['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']]
df['HL_PCT'] = (df['High'] - df['Low']) / df['Adj Close'] * 100.0
df['PCT_change'] = (df['Adj Close'] - df['Open']) / df['Open'] * 100.0

df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]
forecast_col = 'Adj Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.1 * len(df)))
#
#
df_actual_prices = df.copy()
#df_actual_prices = df_actual_prices[-forecast_out:]
#
#
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label','Adj Close'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

################################
# Linear Regression
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

with open('linearregression.pickle','wb') as f:
    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

confidence = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
df['Linear Regression Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date
next_unix = last_unix + timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += timedelta(days=1)
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]    

df_LRforecast = df.copy()

################################
# Lasso
clf = Lasso()
clf.fit(X_train, y_train)

with open('lasso.pickle','wb') as f:
    pickle.dump(clf, f)

pickle_in = open('lasso.pickle','rb')
clf = pickle.load(pickle_in)

confidence = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
df['Lasso Forecast'] = np.nan

#last_date = df.iloc[-1].name
last_unix = last_date
next_unix = last_unix + timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += timedelta(days=1)
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]    

df_lassoforecast = df.copy()

################################
# Ridge
clf = Ridge()
clf.fit(X_train, y_train)

with open('ridge.pickle','wb') as f:
    pickle.dump(clf, f)

pickle_in = open('ridge.pickle','rb')
clf = pickle.load(pickle_in)

confidence = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
df['Ridge Forecast'] = np.nan

#last_date = df.iloc[-1].name
last_unix = last_date
next_unix = last_unix + timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += timedelta(days=1)
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]    


################################
# Plot

#print(df_LRforecast['Linear Regression Forecast'].tail())
#print(df_lassoforecast['Lasso Forecast'].tail())
#print(df['Ridge Forecast'].tail())

#df['Adj Close'].plot()
#df['Linear Regression Forecast'].plot()
#df['Lasso Forecast'].plot()
#Cludge
df_actual_prices['Adj Close'].plot()
#Cludge
df_LRforecast['Linear Regression Forecast'].plot()
#Cludge
df_lassoforecast['Lasso Forecast'].plot()
df['Ridge Forecast'].plot()
plt.legend(loc=4)
plt.title('NVDA Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
