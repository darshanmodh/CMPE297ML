import pandas as pd
import quandl
import math, datetime
import time
import numpy as np
from pandas.tools.plotting import scatter_matrix
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from matplotlib import style
import matplotlib.pyplot as plt

df = quandl.get('GOOG/NASDAQ_GOOGL')

print(df.tail())

df ['OC_CHANGE'] = (df['Close'] - df['Open']) / df['Open'] * 100
df ['HL_CHANGE'] = (df['High'] - df['Low']) / df['Low'] * 100
df = df[['Close', 'HL_CHANGE', 'OC_CHANGE', 'Volume']]

forecast_col = 'Close'
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
print(df)
df.dropna(inplace=True)

X = np.array(df.drop(['label'],1))
y = np.array(df['label'])

print(X)
print(y)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# Linear Regression
clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("Linear Regression: %s " % accuracy)

# SVR
# clr = SVR()
# clr.fit(X_train, y_train)
# accuracy_svr = clr.score(X_test, y_test)
# print("SVR : %s" % accuracy_svr)

X = X[:-forecast_out]
X_lately = X[-forecast_out:]
Forecast_set = clf.predict(X_lately)
print(Forecast_set)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = time.mktime(last_date.timetuple())
one_day = 86400
next_unix = last_unix + one_day

for i in Forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Close'].plot()

df['Forecast'].plot()

plt.legend(loc=4)

plt.xlabel('Date')

plt.ylabel('Price')

plt.show()