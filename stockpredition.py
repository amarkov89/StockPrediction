#Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

apple = pd.read_csv('C:/Users/Alan/Documents/AAPL.2017.csv')
apple.info()
apple['Date'] = pd.to_datetime(apple['Date'])
print(f'Dataframe contains stock prices between {apple.Date.min()} {apple.Date.max()}')
print(f'Total days = {(apple.Date.max() - apple.Date.min()).days} days')

apple.describe()
apple[['Open', 'High','Close','Adj Close']].plot(kind='box')

layout =go.Layout(
    title='Stock price of Apple',
    xaxis=dict(title='Date',titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f')),
    yaxis=dict(title='Price',titlefont=dict(family='Courier New, monospace', size=18, color='#7f7f7f'))
)

apple_data = [{'x':apple['Date'], 'y':apple['Close']}]
plot = go.Figure(data = apple_data, layout=layout)
plot.show()

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score

X = np.array(apple.index).reshape(-1,1)
Y = apple['Close']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=5)

scaler = StandardScaler().fit(X_train)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, Y_train)

trace0 = go.scatter(
    x = X_train.T[0], y = Y_train, mode ='markers', name = 'Actual'
)

trace1 = go.scatter(
    x= X_train.T[0], y = lm.predict(X_train).T, mode = 'lines', name = 'Predicted'
)

apple_data = [trace0, trace1]
layout.xaxis.title.text = 'Day'
plot2 = go.Figure(data=apple_data, layout=layout)

plot2.show()

scores = f'''
{'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'accuracy_score'.ljust(10)}{accuracy_score(Y_train, lm.predict(X_train))}\t{accuracy_score(Y_test, lm.predict(X_test))}
'''
print(scores)