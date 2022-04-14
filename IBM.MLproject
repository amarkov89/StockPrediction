#Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


data = pd.read_csv('C:/Users/Alan/Documents/IBM.2011.2021.csv')
data.info()

X = data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
# y is the target variable
Y = data['Close'].values
#train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=50)

regressor = LinearRegression()
#train the model
regressor.fit(X_train, Y_train)
yprediction = regressor.predict(X_test)
#coeff or intercept numbers
#print(regressor.coef_)
#print(regressor.intercept_)
ValuesFrame = pd.DataFrame({'Actual':Y_test, 'Prediction':yprediction})
print(ValuesFrame.dtypes)
print(ValuesFrame)
#export for excel charting
ValuesFrame.to_csv('PredictionResults.csv')

#metrics
print('R Squared Score: ', r2_score(Y_test, yprediction))
print('Mean Squared Error: ', mean_squared_error(Y_test, yprediction))

#plot
plt.figure(figsize=(16,8))
plt.plot(data['Close'], label='Close Price history')
plt.show()

#plot actual vs. predicted
#plt.scatter(X_train, Y_train, color = 'red')
#plt.plot(X_train, regressor.predict(X_train), color = 'green')
#plt.title('Stock Training Price vs. Stock Prediction')
#plt.xlabel('Date')
#plt.ylabel('Stock Price')
#plt.show()