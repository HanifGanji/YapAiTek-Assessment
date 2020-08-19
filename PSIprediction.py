import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import LinearRegression
from pandas import datetime
from sklearn import metrics
import numpy as np



#Reading the data
df = pd.read_csv(r'C:\Users\mega system\Desktop\YapAiTek Assessment\psi_df_2016_2019.csv')

#Keeping the only columns we need
keep_col = ['timestamp', 'national']
df = df[keep_col]
df.columns = ['Date','PSI']

#Converting our timestamp column to datetime and setting it as index
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d %H").dt.date
df.set_index('Date', inplace=True)

#Uncomment below two lines to take a look at the data
'''df.plot()
plt.show()'''

#Uncomment the below code to see if the data is stationary or not(which is)
'''def adfuller_test(PSI):
    result=adfuller(PSI)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")        
adfuller_test(df['PSI'])'''

#Uncomment the below code to see the ACF and PACF plots
'''acf_plot = plot_acf(df['PSI'], lags=500)
pacf_plot = plot_pacf(df['PSI'], lags=500)'''

#AR model function
def ARmodel(df):
    df['ShiftedPSI'] = df['PSI'].shift(1)
    df.dropna(inplace=True)
    
    y = df.PSI.values
    train_size = int(len(df) * 0.8)   
    train, test = y[0:train_size], y[train_size:]
    
    model = AutoReg(train, lags=1)
    model_fit = model.fit()
    
    preds = model_fit.predict(start=0, end=3000)
    print(model_fit.summary())
    
    plt.plot(test[:], label='actual')
    plt.plot(preds[:], label='predictions')
    plt.legend()
    plt.show()
 
def LRmodel(df):
    #Making our dependent variable column
    df['ShiftedPSI'] = df['PSI'].shift(1)
    df.dropna(inplace=True)
    
    #Splitting our data into X and y , also train and test data and reshaping it
    y = df.PSI.values
    X = df.ShiftedPSI.values
    train_size = int(len(df) * 0.8)
    X_train, X_test = X[0:train_size], X[train_size:]
    y_train, y_test = y[0:train_size], y[train_size:]
    X_train = X_train.reshape(-1,1)
    X_test = X_test.reshape(-1,1)
    
    #Create and fitting our model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    #Getting the predictions
    y_pred = lr.predict(X_test)
    
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse) # or mse**(0.5)  
    r2 = metrics.r2_score(y_test, y_pred)
    
    print("Results of sklearn.metrics:")
    print("MAE:",mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R-Squared:", r2)
    
    #Plot the predictions and the actual data
    plt.plot(y_test[:50], label='actual')
    plt.plot(y_pred[:50], label='predictions')
    plt.legend()
    plt.show()
    
    
LRmodel(df)
ARmodel(df)
