After reading in the dataset and taking a look at the plotted data , I've noticed that the data is stationary .
To make sure , I've decided to run an adfuller test . As expected , the data was stationary .
Then i decided to take a look at AFC and PAFC charts , by looking at the charts i saw that the AFC chart kinda tails off but the PAFC chart cuts off suddenly after some lags (i tried 1, 2 and 3).
Being so , i decided to use an autoregression model instead of MA or ARIMA models for forecasting the PSI.
Also i used a LinearRegression model too , I shifted the PSI columns by 1 and created a dependent variable column.
I'm not sure about the results , as the Linear model's chart looks fitted much better than the AR one . 
