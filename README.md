# Cointegrated Indices Machine Learning Project

## Project motivation 
In my financial analytics class we briefly covered the concepts of time series and the mathematics behind some of these analyses. We covered topics such as autoarima models, transforming time series to be stationary, and testing for cointegration between time series. However, we never got the chance to put these theories to the test using programs such as python. I decided to take two volatility indeces, EVZ(CBOE Eurocurrency Volatility Index) and VXEEM(CBOE Emerging Mkts ETF Volatility Index), and utilize the concepts I learned in class to find if these two indeces are cointegrated or not and use an Auto ARIMA model to predict the indeces future values. 

## How to run the program on your device
1. Download `coint_indeces.py` file 
2. Download the two .csv files provided or download your desired time series'
3. Install required packages listed under the packages header
4. Run program
5. When prompted to select a file, select your first time series and then select the second time series when prompted a second time

## Required Packages
- numpy
- matplotlib
- pandas
- datetime
- statistics
- statsmodels
- pmdarima
- easygui


## What's does this project do?
- Visualize index values
- Calculate and visualize rolling correlations between indeces
- Transforms indeces to be statinoary using differencing 
- Split indeces in half and perform a two sample t test and an F test to evaluate stationarity 
- Use ADF test to check that indeces are stationary 
- Calculate auto correlations and plot to determine paramters for autoarima model
- Plot predictions made by autoarima model against actual values from testing set 
- Conclude whether the indeces are correlated using an augmented Engle-Granger two-step cointegration test

## Why is this useful?
This program can be helpful for forecasting any time series data. Specifically for stocks and indeces it can be used to identify and develop pairs trading strategies and visualize results. When testing EVZ and VXEEM I concluded that EVZ and VXEEM are in fact cointegrated meaning they have a mean reverting relationship that can be statistically exploited for pairs trading. Because these series are correlated in the long term if they ever diverge from this relationship it would give us a trading opportunity to short one and long the other. 

## Future imporvements
- Create an automation where if the series diverge from their long term relationship then a trade will be executed on the two stocks/indices
- Implement a statement that selects the method data transformation of the user's time series' that minimizes the p-value of the ADF test 
- Tune the parameters of the autoarima model to minimize the residuals
- Evaluate the normality of the stationarity of the residuals for the model and the autocorrelation between residuals
