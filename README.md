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


## What's this project do?
The goal of the following project was to predict the future values of EVZ and VXEEM and determine if these indeces are cointegrated. The program takes the two time series defined by the user and visualizes the rolling correlations between the two series before transforming the data to stationary time series. To ensure that the time series are in fact stationary I used an ADF test. Now that the time series are stationary we can split our data into a training and test and fit an autoarima model to the training data. 

## Why is this useful?
