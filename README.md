# Cointegrated Indices Machine Learning Project

## Project motivation 
In my financial statistics classes we briefly covered the concepts of time series and the mathematics behind some of these analyses. We covered topics such as autoarima models, transforming time series to be stationary, and testing for cointegration between time series. However, we never got the chance to put these theories to the test using programs such as python. I decided to take two volatility indeces, EVZ(CBOE Eurocurrency Volatility Index) and VXEEM(CBOE Emerging Mkts ETF Volatility Index), and utilize the concepts I learned in class to find if these two indeces are cointegrated or not and use an Auto ARIMA model to predict the indeces future values. 

## What's this project do?
The goal of the following project was to determine if the volatility indices EVZ and VXEEM are cointegrated. Cointegration refers to a correlation between time series' in the long term. This project takes the two time series defined by the user and transforms the data so that they are stationary. 

## Why is this useful?


## How to run this project on your device
I supplied the csv files I used for this project within this repository. When running the program the user will be prompted twice to select two different time series. You can select the two files I've provided or could select your own time series data to test. If you select your own data it must be formatted the same way as the data I provided: csv file, Dates in mdy format in the first column, and time series values in the second column. If you're interested in downloading other stock data in this format I downloaded this data from the FRED database. 
