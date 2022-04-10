import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
import datetime as dr
import statistics as st
from scipy.stats import ttest_ind
from scipy.stats import f
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.formula.api import ols
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
import easygui
from easygui import *

###Note to user: throughout this code we will be referring to the CBOE EuroCurrency ETF Volatility Index as its ticker, EVZ. We will be referring to...
# the CBOE Emerging Markets ETF Volatility Index as VXEEM. 

#select the csv for EVZ for the first fileopenbox and select the csv for VXEEM for the second fileopenbox
EVZ = easygui.fileopenbox()
VXEEM= easygui.fileopenbox()
EVZ_df = pd.read_csv(EVZ, parse_dates=True,index_col=0,na_values=[".",""])
VXEEM_df = pd.read_csv(VXEEM, parse_dates=True,index_col=0,na_values=[".",""])


#merge on date
df = pd.merge(EVZ_df,VXEEM_df,how="inner",on="DATE")
print(EVZ_df.head)
print(VXEEM_df.head)
print("EVZ number of observations:", len(EVZ_df))
print("VXEEM number of observations:", len(VXEEM_df))

#rename columns to be more user friendly
df=df.rename(columns={"EVZCLS":"EVZ","VXEEMCLS":"VXEEM"})
print("Number of observations for both series:", len(df))
#drop NA values
df=df.dropna()
print("Number of non-missing observations for both series:", len(df))

##plot EVZ and VXEEM values on a line chart with two y axes

fig, ax1 = plt.subplots()

color1 = "green"
ax1.set_xlabel("Date (Year)")
ax1.set_ylabel("EVZ (%)",color=color1)
ax1.set_title("Value of EVZ and VXEEM")
ax1.plot(df.EVZ,data=df,color=color1,label="EVZ")
ax1.tick_params(axis="y",labelcolor=color1)
plt.gca().legend(("EVZ"),frameon=False)

ax2 = ax1.twinx()

##create a second axis and color for VXEEM chart
color2 = "magenta"
ax2.set_ylabel("VXEEM (%)",color=color2)
ax2.set_title("Value of EVZ and VXEEM")
ax2.plot(df.VXEEM,data=df,color=color2,label="VXEEM",alpha=.5)
ax2.tick_params(axis="y",labelcolor=color2)


##control size of labels so they don't hang off page
h1, I1 = ax1.get_legend_handles_labels()
h2, I2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2,I1+I2,loc=2,frameon=False)

fig.tight_layout()
plt.show()


#calculate rolling correlations between EVZ and VXEEM
s1 = pd.Series(df.EVZ)
s2 = pd.Series(df.VXEEM)
s3 = s1.rolling(21).corr(s2)
print(s3)

#drop any na values for rolling correlations
s3 = s3.dropna()
print(s3)

#plot rolling correlations
plt.plot(s3)
plt.title("Rolling correlations for EVZ and VXEEM")
plt.xlabel("Date (Year)")
plt.ylabel("Correlation")
plt.show()

#descriptive stats for rolling correlations
print("highest rolling correlation:",max(s3))
print("Lowest rolling correaltion:", min(s3))
print("Average rolling correaltion:", s3.mean())

#split sample in half to test if mean and standard deviation are significantly different when comparing the two samples
cutoff = round(len(df)/2)
print("Division",cutoff)

df1,df2 = df[0:cutoff],df[cutoff:]
print("df1 statistics",df1.describe())
print("df2 statistics ",df2.describe())

dfs = df.describe()
dfs1 = df1.describe()
dfs2 = df2.describe()

#output descriptive stats into a csv file
dfs.to_csv("stationary_descriptiveStatistics.csv")
dfs1.to_csv("stationary_descriptiveStatistics.csv", mode="a")
dfs2.to_csv("stationary_descriptiveStatistics.csv", mode="a")

#perform two sample t test on EVZ where null is there is no difference between the two means
t1 = ttest_ind(df1.EVZ,df2.EVZ)
print("t-statistic (EVZ) = ",round(t1[0],4)," p-value = ",round(t1[1],4))
#found significant difference in means between two samples

num = max(st.variance(df1.EVZ),st.variance(df2.EVZ))
denom= min(st.variance(df1.EVZ),st.variance(df2.EVZ))
fstat = num/denom
pf = 1 - f.cdf(fstat,len(df1.EVZ),len(df2.EVZ))
print("F (EVZ) = ",round(fstat,4), " p-value = ", round(pf,4))
#significant difference in variances

#perform two sample t test on VXEEM where null is there is no difference between the two means
t2 = ttest_ind(df1.VXEEM,df2.VXEEM)
print("t-statistic (VXEEM) = ",round(t1[0],4)," p-value = ",round(t1[1],4))
#found significant difference in means between two samples

num = max(st.variance(df1.VXEEM),st.variance(df2.VXEEM))
denom= min(st.variance(df1.VXEEM),st.variance(df2.VXEEM))
fstat = num/denom
pf = 1 - f.cdf(fstat,len(df1.VXEEM),len(df2.VXEEM))
print("F (VXEEM) = ",round(fstat,4), " p-value = ", round(pf,4))
#F-test on VXEEM found that there is no significant difference in the variances from our split up samples

#Test for Stationarity - ADF Test - Null hypothesis is that the series is a unit root
EVZ_ADF = adfuller(df.EVZ,autolag="BIC")
print("Untransformed ADF test statistic for EVZ = ",round(EVZ_ADF[0],4)," p-value =",round(EVZ_ADF[1],4))

VXEEM_ADF = adfuller(df.VXEEM,autolag="BIC")
print("Untransformed ADF test statistic for VXEEM = ",round(VXEEM_ADF[0],4)," p-value =",round(VXEEM_ADF[1],4))

#Decomposition - use multiplicative because trends are non-linear

firstSeries = df.EVZ
result = seasonal_decompose(firstSeries,model="multiplicative", period=1)
print(result.trend)
print(result.seasonal)
print(result.resid)
result.plot()
plt.show()

secondSeries = df.VXEEM
result = seasonal_decompose(secondSeries,model="multiplicative", period=1) #can play around with it and do multiplicative model too (must add ,periods=1)
print(result.trend)
print(result.seasonal)
print(result.resid)
result.plot()
plt.show()

#create lags of EVZ and VXEEM and a variable for first differenced EVZ and VXEEM
df['lagEVZ']=df['EVZ'].shift(1)
df['lagVXEEM']=df["VXEEM"].shift(1)
df['EVZDiff']=df['EVZ']-df['lagEVZ']
df['VXEEMDiff']=df['VXEEM']-df['lagVXEEM']

#Transform Data - No need to transform the data in this case but I was...
#curious to see how much it would improve the test statistic for the ADF test

## use differencing to detrend the series resulting in a stationary series
modelg = ols("EVZ ~ lagEVZ",data=df).fit()
gres = modelg.resid
gresADF = adfuller(gres,autolag="BIC")
print("First difference transformed ADF test statistic for EVZ =", round(gresADF[0],4),
      "p-value = ", round(gresADF[1],4))

modelg = ols("VXEEM ~ lagVXEEM",data=df).fit()
gres2 = modelg.resid
gres2ADF = adfuller(gres2,autolag="BIC")
print("First difference transformed ADF test statistic for VXEEM =", round(gres2ADF[0],4),
      "p-value = ", round(gres2ADF[1],4))

#Auto Correlations
plot_acf(df.EVZ)
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation by lag for EVZ")
plt.show()
plot_pacf(df.EVZ)
plt.xlabel("Lag")
plt.ylabel("Partial Autocorrelation")
plt.title("Partial Autocorrelation by lag for EVZ")
plt.show()

plot_acf(df.VXEEM)
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation by lag for VXEEM")
plt.show()
plot_pacf(df.VXEEM)
plt.xlabel("Lag")
plt.ylabel("Partial Autocorrelation")
plt.title("Partial Autocorrelation by lag for VXEEM")
plt.show()

#Autoarima - DONT FORGET TO REPEAT FOR SERIES 2
cutoffArima = round(len(df)/3)

train = df.EVZ[:cutoffArima*2]
test = df.EVZ[cutoffArima*2:]

plt.plot(train)
plt.plot(test)
plt.show()


arima_model = auto_arima(train,start_p=0,d=0,start_q=0,max_p=10,max_d=20,max_q=10,
                         seasonal=False,stepwise = True)

print(arima_model.summary())
                         

num=len(test)
prediction = pd.DataFrame(arima_model.predict(n_periods=num),index=test.index)
prediction.columns = ["Predicted EVZ"]
print(prediction)

plt.figure(figsize = (8,5))
plt.plot(train,label="Training Data")
plt.plot(test, label="Validation Data")
plt.plot(prediction, label ="Predictions")
plt.ylabel("EVZ (%)")
plt.legend()
plt.show()

train2 = df.VXEEM[:cutoffArima*2]
test2 = df.VXEEM[cutoffArima*2:]

plt.plot(train2)
plt.plot(test2)
plt.show()


arima_model2 = auto_arima(train2,start_p=0,d=1,start_q=0,max_p=10,max_d=20,max_q=10,
                         seasonal=False, stepwise=True)

print(arima_model2.summary())
                         

num2=len(test2)
prediction2 = pd.DataFrame(arima_model2.predict(n_periods=num2),index=test.index)
prediction2.columns = ["Predicted Series 2"]
print(prediction2)

plt.figure(figsize = (8,5))
plt.plot(train2,label="Training Data")
plt.plot(test2, label="Validation Data")
plt.plot(prediction2, label ="Predictions")
plt.ylabel("VXEEM (%)")
plt.legend()
plt.show()

#co-integration
a = df.EVZ
b = df.VXEEM
results = coint(a,b)
print("coint t-statistic:",results[0],"coint p-value:", results[1],"critical values for t-statistic:",results[2])

##reject null that series are not cointegrated
