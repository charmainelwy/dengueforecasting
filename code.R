library(fpp2)
library(tidyverse)
library(lubridate)
library(tseries)
library(TSstudio)
library(vars)
library(lmtest)
library(forecast)
data_df <- read.csv("D:/OneDrive - Singapore Management University/DSA301 TIME SERIES/project/clean/Transformed_data.csv",stringsAsFactors = FALSE) 
dengue1 <- ts(data = data_df$dengue, start = decimal_date(ymd("2012-1-1")), frequency = 365/7)
#Explaination for BoxCox for Dengue is in ARIMA models
lambda1 <- BoxCox.lambda(dengue1)
dengue2 <- BoxCox(dengue1, lambda1)
dengue3 <- diff(dengue2)

#For S.ArimaX
dengue4 <-(seasonal(mstl(dengue2)))

#Rainfall and Temperature variable
#Rainfall
rainfall1 <- ts(data=data_df$weekly_rainfall,start = decimal_date(ymd("2012-1-1")), frequency = 365/7 )
autoplot(rainfall1)
tsdisplay(rainfall1)
lambda2 <- BoxCox.lambda(rainfall1)
rainfall2 <- BoxCox(rainfall1, lambda2)
autoplot(rainfall2)
tsdisplay(rainfall2)
ndiffs(rainfall2)
nsdiffs(rainfall2)
#Boxcox transformation made it stationary, zero differencing needed
kpss.test(rainfall2)

#Temperature
temp1 <- ts(data=data_df$weekly_temp,start = decimal_date(ymd("2012-1-1")), frequency = 365/7 )
autoplot(temp1)
lambda3 <- BoxCox.lambda(temp1)
temp2 <- BoxCox(temp1, lambda3)
autoplot(temp2)
#Not much difference in autoplot
tsdisplay(temp1)
tsdisplay(temp2)
#No difference in tsdisplay
ndiffs(temp1)
ndiffs(temp2)
nsdiffs(temp1)
nsdiffs(temp2)
#No differencing needed for temp1 and temp2. Hence take temp1,without BoxCox

#Training and Testing
outofsampleperiod = 105 #So that our training data has the exact same range as the original S-ARIMA.
dengue2_split = ts_split(dengue2, sample.out = outofsampleperiod)
rainfall2_split = ts_split(rainfall2, sample.out = outofsampleperiod)
temp2_split = ts_split(temp2, sample.out = outofsampleperiod)
temp1_split = ts_split(temp1, sample.out = outofsampleperiod)
dengue4_split = ts_split(dengue4, sample.out = outofsampleperiod)

# feed forward NN model 
# ndiffs=1
# nsdiffs=0, no LT trend on yearly basis
# autoplot(forecast(nnetar(dengue2_split$train, p = 12, P = 1, period = 12), h = 120))

# with external reg
nn_model = nnetar(dengue2_split$train, p = 12, P = 2, period = 12, xreg = cbind(rainfall2_split$train,temp2_split$train))
forecasted_nn = forecast(nn_model, PI=TRUE, h = 120, bootstrap=TRUE, npaths=100, xreg = cbind(rainfall2_split$train,temp2_split$train))
# rmse(train) = 0.03091231  , rmse(test)=0.24605914  

accuracy(forecasted_nn, dengue2_split$test)

resid = nn_model$residuals
pvalue1= Box.test(resid[105:length(resid)], fitdf = 14, type=c("Ljung-Box"), lag=60)
pvalue1
acf(resid[105:length(resid)])




# without external reg 
nn_model2 = nnetar(dengue2_split$train, p = 12, P = 3, period = 12)
forecasted_nn2 = forecast(nn_model2, PI=TRUE, h = 120, bootstrap=TRUE, npaths=100) 
# 0.02551873   0.28070039  
      
accuracy(forecasted_nn2, dengue2_split$test)

resid2 = nn_model$residuals
pvalue2= Box.test(resid2[105:length(resid2)], fitdf = 15, type=c("Ljung-Box"), lag=60)
pvalue2
acf(resid2[105:length(resid2)])


autoplot(forecasted_nn2) + autolayer(dengue2_split$test)
    
         
# with xreg, without xreg 
# P=1, 0.0343672 0.3908101 , 0.04197035 0.38029016 
# P=2, 0.02810417 0.22599061 , 0.03506159 0.34941198 --> BEST
# P=3, 0.02566569 0.30489360 , 0.02840284  0.25411577    
# P=4, 0.009270011 0.669463027 , 0.02007329 0.40035862 
# P=5, 0.005787487 0.442834531 , 0.008219474 0.472776302 
# P=6, 0.001021926 0.406976683 , 0.00143251 0.57169523 
