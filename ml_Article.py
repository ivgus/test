import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

data1 = pd.read_excel("/Users/Igoriy1337/Desktop/data.xlsx", sheet_name='Рынок')
data2 = pd.read_excel("/Users/Igoriy1337/Desktop/data.xlsx", sheet_name='КАМАЗ')
bigData = pd.read_excel("/Users/Igoriy1337/Desktop/data.xlsx", sheet_name='Данные')
data1.set_index('дата', inplace = True)
data1.index = pd.to_datetime(data1.index)
data2.set_index('дата', inplace = True)
data2.index = pd.to_datetime(data2.index)
bigData.set_index('дата', inplace = True)
bigData.index = pd.to_datetime(bigData.index)
adf_test1 = adfuller(data1)
print('p-value1 = ' + str(adf_test1[1]))
adf_test2 = adfuller(data2)
print('p-value2 = ' + str(adf_test2[1]))
decompose1 = seasonal_decompose(data1)
decompose1.plot()
plt.show()
plt.clf()
decompose2 = seasonal_decompose(data2)
decompose2.plot()
plt.show()
plt.clf()
data1S = data1[122:137]
s = 0
for i in range(0, 10):
    for j in range(0+i, 5+i):
        s = s + data1S.iat[j, 0]
    data1S.iat[5+i, 0] = float(s)/5
    s = 0
print(data1S)
data2S = data2[105:137]
print(data2S)
ss = 0
for i in range(0, 12):
    for j in range(0+i, 20+i):
        ss = ss + data2S.iat[j, 0]
    data2S.iat[20+i, 0] = float(ss)/20
    ss = 0
print(data2S)
plt.plot(data1, label = 'Рынок', color = 'steelblue')
plt.plot(data2, label = 'КАМАЗ', color = 'black')
plt.plot(data1S, label = 'Скользящее среднее для рынка', color = 'orange')
plt.plot(data2S, label = 'Скользящее среднее для КАМАЗа', color = 'yellow')
plt.legend(title = '', loc = 'upper left')
plt.xlabel('Года')
plt.ylabel('Объём')
plt.show()
err1 = 0
for i in range(0, 10):
    err1 = err1 + abs(float(data1.iat[127+i, 0] - data1S.iat[i, 0])/data1.iat[127+i, 0])
err1 = err1*100/10
print(err1)
err2 = 0
for i in range(0, 12):
    err2 = err2 + abs(float(data2.iat[125+i, 0] - data2S.iat[i, 0])/data2.iat[125+i, 0])
err2 = err2*100/12
print(err2)
plt.clf()
data1E = data1[0:132]
model1 = ExponentialSmoothing(data1E, seasonal_periods=12, trend='add', seasonal='add')
model1_fit = model1.fit()
forecast1 = model1_fit.forecast(5)
forecast1E = forecast1.to_list()
plt.plot(data1, label='оригинальные данные')
plt.plot(model1_fit.fittedvalues, label='сглаживание')
plt.plot(forecast1, label='предсказание')
plt.xlabel('Года')
plt.ylabel('Объём')
plt.title('Тройное экспоненциальное сглаживание по Рынку')
plt.legend()
plt.show()
err1E = 0
for i in range(0, 5):
    err1E = err1E + abs(float(data1.iat[132+i, 0] - float(forecast1E[i]))/data1.iat[132+i, 0])
err1E = err1E*100/5
print(err1E)
plt.clf()
data2E = data2[0:115]
model2 = ExponentialSmoothing(data2E, seasonal_periods=12, trend='add', seasonal='add')
model2_fit = model2.fit()
forecast2 = model2_fit.forecast(22)
forecast2E = forecast2.to_list()
plt.plot(data2, label='оригинальные данные')
plt.plot(model2_fit.fittedvalues, label='сглаживание')
plt.plot(forecast2, label='предсказание')
plt.xlabel('Года')
plt.ylabel('Объём')
plt.title('Тройное экспоненциальное сглаживание по КАМАЗу')
plt.legend()
plt.show()
err2E = 0
for i in range(0, 22):
    err2E = err2E + abs(float(data2.iat[115+i, 0] - float(forecast2E[i]))/data2.iat[115+i, 0])
err2E = err2E*100/22
print(err2E)
plt.clf()
model1S = sm.tsa.statespace.SARIMAX(data1[0:132], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results1 = model1S.fit()
forecast1S = results1.predict(start=132, end=137, dynamic=True)
plt.plot(data1, label='оригинальные данные')
plt.plot(forecast1S, label='предсказание')
plt.xlabel('Года')
plt.ylabel('Объём')
plt.title('SARIMAX по Рынку')
plt.legend()
plt.show()
forecast1S.to_list()
err1S = 0
for i in range(0, 5):
    err1S = err1S + abs(float(data1.iat[132+i, 0] - float(forecast1S[i]))/data1.iat[132+i, 0])
err1S = err1S*100/5
print(err1S)
plt.clf()
model2S = sm.tsa.statespace.SARIMAX(data2[0:125], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results2 = model2S.fit()
forecast2S = results2.predict(start=125, end=137, dynamic=True)
plt.plot(data2, label='оригинальные данные')
plt.plot(forecast2S, label='предсказание')
plt.xlabel('Года')
plt.ylabel('Объём')
plt.title('SARIMAX по КАМАЗу')
plt.legend()
plt.show()
forecast2S.to_list()
err2S = 0
for i in range(0, 12):
    err2S = err2S + abs(float(data2.iat[125+i, 0] - float(forecast2S[i]))/data2.iat[125+i, 0])
err2S = err2S*100/12
print(err2S)
plt.clf()
Y1 = np.array(data1)
Y2 = np.array(data2)
bigDataC = bigData['Рынок (регистрации) г/а 14-40т, шт.']
X = np.array(bigDataC)
plt.scatter(X, Y1)
plt.ylabel('Рынок')
plt.xlabel('Рынок (регистрации) г/а 14-40т, шт.')
plt.title('Корреляция')
slope, intercept = np.polyfit(X, Y1, 1)
plt.plot(X, X * slope + intercept, 'r')
plt.show()
plt.clf()
a = np.round(slope, 2)
b = np.round(intercept, 2)
print(a, b)
bigDataR = bigDataC[0:132]
modelR = ExponentialSmoothing(bigDataR, seasonal_periods=12, trend='add', seasonal='add')
modelR_fit = modelR.fit()
forecastR = modelR_fit.forecast(5)
forecastER = forecastR.to_list()
plt.plot(bigDataC, label='оригинальные данные')
plt.plot(modelR_fit.fittedvalues, label='сглаживание')
plt.plot(forecastR, label='предсказание')
plt.xlabel('Года')
plt.ylabel('Объём')
plt.title('Тройное экспоненциальное сглаживание по Регистрациям на рынке')
plt.legend()
plt.show()
errER = 0
for i in range(0, 5):
    errER = errER + abs(float(bigDataC[132+i] - float(forecastER[i]))/bigDataC[132+i])
errER = errER*100/5
print(errER)
plt.clf()
for i in range(0, 5):
    forecastER[i] = a*forecastER[i]-b

data1R = pd.DataFrame(data=forecastER, index=data1.index[132:137])
data1R.index = pd.to_datetime(data1R.index)
print(data1R)
plt.plot(data1, label='оригинальные данные')
plt.plot(data1R, label='предсказание')
plt.xlabel('Года')
plt.ylabel('Объём')
plt.title('Линейная регрессия по Рынку')
plt.legend()
plt.show()
err1R = 0
for i in range(0, 5):
    err1R = err1R + abs(float(data1.iat[132+i, 0] - float(data1R.iat[i, 0]))/data1.iat[132+i, 0])
err1R = err1R*100/5
print(err1R)
plt.clf()
data22 = pd.read_excel("/Users/Igoriy1337/Desktop/data.xlsx", sheet_name='Лист1')
data22.set_index('дата', inplace = True)
data22.index = pd.to_datetime(data22.index)
data22S = data22[105:144]
print(data22S)
sss = 0
for i in range(0, 19):
    for j in range(0+i, 20+i):
        sss = sss + data22S.iat[j, 0]
    data22S.iat[20+i, 0] = float(sss)/20
    sss = 0
print(data22S)
plt.plot(data2, label = 'КАМАЗ', color = 'black')
plt.plot(data22S, label = 'Скользящее среднее для КАМАЗа', color = 'orange')
plt.legend(title = '', loc = 'upper left')
plt.xlabel('Года')
plt.ylabel('Объём')
plt.show()
plt.clf()
model11S = sm.tsa.statespace.SARIMAX(data1[0:132], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results11 = model11S.fit()
forecast11S = results11.predict(start=132, end=145, dynamic=True)
plt.plot(data1, label='оригинальные данные')
plt.plot(forecast11S, label='предсказание')
plt.xlabel('Года')
plt.ylabel('Объём')
plt.title('SARIMAX по Рынку')
plt.legend()
plt.show()
print(forecast11S)