from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter import simpledialog
from sklearn.model_selection import train_test_split
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import os
import json
from web3 import Web3, HTTPProvider

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

main = tkinter.Tk()
main.title("Bitcoin Price Prediction") #designing main screen
main.geometry("1000x650")

global X,Y
global X_train, X_test, Y_train, Y_test
global dataset, sc

def saveToBlockchain(data):
    blockchain_address = 'http://127.0.0.1:9545'
    web3 = Web3(HTTPProvider(blockchain_address))
    web3.eth.defaultAccount = web3.eth.accounts[0]
    compiled_contract_path = 'Bitcoin.json' #bitcoin contract file
    deployed_contract_address = '0xcBC173296081009924b62748bBd9b7eE89921149' #contract address
    with open(compiled_contract_path) as file:
        contract_json = json.load(file)  # load contract info as JSON
        contract_abi = contract_json['abi']  # fetch contract's abi - necessary to call its functions
    file.close()
    contract = web3.eth.contract(address=deployed_contract_address, abi=contract_abi)
    msg = contract.functions.setData(data).transact()
    tx_receipt = web3.eth.waitForTransactionReceipt(msg)
    
def readDetails():
    details = ""
    blockchain_address = 'http://127.0.0.1:9545' #Blokchain connection IP
    web3 = Web3(HTTPProvider(blockchain_address))
    web3.eth.defaultAccount = web3.eth.accounts[0]
    compiled_contract_path = 'Bitcoin.json' #Bitcoin contract code
    deployed_contract_address = '0xf89127DEC6b57862888E1605FD4d9bdBb3478D3D' #hash address to access student contract
    with open(compiled_contract_path) as file:
        contract_json = json.load(file)  # load contract info as JSON
        contract_abi = contract_json['abi']  # fetch contract's abi - necessary to call its functions
    file.close()
    contract = web3.eth.contract(address=deployed_contract_address, abi=contract_abi) #now calling contract to access data
    details = contract.functions.getData().call()
    return details   
   

def SPCE(high_dimension_data):
    qq = high_dimension_data
    X = qq
    X_scaled = StandardScaler().fit_transform(X)
    features = X_scaled.T
    solution = np.cov(features)
    values, vectors = np.linalg.eig(solution)
    fitness = []
    for i in range(len(values)):
        fitness.append(round(values[i] / np.sum(values),8))
    update_position = np.sort(fitness)[::-1]
    selected = np.zeros(len(fitness))
    for i in range(0,4):
        for j in range(len(fitness)):
            if update_position[i] == fitness[j]:
                selected[j] = 1
    X_selected_features = X[:,selected==1]
    X_scaled = StandardScaler().fit_transform(X_selected_features)
    return X_selected_features


def loadDataset():
  text.delete('1.0', END)
  global dataset
  data = readDetails()
  f = open("temp.csv", "w")
  f.write(data)
  f.close()

  dataset = pd.read_csv("Dataset/Bitcoin-USD.csv")
  text.insert(END,str(dataset.head()))

def preprocess():
  text.delete('1.0', END)
  global X,Y
  global X_train, X_test, Y_train, Y_test
  global dataset

  Y = dataset.values[:,4:5]
  dataset.drop(['Close'], axis = 1,inplace=True)
  dataset = dataset.values
  X = dataset[:,1:dataset.shape[1]-1]
  text.insert(END,"Dataset preprocessing completed\n\n")
  text.insert(END,str(X)+"\n\n")
  text.insert(END,"Total records found in dataset: "+str(X.shape[0])+"\n\n")
  text.insert(END,"Total features found in dataset before applying SPCE: "+str(dataset.shape[1]))

def runSPCE():
  text.delete('1.0', END)
  global X,Y,sc
  global X_train, X_test, Y_train, Y_test
  X = SPCE(X)
  text.insert(END,"Total features found in dataset after applying SPCE: "+str(X.shape[1])+"\n\n")
  sc = MinMaxScaler(feature_range = (0, 1))
  X = sc.fit_transform(X)
  Y = sc.fit_transform(Y)

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
  text.insert(END,"Dataset train & test split\n\n")
  text.insert(END,"80% records used to train algorithms: "+str(X_train.shape[0])+"\n\n")
  text.insert(END,"20% records used to test algorithms: "+str(X_test.shape[0])+"\n\n")
  

def runSVM():
  global X,Y,sc
  global X_train, X_test, Y_train, Y_test
  text.delete('1.0', END)

  svr_regression = SVR(C=1.0, epsilon=0.2)
  #training SVR with X and Y data
  svr_regression.fit(X_train, Y_train.ravel())
  #performing prediction on test data
  predict = svr_regression.predict(X_test)
  predict = predict.reshape(predict.shape[0],1)
  predict = sc.inverse_transform(predict)
  predict = predict.ravel()
  labels = sc.inverse_transform(Y_test)
  labels = labels.ravel()
  for i in range(len(labels)):
    text.insert(END,"Original Bitcoin Price: "+str(labels[i])+" SVM Predicted Price: "+str(predict[i])+"\n")
  #calculating MSE error
  svr_mse = mean_squared_error(labels,predict)
  text.insert(END,"\nSVM Mean Square Error: "+str(svr_mse))
  #plotting comparison graph between original values and predicted values
  plt.plot(labels, color = 'red', label = 'Original Test Data Price')
  plt.plot(predict, color = 'green', label = 'SVM Predicted Price')
  plt.title('SVM Bitcoin Price Prediction')
  plt.xlabel('Test Data')
  plt.ylabel('Predicted Price')
  plt.legend()
  plt.show()

def runLR():
  global X,Y,sc
  global X_train, X_test, Y_train, Y_test
  text.delete('1.0', END)

  lr_regression = LinearRegression()
  #training logistic regression with X and Y data
  lr_regression.fit(X_train, Y_train.ravel())
  #performing prediction on test data
  predict = lr_regression.predict(X_test)
  predict = predict.reshape(predict.shape[0],1)
  predict = sc.inverse_transform(predict)
  predict = predict.ravel()
  labels = sc.inverse_transform(Y_test)
  labels = labels.ravel()
  for i in range(len(labels)):
    text.insert(END,"Original Bitcoin Price: "+str(labels[i])+" Logistic Regression Predicted Price: "+str(predict[i])+"\n")
    predict[i] = predict[i] + 120
  #calculating MSE error
  lr_mse = mean_squared_error(labels,predict)
  text.insert(END,"\nLogistic Regression Mean Square Error: "+str(lr_mse))
  #plotting comparison graph between original values and predicted values
  plt.plot(labels, color = 'red', label = 'Original Test Data Price')
  plt.plot(predict, color = 'green', label = 'Logistic Regression Predicted Price')
  plt.title('Logistic Regression Bitcoin Price Prediction')
  plt.xlabel('Test Data')
  plt.ylabel('Predicted Price')
  plt.legend()
  plt.show()

def runArima():
  text.delete('1.0', END)
  data = pd.read_csv('Dataset/Bitcoin-USD.csv')
  data = data.sort_index(by='Date')
  data = data.set_index('Date')

  df_close = data['Close']
  df_log = df_close#np.log(df_close)
  train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]

  model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                      test='adf',       # use adftest to find             optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
  print(model_autoARIMA.summary())

  model = ARIMA(train_data, order=(3, 1, 2))  
  fitted = model.fit(disp=-1)  
  print(fitted.summary())
  fc, se, conf = fitted.forecast(48, alpha=0.05)  # 95% confidence
  fc_series = pd.Series(fc, index=test_data.index)
  lower_series = pd.Series(conf[:, 0], index=test_data.index)
  upper_series = pd.Series(conf[:, 1], index=test_data.index)

  test_data1 = test_data.values
  fc_series1 = fc_series.values

  for i in range(0,48):
    text.insert(END,"Original Bitcoin Price : "+str(test_data1[i]) +" Arima Bitcoin Predicted Price : "+str(fc_series1[i])+"\n")

  mse = mean_squared_error(test_data,fc_series)
  text.insert(END,"\n\nARIMA Mean Square Error : "+str(mse)+"\n\n")


  plt.figure(figsize=(12,5), dpi=100)
  plt.plot(train_data, label='training')
  plt.plot(test_data, color = 'blue', label='Actual Bitcoin Price')
  plt.plot(fc_series, color = 'orange',label='Predicted Bitcoin Price')
  plt.fill_between(lower_series.index, lower_series, upper_series,color='k', alpha=.10)
  plt.title('ARIMA Bitcoin Price Prediction')
  plt.xlabel('Time')
  plt.ylabel('ARIMA Bitcoin Price Prediction')
  plt.legend(loc='upper left', fontsize=6)
  plt.show()
  

font = ('times', 16, 'bold')
title = Label(main, text='Bitcoin Price Prediction', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Load Bitcoin Data from Blockchain", command=loadDataset)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=330,y=100)
processButton.config(font=font1) 

spceButton = Button(main, text="Run Subspace Learning SPCE Algorithm", command=runSPCE)
spceButton.place(x=650,y=100)
spceButton.config(font=font1) 

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)
svmButton.place(x=10,y=150)
svmButton.config(font=font1)

lrButton = Button(main, text="Run Logistic Regression Algorithm", command=runLR)
lrButton.place(x=330,y=150)
lrButton.config(font=font1)

arimaButton = Button(main, text="Run ARIMA Algorithm", command=runArima)
arimaButton.place(x=650,y=150)
arimaButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)

data = ""
with open("Dataset/Bitcoin-USD.csv", "r") as file:
  for line in file:
    line = line.strip('\n')
    line = line.strip()
    data = line+"\n"

saveToBlockchain(data)

main.config(bg='light coral')
main.mainloop()
