# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 18:26:00 2019

@author: prkumar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir("C:\\Users\\prkumar\\Desktop\\Prepaymentmodelscore")

dataset = pd.read_csv('140293.csv')

dataset.head()
dataset.columns

df = dataset.drop(['Unnamed: 0'],axis=1)

df1 = df[['QualFICO','termLengthMonths_Flag','RevolvingAvailablePercent',
          'Average_DollarDebitTransaction_Last3Months',
'QualADB', 'BigFee', 'SatisfactoryAccountsPerActiveTradeLines',
'DelinquenciesOver30Days_perTotaltrade',
            'Stdev_MonthsTradeWasCurrent_Last12Months',
            'NAICS_2','Sum_MonthsTradeWasActive',
            'Stdev_MonthsTradeWasCurrent_Last12Months_PerMonthTradeWasTracked',
            'InstallmentBalance','RevolvingBalance',
            'StDev_NumberOfCreditTransaction_Last3Months',
            'StDev_NumberOfCreditTransaction_Last6Months',
            'APR','StateCode',
       'CurrentBalance', 'Principal', 'NumberOfAdvances_ToFilter',
'SatisfactoryAccountsPerTotalTrade',
'QualRev30','DV_prepayment_75_per_Prepaid']]
df.columns

df.iloc[1:, ]

df.head()
df.columns
df.shape
X = df[df.columns[:-1]].values 
X.shape
y = df.iloc[:, -1].values
y.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) # Fit to data, then transform it.

X_test = sc.transform(X_test) 



#import keras
from keras.models import Sequential # to initialize ANN
from keras.layers import Dense # to create layers in ANN

# Initialising the ANN (we will define it as a sequence of layers or you can define a graph)
classifier = Sequential()

classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu', input_dim = 250))

# Add the second hidden layer (use prev layer as input nodes)
classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))

# Add the output layer (NOTE: if 3 encoded categories for dependent variable need 3 nodes and softmax activator func)
# choose sigmoid just like in logistic regression
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


classifier.fit(X_train, y_train, batch_size = 10, epochs = 40)

'''
from keras.models import model_from_json

# serialize model to JSON
model_json = classifier.to_json()

#json_file.write(model_json)

import json
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model.h5")
print("Saved model to disk")

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])




y_pred_train = loaded_model.predict(X_train) 

'''
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
y_pred_train = classifier.predict(X_train)
dz = pd.DataFrame(y_pred_train)
#dz = pd.DataFrame(pred)
dz['DV'] = y_train

dz.columns = ['Pred','DV']

import pandas as pd
dz['deciles'] =pd.qcut(dz['Pred'],10)
#pd.qcut(df_wob_train['Probability'],10).value_counts()

dz['Tier'] = pd.qcut(dz['Pred'],10,labels=range(1,11,1))


dz.to_csv("d1.csv")

y_pred = (y_pred_train > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred)

correct = cm[0][0] + cm[1][1]
incorrect = cm[0][1] + cm[1][0]

#y_pred_test = loaded_model.predict(X_test) 
y_pred_1 = (y_pred_train > 0.5)


import sklearn.metrics as metrics
fpr_train, tpr_train, threshold_train = metrics.roc_curve(y_train, y_pred)
roc_auc_train = metrics.auc(fpr_train, tpr_train)


threshold_train

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr_train, tpr_train, 'b', label = 'AUC = %0.2f' % roc_auc_train)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



classifier.compile(optimizer = 'RMSprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
y_pred_test = classifier.predict(X_test)
dtest = pd.DataFrame(y_pred_test)
#dz = pd.DataFrame(pred)
dtest['DV'] = y_test

dtest.columns = ['Pred','DV']

import pandas as pd
dtest['deciles'] =pd.qcut(dtest['Pred'],10)
#pd.qcut(df_wob_train['Probability'],10).value_counts()

dtest['Tier'] = pd.qcut(dtest['Pred'],10,labels=range(1,11,1))


dtest.to_csv("dtest.csv")

y_pred_t = (y_pred_test > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_t)

correct = cm[0][0] + cm[1][1]
incorrect = cm[0][1] + cm[1][0]

#y_pred_test = loaded_model.predict(X_test) 
y_pred_1 = (y_pred_train > 0.5)


import sklearn.metrics as metrics
fpr_test, tpr_test, threshold_test = metrics.roc_curve(y_test, y_pred_t)
roc_auc_test = metrics.auc(fpr_test, tpr_test)


threshold_test

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr_test, tpr_test, 'b', label = 'AUC = %0.2f' % roc_auc_test)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


