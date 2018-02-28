
# In[1]:

import os
mypath = os.getcwd()
os.listdir(mypath)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm
import numpy,pylab; from sklearn.svm import SVC



# In[3]:

from sklearn.cross_validation import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[4]:

train =  pd.read_csv("train_neural.csv")
test= pd.read_csv("test_neural.csv")
print(train.columns.values)
print(train.shape)
print(test.columns.values)

train.shape[0]


# In[5]:

print(test.head)
print(train.head)


# In[6]:

print ("train dataset has "+ str(train.shape[0]) + " rows and " + str(train.shape[1]) + " columns.")

print ("test dataset has "+ str(test.shape[0]) + " rows and " + str(test.shape[1]) + " columns.")


# In[7]:

print(type(train))
print(train.iloc[:,14])

print(type(test))
print(test.iloc[:,14])


# In[8]:

x_train=train.iloc[:,:14]
y_train=train.iloc[:,14]


# In[9]:

x_test=test.iloc[:,:14]
y_test=test.iloc[:,14]


# In[10]:

x_test[20:21]


# In[11]:

x_test.sample(n=1)


# In[12]:

clf = svm.SVC(kernel='linear', C = 1.0)


# In[13]:

clf.fit(x_train,y_train)


# In[14]:

eeg_out=clf.predict(x_train[16:17])


# In[15]:

eeg_out_sample=clf.predict(x_test.sample(n=1))


# In[16]:

print(eeg_out)


# In[16]:

print(eeg_out_sample)