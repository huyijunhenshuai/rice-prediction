#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pylab as py


# In[2]:


df1 = pd.read_excel(r'C:\Users\TEMP\Desktop\水稻数据\NC数据.xls', '第1批')


# In[3]:


df1


# In[4]:


df2 = pd.read_excel(r'C:\Users\TEMP\Desktop\水稻数据\NC数据.xls', '第2批')


# In[5]:


df2


# In[6]:


df3 = pd.read_excel(r'C:\Users\TEMP\Desktop\水稻数据\NC数据.xls', '第3批')


# In[7]:


df3


# In[8]:


df4 = pd.read_excel(r'C:\Users\TEMP\Desktop\水稻数据\NC数据-产量.xlsx')


# In[9]:


df4


# In[10]:


data1=pd.merge(df1,df4,on='ID')
data1=data1.drop(['ID','分蘖数'],axis=1)
data1 = (data1 - data1.min())/(data1.max() - data1.min())
data1


# In[13]:


data2=pd.merge(df2,df4,on='ID')
data2


# In[14]:


data3=pd.merge(df3,df4,on='ID')
data3


# In[15]:


import matplotlib.pyplot as plt
data = df1.corr()[u'鲜重(g)']
plt.bar(range(len(data)), data)
plt.show()


# In[16]:


# 相对误差
def mean_relative_error(y_true, y_pred):
    import numpy as np
    relative_error = np.average(np.abs(y_true - y_pred) / y_true, axis=0)
    return relative_error


# In[ ]:





# In[17]:


df1.corr()[u'干重(g)']


# In[18]:


data1.corr()[u'IFD']


# In[19]:


df1.corr()[u'株高(cm)']


# In[20]:


df2.corr()[u'鲜重(g)']


# In[21]:


x=np.array(df1['SFD'])
# 左侧y轴：y1_recall
y=np.array(df1['干重(g)'])
plt.gcf().set_facecolor(np.ones(3) * 240/255)#设置背景色
fig, ax1 = plt.subplots() # 使用subplots()创建窗口
ax1.scatter(x, y, c='orangered') 
plt.legend(loc=2)
plt.legend(loc=4)
py.grid(True)  # 样式风格：网格型
ax1.set_xlabel('SFD',size=13)  
ax1.set_ylabel('dry weight(g))',size=13)
#plt.gcf().autofmt_xdate() # 自动适应刻度线密度，包括x轴，y轴
plt.show()


# In[22]:


x=np.array(data1['SFD'])
y=np.array(data1['干重(g)'])
plt.scatter(x,y,marker='o')
plt.show()


# # 随机森林

# In[23]:


from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,cross_validate
from sklearn import  metrics as mt


# In[24]:


data1_train,data1_test = train_test_split(df1,test_size=0.2,random_state=0)
x = data1_train[['PW','SA','SA/PH(V)*PW','IFD','SFD','G_g','f1','LD1']]  # 自变量
y = data1_train[['干重(g)']]  # 因变量
x.dropna(inplace=True)


# In[ ]:





# In[25]:


regressor1 = RandomForestRegressor(n_estimators=100,random_state=0)
regressor1.fit(x, y)


# In[26]:


x_test = data1_test[['PW','SA','SA/PH(V)*PW','IFD','SFD','G_g','f1','LD1']]  
y_test = data1_test[['干重(g)']]


# In[27]:


predictY = regressor1.predict(x_test)  # 预测值
predictY


# In[28]:


r2 = mt.r2_score(y_test,predictY)
r2


# In[ ]:





# In[ ]:





# In[31]:


x_1=np.arange(0,85)
plt.gcf().set_facecolor(np.ones(3) * 240/255)#设置背景色
fig, ax1 = plt.subplots() # 使用subplots()创建窗口
plt.plot(x_1,predictY,label="predict",linestyle='dashed')
plt.plot(x_1,y_test,label="test")
plt.legend(loc=2)
plt.legend(loc=4)
py.grid(True)  # 样式风格：网格型
ax1.set_xlabel('Test set of rice',size=13)  
ax1.set_ylabel('Dry weight',size=13)
#plt.gcf().autofmt_xdate() # 自动适应刻度线密度，包括x轴，y轴
plt.show()


# In[32]:


data1_train,data1_test = train_test_split(df1,test_size=0.2,random_state=0)
x = data1_train[['PW','SA','SA/PH(V)*PW','IFD','SFD','G_g','f1','LD1']]  # 自变量
y = data1_train[['鲜重(g)']]  # 因变量
x.dropna(inplace=True)
regressor1 = RandomForestRegressor(n_estimators=50,random_state=0)
regressor1.fit(x, y)
x_test = data1_test[['PW','SA','SA/PH(V)*PW','IFD','SFD','G_g','f1','LD1']]  
y_test = data1_test[['鲜重(g)']]
predictY = regressor1.predict(x_test)  # 预测值
predictY
r2 = mt.r2_score(y_test,predictY)
r2


# In[ ]:





# In[34]:


superpa = []
for i in range(200):
    data1_train,data1_test = train_test_split(df1,test_size=0.2,random_state=0)
    x = data1_train[['PW','SA','SA/PH(V)*PW','IFD','SFD','G_g','f1','LD1']]  # 自变量
    y = data1_train[['鲜重(g)']]  # 因变量
    x.dropna(inplace=True)
    regressor1 = RandomForestRegressor(n_estimators=i+1,random_state=0)
    regressor1.fit(x, y)
    x_test = data1_test[['PW','SA','SA/PH(V)*PW','IFD','SFD','G_g','f1','LD1']]  
    y_test = data1_test[['鲜重(g)']]
    predictY = regressor1.predict(x_test)
    r2=mt.r2_score(y_test,predictY)
    superpa.append(r2)


# In[35]:


x_1=np.arange(1,201)
plt.gcf().set_facecolor(np.ones(3) * 240/255)#设置背景色
fig, ax1 = plt.subplots() # 使用subplots()创建窗口
plt.plot(x_1,superpa)
plt.legend(loc=4)
py.grid(True)  # 样式风格：网格型
ax1.set_xlabel('N_ESTIMATORS',size=13)  
ax1.set_ylabel('R²',size=13)
#plt.gcf().autofmt_xdate() # 自动适应刻度线密度，包括x轴，y轴
plt.show()


# In[36]:


data1_train,data1_test = train_test_split(df1,test_size=0.2,random_state=0)
x = data1_train[['PW','PH(V)','PH','SFD','G_g','f1','f2','LD2']]  # 自变量
y = data1_train[['株高(cm)']]  # 因变量
x.dropna(inplace=True)
regressor3 = RandomForestRegressor(n_estimators=50,random_state=0)
regressor3.fit(x, y)
x_test = data1_test[['PW','PH(V)','PH','SFD','G_g','f1','f2','LD2']]  
y_test = data1_test[['株高(cm)']]
predictY = regressor3.predict(x_test)  # 预测值
predictY
r2 = mt.r2_score(y_test,predictY)
r2


# In[ ]:





# In[38]:


data1_train,data1_test = train_test_split(data1,test_size=0.2,random_state=0)
x = data1_train[['IFD','SFD','总粒数','实粒数','结实率','千粒重']]  # 自变量
y = data1_train[['重量']]  # 因变量
x.dropna(inplace=True)
regressor = RandomForestRegressor(n_estimators=50,random_state=0)
regressor.fit(x, y)
x_test = data1_test[['IFD','SFD','总粒数','实粒数','结实率','千粒重']]  
y_test = data1_test[['重量']]
predictY = regressor.predict(x_test)  # 预测值
predictY
r2 = mt.r2_score(y_test,predictY)
r2


# In[ ]:





# # 支持向量机

# In[39]:


from sklearn.svm import SVR


# In[40]:


data1_train,data1_test = train_test_split(data1,test_size=0.2,random_state=0)
x = data1_train[['PW','SA','SA/PH(V)*PW','IFD','SFD','G_g','f1','LD1']]  # 自变量
y = data1_train[['干重(g)']]  # 因变量
x.dropna(inplace=True)
svr1 = SVR(kernel ='rbf',gamma =4,C =100)
svr1.fit(x, y)
x_test = data1_test[['PW','SA','SA/PH(V)*PW','IFD','SFD','G_g','f1','LD1']]  
y_test = data1_test[['干重(g)']]
predictY = svr1.predict(x_test)  # 预测值
predictY
r2 = mt.r2_score(y_test,predictY)
r2


# In[ ]:





# In[ ]:





# In[43]:


x_1=np.arange(0,84)
plt.gcf().set_facecolor(np.ones(3) * 240/255)#设置背景色
fig, ax1 = plt.subplots() # 使用subplots()创建窗口
plt.plot(x_1,predictY,label="predict",linestyle='dashed')
plt.plot(x_1,y_test,label="test")
plt.legend(loc=2)
plt.legend(loc=4)
py.grid(True)  # 样式风格：网格型
ax1.set_xlabel('Test set of rice',size=13)  
ax1.set_ylabel('Dry weight',size=13)
#plt.gcf().autofmt_xdate() # 自动适应刻度线密度，包括x轴，y轴
plt.show()


# In[44]:


data1_train,data1_test = train_test_split(data1,test_size=0.2,random_state=0)
x = data1_train[['PW','SA','SA/PH(V)*PW','IFD','SFD','G_g','f1','LD1']]  # 自变量
y = data1_train[['鲜重(g)']]  # 因变量
x.dropna(inplace=True)
svr2 = SVR(kernel ='rbf',gamma =4,C =100)
svr2.fit(x, y)
x_test = data1_test[['PW','SA','SA/PH(V)*PW','IFD','SFD','G_g','f1','LD1']]  
y_test = data1_test[['鲜重(g)']]
predictY = svr2.predict(x_test)  # 预测值
predictY
r2 = mt.r2_score(y_test,predictY)
r2


# In[ ]:





# In[46]:


data1_train,data1_test = train_test_split(data1,test_size=0.2,random_state=0)
x = data1_train[['PW','PH(V)','PH','SFD','G_g','f1','f2','LD2']]  # 自变量
y = data1_train[['株高(cm)']]  # 因变量
x.dropna(inplace=True)
svr3 = SVR(kernel ='rbf',C =1)
svr3.fit(x, y)
x_test = data1_test[['PW','PH(V)','PH','SFD','G_g','f1','f2','LD2']]  
y_test = data1_test[['株高(cm)']]
predictY = svr3.predict(x_test)  # 预测值
predictY
r2 = mt.r2_score(y_test,predictY)
r2


# In[ ]:





# In[48]:


data1_train,data1_test = train_test_split(data1,test_size=0.2,random_state=0)
x = data1_train[['IFD','SFD','总粒数','实粒数','结实率','千粒重']]  # 自变量
y = data1_train[['重量']]  # 因变量
x.dropna(inplace=True)
svr = SVR(kernel='linear')
svr.fit(x, y)
x_test = data1_test[['IFD','SFD','总粒数','实粒数','结实率','千粒重']]  
y_test = data1_test[['重量']]
predictY = svr.predict(x_test)  # 预测值
predictY
r2 = mt.r2_score(y_test,predictY)
r2


# In[ ]:





# # 线性回归

# In[49]:


from sklearn import linear_model


# In[1]:


data1_train,data1_test = train_test_split(df1,test_size=0.2,random_state=0)
x = data1_train[['SA','IFD','SFD','G_g','f1','LD1']]  # 自变量
y = data1_train[['干重(g)']]  # 因变量
x.dropna(inplace=True)
reg1= linear_model.LinearRegression()
reg1.fit(x, y)
x_test = data1_test[['SA','IFD','SFD','G_g','f1','LD1']]  
y_test = data1_test[['干重(g)']]
predictY = reg1.predict(x_test)  # 预测值
predictY
r2 = mt.r2_score(y_test,predictY)
print(r2)


# In[ ]:





# In[ ]:





# In[53]:


x_1=np.arange(0,85)
plt.gcf().set_facecolor(np.ones(3) * 240/255)#设置背景色
fig, ax1 = plt.subplots() # 使用subplots()创建窗口
plt.plot(x_1,predictY,label="predict",linestyle='dashed')
plt.plot(x_1,y_test,label="test")
plt.legend(loc=2)
plt.legend(loc=4)
py.grid(True)  # 样式风格：网格型
ax1.set_xlabel('Test set of rice',size=13)  
ax1.set_ylabel('Dry weight',size=13)
#plt.gcf().autofmt_xdate() # 自动适应刻度线密度，包括x轴，y轴
plt.show()


# In[54]:


data1_train,data1_test = train_test_split(df1,test_size=0.2,random_state=0)
x = data1_train[['PW','SA','SA/PH(V)*PW','IFD','SFD','G_g','f1','LD1']]  # 自变量
y = data1_train[['鲜重(g)']]  # 因变量
x.dropna(inplace=True)
reg2= linear_model.LinearRegression()
reg2.fit(x, y)
x_test = data1_test[['PW','SA','SA/PH(V)*PW','IFD','SFD','G_g','f1','LD1']]  
y_test = data1_test[['鲜重(g)']]
predictY = reg2.predict(x_test)  # 预测值
predictY
r2 = mt.r2_score(y_test,predictY)
r2


# In[ ]:





# In[56]:


data1_train,data1_test = train_test_split(df1,test_size=0.2,random_state=0)
x = data1_train[['PW','PH(V)','PH','SFD','G_g','f1','f2','LD2']]  # 自变量
y = data1_train[['株高(cm)']]  # 因变量
x.dropna(inplace=True)
reg2= linear_model.LinearRegression()
reg2.fit(x, y)
x_test = data1_test[['PW','PH(V)','PH','SFD','G_g','f1','f2','LD2']]  
y_test = data1_test[['株高(cm)']]
predictY = reg2.predict(x_test)  # 预测值
predictY
r2 = mt.r2_score(y_test,predictY)
r2


# In[ ]:





# In[58]:


data1_train,data1_test = train_test_split(data1,test_size=0.2,random_state=0)
x = data1_train[['IFD','SFD','总粒数','实粒数','结实率','千粒重']]  # 自变量
y = data1_train[['重量']]  # 因变量
x.dropna(inplace=True)
reg = linear_model.LinearRegression()
reg.fit(x, y)
x_test = data1_test[['IFD','SFD','总粒数','实粒数','结实率','千粒重']]  
y_test = data1_test[['重量']]
predictY = reg.predict(x_test)  # 预测值
predictY
r2 = mt.scorer.r2_score(y_test,predictY)
r2


# 

# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




