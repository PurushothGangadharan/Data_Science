#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
sp500=yf.Ticker("^GSPC")
sp500=sp500.history(period="max")
sp500


# sp500.index

# In[2]:


sp500.index


# In[3]:


sp500.plot.line(y="Close",use_index=True)


# In[4]:


del sp500['Dividends']
del sp500['Stock Splits']


# In[5]:


sp500["Tomorrow"]=sp500["Close"].shift(-1)
sp500


# In[6]:


sp500["Target"]=(sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500


# In[7]:


sp500 = sp500.loc["1990-01-01":].copy()


# In[8]:


sp500


# In[9]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close","Volume","Open","High","Low"]
model.fit(train[predictors],train["Target"])


# In[10]:


from sklearn.metrics import precision_score
preds=model.predict(test[predictors])
preds


# In[11]:


import pandas as pd
preds=pd.Series(preds, index=test.index)
precision_score(test["Target"],preds)


# In[12]:


combined = pd.concat([test["Target"], preds], axis=1)
combined.plot()


# In[13]:


def predict(train,test,predictors,model):
    model.fit(train[predictors],train["Target"])
    preds=model.predict(test[predictors])
    preds=pd.Series(preds, index=test.index, name="Predictions")
    combined= pd.concat([test["Target"],preds],axis=1)
    return combined
    


# In[14]:


def backtest(data,model,predictors,start=2500,step=250):
    all_predictions=[]
    
    for i in range(start,data.shape[0], step):
        train=data.iloc[0:i].copy()
        test=data.iloc[i:(i+step)].copy()
        predictions=predict(train,test,predictors,model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)    


# In[15]:


predictions = backtest(sp500,model,predictors)


# In[16]:


predictions["Predictions"].value_counts()


# In[17]:


precision_score(predictions["Target"],predictions["Predictions"])


# In[18]:


predictions["Target"].value_counts()/predictions.shape[0]


# In[19]:


horizons = [2,5,60,250,1000]
new_predictors=[]

for horizon in horizons:
    rolling_averages=sp500.rolling(horizon).mean()
    
    ratio_column=f"Close_Ratio_{horizon}"
    sp500[ratio_column]=sp500["Close"]/rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    sp500[trend_column]=sp500.shift(1).rolling(horizon).sum()["Target"]
    
    
    new_predictors += [ratio_column,trend_column]
    


# In[20]:


sp500 = sp500.dropna()


# In[21]:


sp500


# In[23]:


model = RandomForestClassifier(n_estimators=200,min_samples_split=50,random_state=1)


# In[24]:


def predict(train,test,predictors,model):
    model.fit(train[predictors],train["Target"])
    preds=model.predict_proba(test[predictors])[:,1]
    preds[preds>=.6]=1
    preds[preds<.6]=0
    preds=pd.Series(preds, index=test.index, name="Predictions")
    combined= pd.concat([test["Target"],preds],axis=1)
    return combined


# In[29]:


predictions=backtest(sp500,model,new_predictors)


# In[30]:


predictions["Predictions"].value_counts()


# In[31]:


precision_score(predictions["Target"],predictions["Predictions"])


# In[ ]:




