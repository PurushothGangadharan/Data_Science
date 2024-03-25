#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.common.keys import Keys
from textblob import TextBlob
from bs4 import BeautifulSoup
from selenium import webdriver


# In[2]:


driver=webdriver.Chrome()
#driver.get("https://www.youtube.com/watch?v=m9Z2u8yajXo&list=WL&index=22")
driver.get("https://www.youtube.com/watch?v=RQ20Y1OZuH0")
SCROLL_PAUSE_TIME=2
last_height=driver.execute_script("return document.documentElement.scrollHeight")

while True:
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    time.sleep(SCROLL_PAUSE_TIME)
    new_height = driver.execute_script("return document.documentElement.scrollHeight")
    if new_height==last_height:
        break
    last_height =new_height


# In[3]:


comments_elements = driver.find_elements(By.CSS_SELECTOR, "#content-text")
comments = [comment.text for comment in comments_elements]

driver.quit()


# In[4]:


df=pd.DataFrame({'Comments':comments})
print(df)
df.to_csv('comments_dataframe.csv', index=False)
df


# In[5]:


df['Time']=df['Comments'].str.extract(r'(\d{1,2}:\d{2})')
df['Comment'] = df ['Comments'].str.replace(r'\d{1,2}:\d{2}','').str.strip()


# In[7]:


df=df[['Comment']]
print(df)


# In[8]:


import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string


# In[9]:


df['Comment'] = df['Comment'].apply(lambda x: re.sub(r'[^\w\s]','',x))


# In[10]:


df['Tokenized_Comments']=df['Comment'].apply(word_tokenize)


# In[11]:


df['Tokenized_Comments'] = df['Tokenized_Comments'].apply(lambda x:[word.lower() for word in x])


# In[12]:


stop_words= set(stopwords.words('english'))
df['Tokenized_Comments']=df['Tokenized_Comments'].apply(lambda x:[word for word in x if word not in stop_words])


# In[13]:


df['Tokenized_Comments']=df['Tokenized_Comments'].apply(lambda x:[word for word in x if word.isalpha()])


# In[14]:


df


# In[17]:


df['Processed_Comments']= df['Tokenized_Comments'].apply(lambda x: ' '.join(x))


# In[18]:


df


# In[19]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[30]:


all_comments = ''.join(df['Comment'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_comments)


# In[31]:


plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("YouTube Comments Word Cloud")
plt.axis('off')
plt.show()


# In[32]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
import plotly.express as px

analyzer = SentimentIntensityAnalyzer()


# In[33]:


df['Sentiment_Scores'] = df['Processed_Comments'].apply(lambda x: analyzer.polarity_scores(x))
df['Compound_Score'] = df['Sentiment_Scores'].apply(lambda x: x['compound'])
df['Sentiment'] = df['Compound_Score'].apply(lambda score: 'positive' if score > 0.05 else ('negative' if score < 0.05 else 'neutral'))
fig = px.histogram(df, x='Sentiment',color='Sentiment', title='Sentiment Distribution',
                  labels={'Sentiment':'Sentiment Category','count':'Frequency'},
                  color_discrete_map={'postive':'green','negative':"red",'neutral':'blue'})
fig.show()


# In[34]:


# Print positive comments and their scores
positive_comments = df[df['Sentiment'] == 'positive']
print("Positive Comments:")
for index, row in positive_comments.iterrows():
    print(f"Comment: {row['Processed_Comments']}")
    print(f"Sentiment Score: {row['Compound_Score']}")
    print()

# Print negative comments and their scores
negative_comments = df[df['Sentiment'] == 'negative']
print("Negative Comments:")
for index, row in negative_comments.iterrows():
    print(f"Comment: {row['Processed_Comments']}")
    print(f"Sentiment Score: {row['Compound_Score']}")
    print()


# In[ ]:




