#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies=pd.read_csv('tmdb_5000_movies.csv',low_memory=False)
credits=pd.read_csv('tmdb_5000_credits.csv')


# movies.head()

# In[3]:


movies.head(1)

credits

# In[4]:


movies=movies.merge(credits,on='title')


# ### imp columns-> genre,id,keywords,original language,title,overview,cast ,crew

# In[5]:


movies.head()


# In[6]:


movies=movies[['genres','id','original_language','overview','title','cast','keywords','crew']]


# In[7]:


movies.isnull().sum()


# In[8]:


movies.dropna(inplace=True)


# In[9]:


movies.isnull().sum()


# In[10]:


movies.duplicated().sum()


# In[11]:


movies=movies.drop_duplicates()


# In[12]:


movies.duplicated().sum()


# In[13]:


movies.iloc[0].genres


# In[14]:


def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l


# In[15]:


import ast
movies['genres']=movies['genres'].apply(convert)


# In[16]:


movies['keywords']=movies['keywords'].apply(convert)
movies.head()


# In[17]:


def convert2(obj):
    counter=0
    l=[]
    for i in ast.literal_eval(obj):
        if counter != 3:
            l.append(i['name'])
            counter+=1
        else:
            break
    return l


# In[18]:


movies['cast']=movies['cast'].apply(convert2)


# In[19]:


def fetch_director(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l


# 

# In[20]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[21]:


movies.head()


# In[22]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[23]:


movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])


# In[24]:


movies['tags']=movies['keywords']+movies['cast']+movies['crew']+movies['genres']+movies['overview']


# In[25]:


movies['tags'][0]


# In[26]:


new_df=movies[['id','title','tags']]


# In[27]:


new_df


# In[28]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[29]:


new_df['tags'][0]


# In[30]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[31]:


get_ipython().system('pip install nltk')


# In[32]:


new_df.head()


# In[33]:


import nltk


# In[34]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[35]:


def stem(text):
    l=[]
    for i in text.split():
        l.append(ps.stem(i))
    return " ".join(l)


# In[36]:


new_df['tags']=new_df['tags'].apply(stem)


# In[37]:


get_ipython().system('pip install scikit-learn')


# In[38]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=4000,stop_words='english')


# In[39]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[40]:


vectors.shape


# In[41]:


vectors.dtype


# In[42]:


from sklearn.metrics.pairwise import cosine_similarity


# In[43]:


similarity=cosine_similarity(vectors)


# In[44]:


similarity[1]


# In[45]:


def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movie_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)


# In[46]:


recommend('Titanic')


# In[47]:


import pickle


# In[48]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[49]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




